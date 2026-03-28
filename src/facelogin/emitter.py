"""IR emitter activation via UVC ioctl with multi-strategy fallback.

Strategies tried in order:
1. Direct UVC extension-unit ioctl (if config has unit/selector/control_data)
2. Parse linux-enable-ir-emitter TOML config to extract UVC parameters
3. CLI fallback: ``linux-enable-ir-emitter run``
4. Skip — log a warning and continue (emitter may not be present)

The UVC ioctl speaks the ``UVCIOC_CTRL_QUERY`` interface from the Linux
UVC driver.  The query struct is packed as::

    struct uvc_xu_control_query {
        __u8  unit;
        __u8  selector;
        __u8  query;        // UVC_SET_CUR or UVC_GET_CUR
        __u8  pad;
        __u16 size;
        __u16 pad2;
        __u64 data;         // userspace pointer to the control buffer
    };

Total: 16 bytes, packed with ``=BBBxHxxQ``.
"""

from __future__ import annotations

import ctypes
import fcntl
import logging
import os
import struct
import subprocess
from pathlib import Path
from typing import Optional

from .config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UVC ioctl constants
# ---------------------------------------------------------------------------

UVCIOC_CTRL_QUERY: int = 0xC0107521
UVC_SET_CUR: int = 0x01
UVC_GET_CUR: int = 0x81

# Paths searched for linux-enable-ir-emitter TOML configs
_EMITTER_TOML_SEARCH_PATHS = [
    os.path.expanduser("~/.config/linux-enable-ir-emitter.toml"),
    "/root/.config/linux-enable-ir-emitter.toml",
]

_EMITTER_CLI_BINARY = "linux-enable-ir-emitter"


# ---------------------------------------------------------------------------
# Hex string ↔ bytes helpers
# ---------------------------------------------------------------------------

def _parse_control_data(raw: str) -> bytes:
    """Parse a hex control_data string like ``"0x01 0x03 0x02"`` into bytes.

    Accepts:
      - Space-separated hex with 0x prefix: ``"0x01 0x03 0x02 0x00"``
      - Comma-separated: ``"0x01, 0x03, 0x02"``
      - Plain hex string: ``"01030200"``
    """
    raw = raw.strip()
    if not raw:
        return b""

    # "0x01 0x03 ..." or "0x01, 0x03, ..."
    if "0x" in raw.lower():
        tokens = raw.replace(",", " ").split()
        return bytes(int(t, 16) for t in tokens)

    # Plain hex string "0103020000000000"
    return bytes.fromhex(raw)


# ---------------------------------------------------------------------------
# Low-level UVC helpers
# ---------------------------------------------------------------------------

def _uvc_query(
    device: str,
    unit: int,
    selector: int,
    query_type: int,
    data: Optional[bytes] = None,
    size: Optional[int] = None,
) -> bytes:
    """Send a single UVC extension-unit control query via ioctl.

    Parameters
    ----------
    device:
        V4L2 device path, e.g. ``/dev/video2``.
    unit:
        Extension-unit ID on the device.
    selector:
        Control selector within the extension unit.
    query_type:
        ``UVC_SET_CUR`` (0x01) to write, ``UVC_GET_CUR`` (0x81) to read.
    data:
        Payload bytes for SET_CUR.  For GET_CUR this is ignored (a
        zero-filled buffer of *size* bytes is used).
    size:
        Buffer size.  Defaults to ``len(data)`` when *data* is given.

    Returns
    -------
    bytes
        The control buffer after the ioctl completes.
    """
    if size is None:
        if data is not None:
            size = len(data)
        else:
            raise ValueError("Must provide data or size")

    buf = ctypes.create_string_buffer(data if data else bytes(size), size)
    ptr = ctypes.addressof(buf)

    # Pack the uvc_xu_control_query struct
    query_struct = struct.pack("=BBBxHxxQ", unit, selector, query_type, size, ptr)

    fd = os.open(device, os.O_RDWR)
    try:
        fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, query_struct)
        return bytes(buf)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------

def check_emitter_status(device: Optional[str] = None) -> bool:
    """Read the current UVC control value and return True if emitter is on.

    Uses GET_CUR to read the control register and compares against the
    configured ``control_data``.  Returns False on any ioctl failure
    (device missing, permissions, no config, etc.).
    """
    config = get_config()
    device = device or config.camera.device
    if not device:
        logger.debug("check_emitter_status: no device specified")
        return False

    ecfg = config.emitter
    if not ecfg.unit or not ecfg.selector or not ecfg.control_data:
        logger.debug("check_emitter_status: incomplete emitter config")
        return False

    expected = _parse_control_data(ecfg.control_data)
    if not expected:
        return False

    try:
        current = _uvc_query(
            device,
            unit=ecfg.unit,
            selector=ecfg.selector,
            query_type=UVC_GET_CUR,
            size=len(expected),
        )
        is_on = current == expected
        logger.debug(
            "Emitter status on %s: %s (current=%s, expected=%s)",
            device, "ON" if is_on else "OFF",
            current.hex(), expected.hex(),
        )
        return is_on
    except OSError as exc:
        logger.debug("check_emitter_status ioctl failed on %s: %s", device, exc)
        return False


# ---------------------------------------------------------------------------
# TOML config parsing (linux-enable-ir-emitter)
# ---------------------------------------------------------------------------

def _parse_emitter_toml(path: str) -> Optional[tuple[int, int, bytes]]:
    """Extract (unit, selector, control_data) from a linux-enable-ir-emitter TOML.

    The TOML typically contains sections like::

        [emitter0]
        unit = 4
        selector = 6
        control = [1, 3, 2, 0, 0, 0, 0, 0, 0]

    We parse it minimally without requiring a TOML library — the format
    is simple enough for line-by-line extraction.
    """
    try:
        text = Path(path).read_text()
    except OSError:
        return None

    unit: Optional[int] = None
    selector: Optional[int] = None
    control: Optional[bytes] = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("unit"):
            try:
                unit = int(stripped.split("=", 1)[1].strip())
            except (IndexError, ValueError):
                pass

        elif stripped.startswith("selector"):
            try:
                selector = int(stripped.split("=", 1)[1].strip())
            except (IndexError, ValueError):
                pass

        elif stripped.startswith("control"):
            try:
                raw = stripped.split("=", 1)[1].strip()
                # Handle both [1, 3, 2] and "1, 3, 2" formats
                raw = raw.strip("[]")
                values = [int(v.strip()) for v in raw.split(",") if v.strip()]
                control = bytes(values)
            except (IndexError, ValueError):
                pass

    if unit is not None and selector is not None and control is not None:
        logger.info("Parsed emitter TOML %s: unit=%d selector=%d data=%s",
                     path, unit, selector, control.hex())
        return unit, selector, control

    logger.debug("Incomplete emitter config in %s", path)
    return None


# ---------------------------------------------------------------------------
# Activation strategies
# ---------------------------------------------------------------------------

def _try_direct_ioctl(device: str) -> bool:
    """Strategy 1: direct UVC ioctl using config values."""
    config = get_config()
    ecfg = config.emitter
    if not ecfg.unit or not ecfg.selector or not ecfg.control_data:
        logger.debug("Direct ioctl skipped — incomplete config (unit=%s selector=%s data=%r)",
                      ecfg.unit, ecfg.selector, ecfg.control_data)
        return False

    control_bytes = _parse_control_data(ecfg.control_data)
    if not control_bytes:
        return False

    try:
        _uvc_query(
            device,
            unit=ecfg.unit,
            selector=ecfg.selector,
            query_type=UVC_SET_CUR,
            data=control_bytes,
        )
        logger.info("Emitter activated via direct ioctl on %s", device)
        return True
    except OSError as exc:
        logger.debug("Direct ioctl failed on %s: %s", device, exc)
        return False


def _try_toml_ioctl(device: str) -> bool:
    """Strategy 2: parse linux-enable-ir-emitter TOML, then ioctl."""
    for toml_path in _EMITTER_TOML_SEARCH_PATHS:
        params = _parse_emitter_toml(toml_path)
        if params is None:
            continue
        unit, selector, control_data = params
        try:
            _uvc_query(
                device,
                unit=unit,
                selector=selector,
                query_type=UVC_SET_CUR,
                data=control_data,
            )
            logger.info("Emitter activated via TOML config %s on %s", toml_path, device)
            return True
        except OSError as exc:
            logger.debug("TOML-based ioctl failed (%s) on %s: %s", toml_path, device, exc)
    return False


def _try_cli(device: str) -> bool:
    """Strategy 3: invoke ``linux-enable-ir-emitter run`` CLI."""
    # Try with explicit config files first
    for toml_path in _EMITTER_TOML_SEARCH_PATHS:
        if not Path(toml_path).exists():
            continue
        try:
            result = subprocess.run(
                [_EMITTER_CLI_BINARY, "run", "--config", toml_path],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                logger.info("Emitter activated via CLI with config %s", toml_path)
                return True
            logger.debug("CLI with config %s exited %d: %s",
                         toml_path, result.returncode, result.stderr.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.debug("CLI with config %s failed: %s", toml_path, exc)

    # Bare invocation (uses tool's default config discovery)
    try:
        result = subprocess.run(
            [_EMITTER_CLI_BINARY, "run"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            logger.info("Emitter activated via bare CLI")
            return True
        logger.debug("Bare CLI exited %d: %s", result.returncode, result.stderr.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.debug("Bare CLI failed: %s", exc)

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def activate_emitter(device: Optional[str] = None) -> bool:
    """Enable the IR emitter, skipping if already on.

    Tries strategies in order: direct ioctl → TOML-parsed ioctl → CLI
    fallback.  Returns True if the emitter was activated (or was already
    on), False if all strategies failed.

    Parameters
    ----------
    device:
        V4L2 device path.  Defaults to ``config.camera.device``.
    """
    config = get_config()

    if not config.emitter.enabled:
        logger.debug("Emitter disabled in config — skipping")
        return True  # not a failure, just opted out

    device = device or config.camera.device
    if not device:
        logger.warning("activate_emitter: no device — set camera.device in config or pass device=")
        return False

    # Short-circuit if already on
    if check_emitter_status(device):
        logger.info("Emitter already on for %s — skipping activation", device)
        return True

    # Strategy 1: direct ioctl from config
    if _try_direct_ioctl(device):
        return True

    # Strategy 2: TOML config → ioctl
    if _try_toml_ioctl(device):
        return True

    # Strategy 3: CLI fallback
    if _try_cli(device):
        return True

    # Strategy 4: give up gracefully
    logger.warning(
        "All emitter activation strategies failed for %s. "
        "IR frames may be dark. Install linux-enable-ir-emitter or "
        "configure emitter UVC parameters in config.",
        device,
    )
    return False
