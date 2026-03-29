"""V4L2 camera auto-detection and frame reading.

Enumerates /dev/video* devices via v4l2-ctl, classifies them as IR or
RGB by their supported pixel formats, and provides a Camera class that
handles the quirks of IR sensors (alternating emitter-on/off frames).
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import get_config

logger = logging.getLogger(__name__)

# Pixel formats that indicate an IR (greyscale) sensor
_IR_FORMATS = frozenset({"GREY", "Y8", "Y10", "Y16", "Y10B", "Y12"})
# Pixel formats that indicate an RGB (colour) sensor
_RGB_FORMATS = frozenset({"YUYV", "MJPG", "NV12", "YU12", "NV21", "RGB3", "BGR3"})

# Defaults when config doesn't specify
_DEFAULT_IR_FRAME_ATTEMPTS = 15
_DEFAULT_RGB_FRAME_ATTEMPTS = 3
_DEFAULT_V4L2_BUFFER_SIZE = 1


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraInfo:
    """Metadata for a single V4L2 video-capture device."""

    device: str          # e.g. "/dev/video2"
    type: str            # "ir" or "rgb"
    formats: list[str]   # e.g. ["GREY", "Y8"]
    name: str            # human-readable card name


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _device_name(device: str) -> str:
    """Read the card name from sysfs, falling back to the device path."""
    idx = device.rsplit("video", 1)[-1]
    name_path = Path(f"/sys/class/video4linux/video{idx}/name")
    try:
        return name_path.read_text().strip()
    except OSError:
        return device


def _has_video_capture(device: str) -> bool:
    """Return True if the device supports Video Capture (not metadata-only)."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--all"],
            capture_output=True, text=True, timeout=5,
        )
        return "Video Capture" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _parse_formats(device: str) -> list[str]:
    """Run v4l2-ctl --list-formats-ext and return the FourCC codes."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    # Lines look like:  [0]: 'GREY' (8-bit Greyscale)
    fourcc_re = re.compile(r"'(\w+)'")
    formats: list[str] = []
    for line in result.stdout.splitlines():
        m = fourcc_re.search(line)
        if m:
            fmt = m.group(1)
            if fmt not in formats:
                formats.append(fmt)
    return formats


def _classify(formats: list[str]) -> Optional[str]:
    """Classify a device as 'ir', 'rgb', or None (unknown/skip)."""
    fmt_set = {f.upper() for f in formats}
    has_ir = bool(fmt_set & _IR_FORMATS)
    has_rgb = bool(fmt_set & _RGB_FORMATS)

    if has_ir and not has_rgb:
        return "ir"
    if has_rgb and not has_ir:
        return "rgb"
    if has_ir and has_rgb:
        # Hybrid — prefer IR classification (rare in practice)
        return "ir"
    return None


def detect_cameras() -> list[CameraInfo]:
    """Enumerate all V4L2 video-capture devices and classify them.

    Skips metadata-only nodes (no Video Capture capability) and devices
    whose pixel formats don't match any known IR or RGB set.
    """
    cameras: list[CameraInfo] = []

    dev_nodes = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
    if not dev_nodes:
        logger.warning("No /dev/video* devices found")
        return cameras

    for dev_path in dev_nodes:
        device = str(dev_path)

        if not _has_video_capture(device):
            logger.debug("Skipping %s — no Video Capture capability", device)
            continue

        formats = _parse_formats(device)
        if not formats:
            logger.debug("Skipping %s — no parseable formats", device)
            continue

        cam_type = _classify(formats)
        if cam_type is None:
            logger.debug("Skipping %s — unrecognised formats: %s", device, formats)
            continue

        name = _device_name(device)
        info = CameraInfo(device=device, type=cam_type, formats=formats, name=name)
        cameras.append(info)
        logger.info("Detected %s camera: %s (%s) formats=%s", cam_type, device, name, formats)

    return cameras


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera:
    """Thin wrapper around cv2.VideoCapture with IR bright-frame logic.

    For IR cameras the sensor alternates emitter-on / emitter-off frames.
    We skip dark frames (mean pixel value ≤ brightness_threshold) and
    convert single-channel images to BGR so downstream code always gets
    a 3-channel numpy array.
    """

    def __init__(
        self,
        device: str,
        cam_type: str,
        brightness_threshold: float = 20.0,
    ) -> None:
        self.device = device
        self.cam_type = cam_type
        self.brightness_threshold = brightness_threshold
        self._cap: Optional[cv2.VideoCapture] = None

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> np.ndarray:
        """Open the device and return the first usable frame.

        Raises RuntimeError if no bright frame is captured within the
        configured attempt budget.
        """
        index = self._device_index()
        self._cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, _DEFAULT_V4L2_BUFFER_SIZE)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open {self.device} (index {index})")

        if self.cam_type == "ir":
            return self._read_bright_frame(_DEFAULT_IR_FRAME_ATTEMPTS)
        return self._read_frame(_DEFAULT_RGB_FRAME_ATTEMPTS)

    def read(self) -> np.ndarray:
        """Read the next usable frame from the already-open device."""
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera not open — call open() first")

        if self.cam_type == "ir":
            return self._read_bright_frame(_DEFAULT_IR_FRAME_ATTEMPTS)
        return self._read_frame(_DEFAULT_RGB_FRAME_ATTEMPTS)

    def release(self) -> None:
        """Release the underlying VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.release()

    # -- class-level factory -------------------------------------------------

    @classmethod
    def auto_detect(cls) -> "Camera":
        """Pick the best camera using config + detection.

        Priority:
        1. config.camera.device if set and present
        2. First IR camera found
        3. First RGB camera found
        """
        config = get_config()
        cameras = detect_cameras()
        if not cameras:
            raise RuntimeError("No usable V4L2 cameras detected")

        # Check config for a preferred device
        pref = config.camera.device
        if pref:
            for cam in cameras:
                if cam.device == pref:
                    cam_type = config.camera.type or cam.type
                    threshold = config.camera.ir_brightness_threshold
                    logger.info("Using configured device %s (type=%s)", pref, cam_type)
                    return cls(cam.device, cam_type, brightness_threshold=threshold)
            logger.warning("Configured device %s not found, falling back to auto", pref)

        # Prefer IR over RGB
        threshold = config.camera.ir_brightness_threshold
        ir_cams = [c for c in cameras if c.type == "ir"]
        if ir_cams:
            chosen = ir_cams[0]
            logger.info("Auto-selected IR camera %s (%s)", chosen.device, chosen.name)
            return cls(chosen.device, chosen.type, brightness_threshold=threshold)

        chosen = cameras[0]
        logger.info("Auto-selected RGB camera %s (%s)", chosen.device, chosen.name)
        return cls(chosen.device, chosen.type, brightness_threshold=threshold)

    # -- internals -----------------------------------------------------------

    def _device_index(self) -> int:
        """Extract the numeric index from /dev/videoN."""
        return int(self.device.rsplit("video", 1)[-1])

    @staticmethod
    def _to_bgr(frame: np.ndarray) -> np.ndarray:
        """Convert single-channel (greyscale) frames to 3-channel BGR."""
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    def _read_frame(self, max_attempts: int) -> np.ndarray:
        """Read a frame, retrying on transient failures."""
        assert self._cap is not None
        for _ in range(max_attempts):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return self._to_bgr(frame)
        raise RuntimeError(
            f"No frame from {self.device} after {max_attempts} attempts"
        )

    def _read_bright_frame(self, max_attempts: int) -> np.ndarray:
        """Read the next emitter-on (bright) frame from an IR sensor.

        IR cameras interleave emitter-on and emitter-off frames.  We
        skip any frame whose mean pixel value is below the threshold.
        """
        assert self._cap is not None
        for attempt in range(max_attempts):
            ret, frame = self._cap.read()
            if not ret or frame is None:
                continue
            if frame.mean() > self.brightness_threshold:
                return self._to_bgr(frame)
            logger.debug(
                "Dark frame %d/%d from %s (mean=%.1f, threshold=%.1f)",
                attempt + 1, max_attempts, self.device,
                frame.mean(), self.brightness_threshold,
            )
        raise RuntimeError(
            f"No bright frame from {self.device} after {max_attempts} attempts "
            f"(threshold={self.brightness_threshold})"
        )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING)
    cameras = detect_cameras()
    if not cameras:
        print("No cameras detected.")
        sys.exit(1)
    for cam in cameras:
        print(f"{cam.device}  type={cam.type}  formats={','.join(cam.formats)}  name={cam.name!r}")
