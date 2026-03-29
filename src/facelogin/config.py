"""TOML configuration loader with system/user/default fallback.

Load order (first found wins):
  1. /etc/ubuntu-face-login/config.toml   (system-wide)
  2. ~/.config/ubuntu-face-login/config.toml  (per-user)
  3. Built-in defaults
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

# Python 3.11+ has tomllib; older versions need the tomli backport.
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Install-dir heuristic
# ---------------------------------------------------------------------------

def get_install_dir() -> Path:
    """Return the package install root.

    Checks (in order):
      1. FACELOGIN_DIR environment variable
      2. /usr/share/ubuntu-face-login  (system package)
      3. The directory containing this source file (dev checkout)
    """
    env = os.environ.get("FACELOGIN_DIR")
    if env:
        return Path(env)

    system_dir = Path("/usr/share/ubuntu-face-login")
    if system_dir.is_dir():
        return system_dir

    # Dev fallback: two levels up from src/facelogin/config.py → repo root
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    device: str = ""
    type: str = ""  # 'ir' or 'rgb'
    ir_brightness_threshold: int = 20


@dataclass
class EmitterConfig:
    enabled: bool = True
    unit: int = 0
    selector: int = 0
    control_data: str = ""


@dataclass
class AuthConfig:
    threshold: float = 0.45
    timeout: Dict[str, int] = field(default_factory=lambda: {
        "sudo": 3,
        "gdm-password": 5,
        "polkit-1": 5,
        "default": 5,
    })
    max_attempts: int = 30


@dataclass
class EnrollmentConfig:
    samples: int = 20
    min_confidence: float = 0.6


@dataclass
class ModelsConfig:
    dir: str = ""  # resolved lazily from install dir


@dataclass
class DataConfig:
    dir: str = ""  # resolved lazily from install dir


@dataclass
class FaceLoginConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    emitter: EmitterConfig = field(default_factory=EmitterConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    enrollment: EnrollmentConfig = field(default_factory=EnrollmentConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    data: DataConfig = field(default_factory=DataConfig)


# ---------------------------------------------------------------------------
# TOML → dataclass merging
# ---------------------------------------------------------------------------

def _merge_dict_into_dataclass(dc: object, raw: dict) -> None:
    """Recursively update dataclass fields from a dict, ignoring unknown keys."""
    for key, value in raw.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
        elif hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dict_into_dataclass(current, value)
        else:
            setattr(dc, key, value)


def _resolve_defaults(cfg: FaceLoginConfig) -> None:
    """Fill in path defaults that depend on install dir."""
    install = get_install_dir()
    if not cfg.models.dir:
        cfg.models.dir = str(install / "models")
    if not cfg.data.dir:
        cfg.data.dir = str(install / "data")


# ---------------------------------------------------------------------------
# Config file discovery
# ---------------------------------------------------------------------------

_CONFIG_PATHS = [
    Path("/etc/ubuntu-face-login/config.toml"),
]


def _user_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "ubuntu-face-login" / "config.toml"


def load_config(path: Optional[str] = None) -> FaceLoginConfig:
    """Load configuration from TOML file with fallback chain.

    Args:
        path: Explicit config file path.  When provided, only that file
              is tried (no fallback chain).

    Returns:
        Fully resolved FaceLoginConfig.
    """
    cfg = FaceLoginConfig()

    candidates = [Path(path)] if path else [*_CONFIG_PATHS, _user_config_path()]

    for candidate in candidates:
        if candidate.is_file():
            with open(candidate, "rb") as f:
                raw = tomllib.load(f)
            _merge_dict_into_dataclass(cfg, raw)
            break  # first found wins

    _resolve_defaults(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_config: Optional[FaceLoginConfig] = None


def get_config() -> FaceLoginConfig:
    """Return the singleton config, loading on first call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


if __name__ == "__main__":
    import dataclasses
    import json
    import sys

    cfg = get_config()

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        return obj

    print(json.dumps(_to_dict(cfg), indent=2))
