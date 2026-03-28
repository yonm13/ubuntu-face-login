"""FaceNet-512 embedding via ONNX Runtime (CPU only).

No PyTorch dependency.  Loads the pre-exported ONNX model (~0.7 MB)
in ~180 ms; inference takes ~15 ms per face crop.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import onnxruntime as ort

from facelogin.config import get_config

_FACENET_FILENAME = "facenet512.onnx"

_session: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:
    """Lazy-init the ONNX inference session from config.models.dir."""
    global _session
    if _session is None:
        model_path = os.path.join(get_config().models.dir, _FACENET_FILENAME)
        _session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
    return _session


def get_embedding(face_bgr_160: np.ndarray) -> np.ndarray:
    """Compute a 512-dim L2 embedding from a 160×160 BGR face crop.

    Args:
        face_bgr_160: ``(160, 160, 3)`` uint8 array in BGR colour order.

    Returns:
        ``(512,)`` float32 embedding vector.
    """
    rgb = cv2.cvtColor(face_bgr_160, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb / 255.0 - 0.5) / 0.5  # normalise to [-1, 1]
    tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 160, 160)

    result = _get_session().run(None, {"input": tensor})
    return result[0].squeeze()  # (512,)
