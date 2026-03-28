"""Face detection and landmark-based liveness validation using OpenCV YuNet.

YuNet provides face bounding box + 5 landmarks (right eye, left eye,
nose tip, right mouth corner, left mouth corner) in a single ~5ms pass.

Liveness checks (no separate model needed):
  1. Face is centered in frame
  2. Eyes are at expected distance apart (frontal, not profile)
  3. Nose is between and below eyes (facing camera, not rotated photo)
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from facelogin.config import get_config

_YUNET_FILENAME = "yunet.onnx"

_yunet = None

# Type aliases
Box = Tuple[int, int, int, int]  # (x, y, w, h)
Landmarks = Dict[str, Tuple[float, float]]
DetectionResult = Tuple[Optional[Box], Optional[Landmarks], Optional[float]]


def _get_yunet(width: int = 640, height: int = 360) -> cv2.FaceDetectorYN:
    """Lazy-init YuNet detector, loading the model from config.models.dir."""
    global _yunet
    if _yunet is None:
        model_path = os.path.join(get_config().models.dir, _YUNET_FILENAME)
        _yunet = cv2.FaceDetectorYN.create(model_path, "", (width, height), 0.5)
    return _yunet


def detect_face(frame: np.ndarray) -> DetectionResult:
    """Detect the best (highest-confidence) face in *frame*.

    Returns:
        (box, landmarks, confidence) where box = (x, y, w, h) and
        landmarks is a dict with keys right_eye, left_eye, nose,
        right_mouth, left_mouth — each a (x, y) float pair.
        All three are None when no face is found.
    """
    h, w = frame.shape[:2]
    yunet = _get_yunet(w, h)
    yunet.setInputSize((w, h))

    _, faces = yunet.detect(frame)
    if faces is None or len(faces) == 0:
        return None, None, None

    face = faces[0]
    box: Box = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
    landmarks: Landmarks = {
        "right_eye": (float(face[4]), float(face[5])),
        "left_eye": (float(face[6]), float(face[7])),
        "nose": (float(face[8]), float(face[9])),
        "right_mouth": (float(face[10]), float(face[11])),
        "left_mouth": (float(face[12]), float(face[13])),
    }
    confidence = float(face[14])

    return box, landmarks, confidence


def validate_liveness(
    box: Optional[Box],
    landmarks: Optional[Landmarks],
    frame_shape: Tuple[int, ...],
    center_thresh: float = 0.25,
) -> Tuple[bool, str]:
    """Check that the detected face is real, frontal, and centered.

    Returns:
        (is_valid, reason) — reason is ``"ok"`` on success.
    """
    if box is None or landmarks is None:
        return False, "no face"

    h, w = frame_shape[:2]
    fx, fy, fw, fh = box

    # 1. Face centered in frame
    face_cx = (fx + fw / 2) / w
    face_cy = (fy + fh / 2) / h
    offset = max(abs(face_cx - 0.5), abs(face_cy - 0.5))
    if offset > center_thresh:
        return False, f"not centered (offset={offset:.2f})"

    # 2. Eye distance ratio — frontal faces have eyes ~20-60% of face width
    le = landmarks["left_eye"]
    re = landmarks["right_eye"]
    eye_dist = np.sqrt((le[0] - re[0]) ** 2 + (le[1] - re[1]) ** 2)
    ratio = eye_dist / max(fw, 1)
    if not (0.20 < ratio < 0.60):
        return False, f"eye ratio off (ratio={ratio:.2f})"

    # 3. Nose between and below eyes (detects profile / unusual angle)
    nose = landmarks["nose"]
    eye_min_x = min(le[0], re[0])
    eye_max_x = max(le[0], re[0])
    eye_max_y = max(le[1], re[1])

    if not (eye_min_x - 5 < nose[0] < eye_max_x + 5):
        return False, "nose not between eyes (profile)"
    if nose[1] < eye_max_y:
        return False, "nose above eyes (unusual angle)"

    return True, "ok"


def crop_face(frame: np.ndarray, box: Box) -> np.ndarray:
    """Crop and resize the face region to 160×160 BGR for FaceNet input."""
    h, w = frame.shape[:2]
    fx, fy, fw, fh = box
    x1 = max(0, fx)
    y1 = max(0, fy)
    x2 = min(w, fx + fw)
    y2 = min(h, fy + fh)
    face = frame[y1:y2, x1:x2]
    return cv2.resize(face, (160, 160))
