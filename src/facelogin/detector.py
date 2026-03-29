"""Face detection and landmark-based liveness validation using OpenCV YuNet.

YuNet provides face bounding box + 5 landmarks (right eye, left eye,
nose tip, right mouth corner, left mouth corner) in a single ~5ms pass.

Liveness checks (no separate model needed):
  1. Face is centered in frame
  2. Eyes are at expected distance apart (frontal, not profile)
  3. Nose is between and below eyes (facing camera, not rotated photo)

Face alignment uses the two eye landmarks to correct for head roll
(rotation) and distance (scale), placing eyes at fixed positions in the
160×160 output crop.  This dramatically improves FaceNet-512 embedding
consistency across frames.
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


def align_face(
    frame: np.ndarray,
    landmarks: Landmarks,
    output_size: int = 160,
) -> np.ndarray:
    """Geometrically align a face using eye landmark positions.

    Corrects for head roll (rotation) and subject distance (scale) so
    that both eyes land at fixed horizontal positions in the output crop.
    This makes FaceNet-512 embeddings far more consistent across frames
    because the model sees the face at a canonical pose every time.

    Target geometry (in the output_size × output_size crop):
      - Eye midpoint at (output_size/2, 38% of output_size)
      - Inter-eye distance = 30% of output_size
      - Rotation corrected so the eye line is horizontal

    Args:
        frame:       Full camera frame (BGR or greyscale-as-BGR).
        landmarks:   YuNet landmark dict with 'right_eye' and 'left_eye'.
        output_size: Side length of the square output crop (default 160).

    Returns:
        Aligned ``(output_size, output_size, 3)`` uint8 BGR array.
    """
    # right_eye has lower x in image (person's right, camera's left)
    # left_eye has higher x in image (person's left, camera's right)
    re = np.array(landmarks["right_eye"], dtype=np.float64)
    le = np.array(landmarks["left_eye"],  dtype=np.float64)

    # Angle of the inter-eye line (y-axis points down in image coords)
    dy = le[1] - re[1]
    dx = le[0] - re[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Midpoint between the eyes — this is the anchor for the transform
    eye_center = ((re + le) / 2).astype(np.float32)

    # Scale so the inter-eye distance becomes 30% of the output width
    target_eye_dist = 0.30 * output_size
    actual_eye_dist = np.hypot(dx, dy)
    scale = target_eye_dist / max(actual_eye_dist, 1e-6)

    # Rotation+scale matrix around the eye midpoint.
    # Negating the angle rotates the image so the eye line becomes horizontal.
    M = cv2.getRotationMatrix2D(tuple(eye_center), -angle, scale)

    # Translate so the eye midpoint lands at (output_size/2, 38% height).
    # After getRotationMatrix2D with eye_center as pivot, the pivot maps
    # to itself — we then shift it to the desired output position.
    M[0, 2] += output_size / 2        - eye_center[0]
    M[1, 2] += 0.38 * output_size     - eye_center[1]

    return cv2.warpAffine(
        frame, M, (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_face(
    frame: np.ndarray,
    box: Box,
    landmarks: Optional[Landmarks] = None,
    output_size: int = 160,
) -> np.ndarray:
    """Return a standardised face crop for FaceNet-512 input.

    When *landmarks* are provided the crop is geometrically aligned using
    ``align_face()`` (recommended — corrects head roll and scale).
    Falls back to a plain bounding-box crop when landmarks are absent.

    Args:
        frame:       Full camera frame.
        box:         Bounding box ``(x, y, w, h)`` from YuNet.
        landmarks:   YuNet landmark dict (pass to enable alignment).
        output_size: Output square side length (default 160).

    Returns:
        ``(output_size, output_size, 3)`` uint8 BGR array.
    """
    if landmarks is not None:
        return align_face(frame, landmarks, output_size)

    # Fallback: plain bounding-box crop (no alignment)
    h, w = frame.shape[:2]
    fx, fy, fw, fh = box
    x1, y1 = max(0, fx), max(0, fy)
    x2, y2 = min(w, fx + fw), min(h, fy + fh)
    return cv2.resize(frame[y1:y2, x1:x2], (output_size, output_size))
