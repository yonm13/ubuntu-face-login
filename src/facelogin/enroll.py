#!/usr/bin/env python3
"""Face enrollment — capture embeddings for a user.

Captures frames from the camera, validates face quality via YuNet
landmarks, computes FaceNet-512 embeddings, and saves them as ``.npy``
files alongside face thumbnails.

Layout in data_dir::

    data/
      {user_id}_0.npy
      {user_id}_1.npy
      ...
      faces/
        {user_id}_0.jpg
        {user_id}_1.jpg
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .camera import Camera
from .config import get_config
from .detector import detect_face, validate_liveness, crop_face, Box, Landmarks
from .embedder import get_embedding
from .emitter import activate_emitter

logger = logging.getLogger(__name__)

# Callback type hints (not enforced, for documentation)
# on_frame(frame, box, landmarks, confidence, valid, reason)
OnFrameCallback = Callable[
    [np.ndarray, Optional[Box], Optional[Landmarks], Optional[float], bool, str],
    None,
]
# on_sample(sample_index, total_samples)
OnSampleCallback = Callable[[int, int], None]


def enroll_user(
    user_id: str,
    data_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    on_sample: Optional[OnSampleCallback] = None,
    on_frame: Optional[OnFrameCallback] = None,
) -> int:
    """Capture and save face embeddings for *user_id*.

    Parameters
    ----------
    user_id:
        Identifier for the enrolled user (e.g. Unix username).
    data_dir:
        Directory to store ``.npy`` embeddings and ``faces/`` thumbnails.
        Defaults to ``config.data.dir``.
    num_samples:
        Number of valid face samples to capture.
        Defaults to ``config.enrollment.samples``.
    on_sample:
        Called after each valid sample is saved: ``(index, total)``.
    on_frame:
        Called for every frame read from the camera, regardless of
        whether a face was detected or valid.  Useful for UI overlays:
        ``(frame, box, landmarks, confidence, valid, reason)``.

    Returns
    -------
    int
        Number of samples successfully saved.
    """
    config = get_config()
    data_dir = data_dir or config.data.dir
    num_samples = num_samples if num_samples is not None else config.enrollment.samples
    min_confidence = config.enrollment.min_confidence

    # Ensure output directories exist
    faces_dir = os.path.join(data_dir, "faces")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)

    # Setup camera + emitter
    camera = Camera.auto_detect()
    activate_emitter(camera.device)
    frame = camera.open()

    saved = 0

    try:
        while saved < num_samples:
            box, landmarks, confidence = detect_face(frame)

            valid = False
            reason = "no face"

            if confidence is not None and confidence >= min_confidence:
                valid, reason = validate_liveness(box, landmarks, frame.shape)
            elif confidence is not None:
                reason = f"low confidence ({confidence:.2f})"

            # Fire per-frame callback (UI overlay, etc.)
            if on_frame is not None:
                on_frame(frame, box, landmarks, confidence, valid, reason)

            if valid:
                face_crop = crop_face(frame, box)
                emb = get_embedding(face_crop)

                # Save embedding
                npy_path = os.path.join(data_dir, f"{user_id}_{saved}.npy")
                np.save(npy_path, emb)

                # Save face thumbnail
                jpg_path = os.path.join(faces_dir, f"{user_id}_{saved}.jpg")
                cv2.imwrite(jpg_path, frame)

                saved += 1
                logger.info("Sample %d/%d saved for %s", saved, num_samples, user_id)

                if on_sample is not None:
                    on_sample(saved, num_samples)

            # Next frame
            try:
                frame = camera.read()
            except RuntimeError:
                logger.warning("Camera read failed — stopping enrollment")
                break

    except KeyboardInterrupt:
        logger.info("Enrollment interrupted by user")
    finally:
        camera.release()

    logger.info("Enrollment complete: %d/%d samples for %s", saved, num_samples, user_id)
    return saved


# ── CLI entry point ────────────────────────────────────────────────────

def enroll_cli() -> None:
    """CLI wrapper for enrollment with text progress output."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enroll face embeddings for a user",
    )
    parser.add_argument("user_id", help="User ID to enroll")
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Number of face samples to capture (default: from config)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Data directory for embeddings (default: from config)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = get_config()
    total = args.samples or config.enrollment.samples

    print(f"Enrolling '{args.user_id}' — look at the camera, eyes open.")
    print(f"Capturing {total} samples.  Press Ctrl+C to stop early.\n")

    def on_sample(index: int, total: int) -> None:
        print(f"  [✓] Sample {index}/{total}", flush=True)

    saved = enroll_user(
        user_id=args.user_id,
        data_dir=args.data_dir,
        num_samples=args.samples,
        on_sample=on_sample,
    )

    print(f"\n✅ Saved {saved} face embeddings for '{args.user_id}'.")
    if saved == 0:
        print("⚠  No faces captured — make sure you're facing the camera with eyes open.")
        sys.exit(1)


if __name__ == "__main__":
    enroll_cli()
