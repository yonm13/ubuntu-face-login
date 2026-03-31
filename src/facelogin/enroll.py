#!/usr/bin/env python3
"""Face enrollment — capture embeddings for a user.

Captures frames from the camera, validates face quality via YuNet
landmarks, computes FaceNet-512 embeddings, and saves them as ``.npy``
files alongside face thumbnails.

Pose-guided enrollment captures samples across several head positions
(straight, left, right, up, down) so that the matcher has coverage of
the small angle variations that occur during real authentication.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .camera import Camera
from .config import get_config
from .detector import detect_face, validate_liveness, crop_face, Box, Landmarks
from .embedder import get_embedding
from .emitter import activate_emitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pose guide
# ---------------------------------------------------------------------------

@dataclass
class Pose:
    """A single guided head position with instructions for the user."""
    label: str            # short machine-friendly id, e.g. "close_straight"
    instruction: str      # human-readable prompt shown to the user
    samples: int = 4      # how many samples to capture at this pose
    transition_delay: Optional[float] = None  # override global delay before this pose


# ---------------------------------------------------------------------------
# Distance-aware pose builder
# ---------------------------------------------------------------------------

# Base directions (label, instruction, default samples)
_DIRECTIONS: List[tuple[str, str, int]] = [
    ("straight", "Look straight at the camera", 6),
    ("left",     "Turn slightly left",           4),
    ("right",    "Turn slightly right",          4),
    ("up",       "Tilt head up slightly",        3),
    ("down",     "Tilt head down slightly",      3),
]

# Distance levels (label, short cue for instructions)
_DISTANCES: List[tuple[str, str]] = [
    ("close",  "close (~30 cm / 1 ft)"),
    ("medium", "medium (~60 cm / 2 ft)"),
    ("far",    "far (~1 m / 3 ft)"),
]

# Pause given before the first pose of a new distance group so the user
# has time to reposition.  Shown as a countdown in the UI.
DISTANCE_TRANSITION_DELAY: float = 5.0


def build_poses(
    n_distances: int = 3,
    direction_samples: Optional[List[int]] = None,
) -> List[Pose]:
    """Build a flat guided pose list with *n_distances* distance phases.

    Args:
        n_distances:       1, 2, or 3 — number of distance levels to include.
        direction_samples: Override sample counts per direction (5 values,
                           matching the order in ``_DIRECTIONS``).  When
                           ``None``, the defaults in ``_DIRECTIONS`` are used.

    The first pose of each distance phase (except the very first) gets
    ``transition_delay = DISTANCE_TRANSITION_DELAY`` so the enrollment loop
    gives the user time to reposition before capturing resumes.
    """
    if direction_samples is None:
        direction_samples = [samples for _, _, samples in _DIRECTIONS]

    poses: List[Pose] = []
    for d_idx, (dist_key, dist_label) in enumerate(_DISTANCES[:n_distances]):
        for p_idx, ((dir_key, dir_instruction, _), samples) in enumerate(
            zip(_DIRECTIONS, direction_samples)
        ):
            is_first_of_new_distance = d_idx > 0 and p_idx == 0

            if is_first_of_new_distance:
                # Prominent instruction: distance cue + first direction
                instruction = f"📍 Move to {dist_label}  —  {dir_instruction}"
                delay = DISTANCE_TRANSITION_DELAY
            elif n_distances > 1:
                # Subsequent poses show the distance label in brackets
                instruction = f"{dir_instruction}  [{dist_label}]"
                delay = None
            else:
                # Single-distance mode: plain instruction (matches old default)
                instruction = dir_instruction
                delay = None

            poses.append(Pose(
                label=f"{dist_key}_{dir_key}",
                instruction=instruction,
                samples=samples,
                transition_delay=delay,
            ))
    return poses


DEFAULT_POSES: List[Pose] = build_poses(n_distances=3)
# Total: (6+4+4+3+3) × 3 distances = 60 samples


# ---------------------------------------------------------------------------
# Callback type hints
# ---------------------------------------------------------------------------

# on_frame(frame, box, landmarks, confidence, valid, reason)
OnFrameCallback = Callable[
    [np.ndarray, Optional[Box], Optional[Landmarks], Optional[float], bool, str],
    None,
]
# on_sample(sample_index, total_samples)
OnSampleCallback = Callable[[int, int], None]
# on_pose(pose_index, pose, samples_this_pose, total_poses)
OnPoseCallback = Callable[[int, Pose, int, int], None]


# ---------------------------------------------------------------------------
# Core enrollment function
# ---------------------------------------------------------------------------

def enroll_user(
    user_id: str,
    data_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    poses: Optional[List[Pose]] = None,
    sample_delay: float = 0.6,
    pose_transition_delay: float = 3.0,
    wipe_existing: bool = False,
    on_sample: Optional[OnSampleCallback] = None,
    on_frame: Optional[OnFrameCallback] = None,
    on_pose: Optional[OnPoseCallback] = None,
    on_pose_transition: Optional[Callable[[Pose, int], None]] = None,
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
        Total number of valid face samples to capture.  When *poses* is
        provided this is ignored — pose sample counts determine the total.
        When neither *poses* nor *num_samples* is given, ``DEFAULT_POSES``
        is used.
    poses:
        Ordered list of :class:`Pose` objects describing guided head
        positions.  Pass an empty list or ``None`` for unguided capture.
        If ``None`` and ``num_samples`` is set, falls back to unguided
        capture with that many samples.
    sample_delay:
        Minimum seconds between consecutive saved samples.  Prevents
        consecutive near-identical frames from inflating the sample count.
        Default 0.6 s.
    pose_transition_delay:
        Seconds to pause between poses so the user can reposition.
        A per-second countdown fires ``on_pose_transition`` during the gap.
        Default 3.0 s.
    wipe_existing:
        When ``True``, delete all previously saved embeddings and face
        thumbnails for *user_id* before capturing new samples.  Use for
        a full re-enroll.  Default ``False`` (append to existing).
    on_sample:
        Called after each valid sample is saved: ``(index, total)``.
    on_frame:
        Called for every camera frame: ``(frame, box, landmarks,
        confidence, valid, reason)``.  Useful for UI overlays.
    on_pose:
        Called when the current pose becomes active:
        ``(pose_index, pose, samples_this_pose, total_poses)``.
        Not called in unguided mode.
    on_pose_transition:
        Called once per second during the gap between poses:
        ``(next_pose, seconds_remaining)``.  Use to show a countdown.

    Returns
    -------
    int
        Number of samples successfully saved.
    """
    config = get_config()
    data_dir = data_dir or config.data.dir
    min_confidence = config.enrollment.min_confidence

    # Resolve capture plan
    if poses is None and num_samples is None:
        # Default: guided multi-pose
        capture_poses = DEFAULT_POSES
        total_samples = sum(p.samples for p in capture_poses)
    elif poses is not None:
        # Explicit pose list
        capture_poses = poses
        total_samples = sum(p.samples for p in capture_poses)
    else:
        # num_samples set, no poses — unguided flat capture
        capture_poses = []
        total_samples = num_samples  # type: ignore[assignment]

    # Ensure output directories exist
    faces_dir = os.path.join(data_dir, "faces")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)

    # Optionally wipe existing samples for this user
    if wipe_existing:
        for fname in os.listdir(data_dir):
            if fname.startswith(f"{user_id}_") and fname.endswith(".npy"):
                os.remove(os.path.join(data_dir, fname))
        thumb_dir = os.path.join(data_dir, "faces")
        if os.path.isdir(thumb_dir):
            for fname in os.listdir(thumb_dir):
                if fname.startswith(f"{user_id}_"):
                    os.remove(os.path.join(thumb_dir, fname))

    # Find next available index (don't overwrite existing embeddings)
    existing = [
        f for f in os.listdir(data_dir)
        if f.startswith(f"{user_id}_") and f.endswith(".npy")
    ]
    start_index = len(existing)

    camera = Camera.auto_detect()
    activate_emitter(camera.device)
    frame = camera.open()

    saved = 0
    last_saved_time = 0.0

    try:
        if capture_poses:
            # ---- Guided multi-pose capture ----
            for pose_idx, pose in enumerate(capture_poses):
                pose_saved = 0

                # Countdown before pose starts (skip for first pose)
                if pose_idx > 0:
                    delay = (
                        pose.transition_delay
                        if pose.transition_delay is not None
                        else pose_transition_delay
                    )
                    if delay > 0:
                        next_pose = pose
                        remaining = int(delay)
                        tick_deadline = time.monotonic()
                        while remaining > 0:
                            if on_pose_transition is not None:
                                on_pose_transition(next_pose, remaining)
                            tick_deadline += 1.0
                            # Keep reading frames so the video stays live
                            while time.monotonic() < tick_deadline:
                                if on_frame is not None:
                                    box, lm, conf = detect_face(frame)
                                    valid, reason = validate_liveness(box, lm, frame.shape)
                                    on_frame(frame, box, lm, conf, valid, reason)
                                frame = camera.read()
                            remaining -= 1
                        # Fire one last tick at 0 so UI can clear the countdown
                        if on_pose_transition is not None:
                            on_pose_transition(next_pose, 0)

                if on_pose is not None:
                    on_pose(pose_idx, pose, pose.samples, len(capture_poses))

                logger.info(
                    "Pose %d/%d: %s (%d samples)",
                    pose_idx + 1, len(capture_poses), pose.instruction, pose.samples,
                )

                while pose_saved < pose.samples:
                    box, landmarks, confidence = detect_face(frame)
                    valid = False
                    reason = "no face"

                    if confidence is not None and confidence >= min_confidence:
                        valid, reason = validate_liveness(box, landmarks, frame.shape)
                    elif confidence is not None:
                        reason = f"low confidence ({confidence:.2f})"

                    if on_frame is not None:
                        on_frame(frame, box, landmarks, confidence, valid, reason)

                    now = time.monotonic()
                    if valid and (now - last_saved_time) >= sample_delay:
                        face_crop = crop_face(frame, box, landmarks=landmarks)
                        emb = get_embedding(face_crop)

                        idx = start_index + saved
                        np.save(os.path.join(data_dir, f"{user_id}_{idx}.npy"), emb)
                        cv2.imwrite(
                            os.path.join(faces_dir, f"{user_id}_{idx}.jpg"), frame
                        )

                        saved += 1
                        pose_saved += 1
                        last_saved_time = now
                        logger.info(
                            "Sample %d/%d (pose: %s)",
                            saved, total_samples, pose.label,
                        )

                        if on_sample is not None:
                            on_sample(saved, total_samples)

                    try:
                        frame = camera.read()
                    except RuntimeError:
                        logger.warning("Camera read failed — stopping")
                        raise KeyboardInterrupt

        else:
            # ---- Unguided flat capture ----
            while saved < total_samples:
                box, landmarks, confidence = detect_face(frame)
                valid = False
                reason = "no face"

                if confidence is not None and confidence >= min_confidence:
                    valid, reason = validate_liveness(box, landmarks, frame.shape)
                elif confidence is not None:
                    reason = f"low confidence ({confidence:.2f})"

                if on_frame is not None:
                    on_frame(frame, box, landmarks, confidence, valid, reason)

                now = time.monotonic()
                if valid and (now - last_saved_time) >= sample_delay:
                    face_crop = crop_face(frame, box, landmarks=landmarks)
                    emb = get_embedding(face_crop)

                    idx = start_index + saved
                    np.save(os.path.join(data_dir, f"{user_id}_{idx}.npy"), emb)
                    cv2.imwrite(
                        os.path.join(faces_dir, f"{user_id}_{idx}.jpg"), frame
                    )

                    saved += 1
                    last_saved_time = now

                    if on_sample is not None:
                        on_sample(saved, total_samples)

                try:
                    frame = camera.read()
                except RuntimeError:
                    logger.warning("Camera read failed — stopping")
                    break

    except KeyboardInterrupt:
        logger.info("Enrollment interrupted")
    finally:
        camera.release()

    logger.info(
        "Enrollment complete: %d/%d samples for %s", saved, total_samples, user_id
    )
    return saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def enroll_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Enroll face embeddings for a user",
    )
    parser.add_argument("user_id", help="User ID to enroll")
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Number of face samples (unguided mode, overrides default poses)",
    )
    parser.add_argument(
        "--no-poses", action="store_true",
        help="Disable guided pose prompts, just capture N samples",
    )
    parser.add_argument(
        "--delay", type=float, default=0.4,
        help="Minimum seconds between saved samples (default: 0.4)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Data directory for embeddings (default: from config)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.no_poses or args.samples:
        poses = []
        total = args.samples or sum(p.samples for p in DEFAULT_POSES)
        print(f"Enrolling '{args.user_id}' — look at the camera, eyes open.")
        print(f"Capturing {total} samples.  Press Ctrl+C to stop early.\n")
    else:
        poses = None
        total = sum(p.samples for p in DEFAULT_POSES)
        print(f"Enrolling '{args.user_id}' with guided pose capture ({total} samples).")
        print("Follow the on-screen prompts.  Press Ctrl+C to stop early.\n")

    current_pose: list[str] = []

    def on_pose(idx: int, pose: Pose, n: int, total_poses: int) -> None:
        current_pose.clear()
        current_pose.append(pose.instruction)
        print(f"\n[Pose {idx+1}/{total_poses}] {pose.instruction} ({n} samples needed)")

    def on_sample(index: int, total: int) -> None:
        pose_label = f" — {current_pose[0]}" if current_pose else ""
        print(f"  [✓] Sample {index}/{total}{pose_label}", flush=True)

    saved = enroll_user(
        user_id=args.user_id,
        data_dir=args.data_dir,
        num_samples=args.samples if (args.no_poses or args.samples) else None,
        poses=poses,
        sample_delay=args.delay,
        on_sample=on_sample,
        on_pose=on_pose if not args.no_poses else None,
    )

    print(f"\n{'✅' if saved > 0 else '⚠'} Saved {saved} face embeddings for '{args.user_id}'.")
    if saved == 0:
        print("No faces captured — make sure you're facing the camera with eyes open.")
        sys.exit(1)


if __name__ == "__main__":
    enroll_cli()
