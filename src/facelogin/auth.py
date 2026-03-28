#!/usr/bin/env python3
"""Face authentication orchestrator.

Coordinates camera, IR emitter, face detection, embedding, and matching
into a single ``authenticate()`` call.  Parallel-loads camera+emitter vs
ONNX model to minimize cold-start latency (~1s total).

Exit codes (CLI):
  0 = authenticated
  1 = no match / timeout / error
"""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from .camera import Camera
from .config import get_config
from .detector import detect_face, validate_liveness, crop_face
from .embedder import get_embedding, _get_session
from .emitter import activate_emitter
from .matcher import EmbeddingDB

logger = logging.getLogger(__name__)


def authenticate(
    timeout: Optional[int] = None,
    pam_service: str = "default",
) -> Optional[str]:
    """Run the face-auth pipeline and return the matched user_id or None.

    Parameters
    ----------
    timeout:
        Maximum seconds to attempt authentication.  When *None*, looked
        up from ``config.auth.timeout[pam_service]`` (falling back to
        the ``"default"`` key).
    pam_service:
        PAM service name (e.g. ``"sudo"``, ``"gdm-password"``).  Used
        to select the per-service timeout from config.
    """
    config = get_config()

    if timeout is None:
        timeouts = config.auth.timeout
        timeout = timeouts.get(pam_service, timeouts.get("default", 5))

    # ── Load embeddings ────────────────────────────────────────────────
    db = EmbeddingDB()
    if db.empty:
        logger.error("No enrolled embeddings found in %s", config.data.dir)
        return None

    # ── Parallel init: camera+emitter vs ONNX model ───────────────────
    camera: Optional[Camera] = None
    first_frame: Optional[np.ndarray] = None

    def setup_camera() -> tuple[Camera, np.ndarray]:
        cam = Camera.auto_detect()
        activate_emitter(cam.device)
        frame = cam.open()
        return cam, frame

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_cam = pool.submit(setup_camera)
        f_model = pool.submit(_get_session)

        f_model.result()  # ensure model is warm
        camera, first_frame = f_cam.result()

    # ── Frame loop ─────────────────────────────────────────────────────
    start = time.monotonic()
    frame = first_frame

    try:
        while time.monotonic() - start < timeout:
            box, landmarks, confidence = detect_face(frame)

            if confidence is not None and confidence >= config.enrollment.min_confidence:
                valid, reason = validate_liveness(box, landmarks, frame.shape)
                if valid:
                    face_crop = crop_face(frame, box)
                    emb = get_embedding(face_crop)
                    user_id, dist = db.match(emb)

                    if user_id is not None:
                        logger.info(
                            "Authenticated as %s (dist=%.3f, service=%s)",
                            user_id, dist, pam_service,
                        )
                        return user_id
                else:
                    logger.debug("Liveness check failed: %s", reason)

            # Next frame
            try:
                frame = camera.read()
            except RuntimeError:
                logger.warning("Camera read failed — stopping")
                break

    finally:
        camera.release()

    logger.info(
        "Authentication failed after %.1fs (service=%s)",
        time.monotonic() - start, pam_service,
    )
    return None


# ── CLI entry point ────────────────────────────────────────────────────

def main() -> None:
    """CLI wrapper: parse args, run authenticate, exit 0/1."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Face authentication via IR camera + FaceNet-512",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Max seconds to attempt auth (default: from config)",
    )
    parser.add_argument(
        "--service", type=str, default="default",
        help="PAM service name for timeout lookup",
    )
    parser.add_argument(
        "--user", type=str, default=None,
        help="PAM user (accepted for compatibility, not used for matching)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    user = authenticate(timeout=args.timeout, pam_service=args.service)
    if user:
        print(f"✅ {user}", flush=True)
        sys.exit(0)
    else:
        print("❌ Face authentication failed.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
