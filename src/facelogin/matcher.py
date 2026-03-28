"""Embedding database and L2-distance matching.

Loads ``.npy`` embedding files from the configured data directory.
Each file is named ``{user_id}_{index}.npy`` and contains a single
512-dim float32 vector.

Matching uses the *top-K average* strategy: for each enrolled user
the K closest stored embeddings are averaged to produce a stable
distance metric that is robust to single-sample outliers.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from facelogin.config import get_config

_TOP_K = 5  # number of closest embeddings to average


class EmbeddingDB:
    """In-memory store of per-user face embeddings loaded from ``.npy`` files."""

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self._db: Dict[str, List[np.ndarray]] = {}
        self._data_dir = data_dir or get_config().data.dir
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Scan *data_dir* for ``{user}_{n}.npy`` files."""
        self._db.clear()
        if not os.path.isdir(self._data_dir):
            return
        for fname in os.listdir(self._data_dir):
            if not fname.endswith(".npy"):
                continue
            # user_id is everything before the last underscore
            parts = fname[:-4].rsplit("_", 1)
            if len(parts) != 2:
                continue
            user_id = parts[0]
            path = os.path.join(self._data_dir, fname)
            emb = np.load(path)
            self._db.setdefault(user_id, []).append(emb)

    def reload(self) -> None:
        """Re-scan the data directory (e.g. after enrollment)."""
        self._load()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def users(self) -> List[str]:
        """Enrolled user IDs."""
        return list(self._db.keys())

    @property
    def empty(self) -> bool:
        return len(self._db) == 0

    def match(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        """Find the closest enrolled user to *embedding*.

        Uses top-K average L2 distance for each user.

        Args:
            embedding: ``(512,)`` float32 query vector.
            threshold: Maximum distance to accept.  Defaults to
                       ``config.auth.threshold``.

        Returns:
            ``(user_id, distance)`` if the best match is within
            *threshold*, otherwise ``(None, float('inf'))``.
        """
        if threshold is None:
            threshold = get_config().auth.threshold

        best_user: Optional[str] = None
        best_dist = float("inf")

        for user_id, embeddings in self._db.items():
            dists = sorted(
                float(np.linalg.norm(embedding - ref)) for ref in embeddings
            )
            k = min(_TOP_K, len(dists))
            avg_dist = float(np.mean(dists[:k]))
            if avg_dist < best_dist:
                best_dist = avg_dist
                best_user = user_id

        if best_dist < threshold:
            return best_user, best_dist
        return None, float("inf")
