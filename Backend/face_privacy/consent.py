"""Consent matching for faces that YOLO has already found.

``FaceConsentMatcher.known_indices`` answers: "which of these YOLO face boxes
belong to a person who is registered (consented) in the D2 face registry?"

Safety properties
-----------------
* Any failure (DB error, embedding error, shape mismatch) => the face is NOT
  treated as consented, i.e. it stays redacted. Failing closed is deliberate.
* The enrolled embeddings are cached and only reloaded when the registry row
  count changes (the old code re-read and JSON-parsed every embedding for
  every frame).
* Matching uses the face crop of the YOLO box itself - the same crop source
  used when registering people - so crops are consistent.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FaceConsentMatcher:
    def __init__(self, registry, embedder, matcher) -> None:
        self.registry = registry
        self.embedder = embedder
        self.matcher = matcher
        self._enrolled: List[Tuple[str, np.ndarray]] = []
        self._count = -1
        self.last_error: str | None = None

    @property
    def ready(self) -> bool:
        return all((self.registry, self.embedder, self.matcher))

    def _refresh(self) -> None:
        count = self.registry.count()
        if count != self._count:
            self._enrolled = [(r.person_name, np.asarray(r.embedding, dtype=np.float64)) for r in self.registry.all_embeddings()]
            self._count = count

    def known_indices(self, frame: np.ndarray, boxes: Sequence[Sequence[float]]) -> set:
        """Indices into ``boxes`` that match a registered person."""
        if not self.ready or not boxes:
            return set()
        try:
            self._refresh()
            if not self._enrolled:
                return set()
            from .models import FaceBox

            height, width = frame.shape[:2]
            known = set()
            for index, box in enumerate(boxes):
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                if x2 - x1 < 12 or y2 - y1 < 12:
                    continue  # too small to identify reliably -> stays redacted
                embedding = self.embedder.embed(frame[y1:y2, x1:x2])
                result = self.matcher.match(FaceBox(bbox=(x1, y1, x2, y2), confidence=1.0), embedding, self._enrolled)
                if result.is_known:
                    known.add(index)
            self.last_error = None
            return known
        except Exception as exc:
            self.last_error = str(exc)
            logger.exception("Face consent matching failed; treating all faces as NOT consented (fail closed).")
            return set()
