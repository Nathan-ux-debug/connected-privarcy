"""Enrollment: the "register this person" convenience path.

Combines FaceDetector + FaceEmbedder + a data_store.FaceRegistryDB into
one call, so enrolling someone from a photo is a single function call
instead of wiring the three pieces together by hand every time.
"""

import logging
from typing import List

import numpy as np

from .detector import FaceDetector
from .embedder import FaceEmbedder
from .exceptions import FacePrivacyError

logger = logging.getLogger(__name__)


def enroll_person(
    person_name: str,
    photo: np.ndarray,
    registry,  # data_store.FaceRegistryDB — typed loosely to avoid a hard import dependency
    detector: FaceDetector | None = None,
    embedder: FaceEmbedder | None = None,
    source: str = "",
) -> int:
    """Detect the (single, largest) face in `photo`, embed it, and store
    it in `registry` under `person_name`.

    Args:
        person_name: Identity to enroll this face under.
        photo:        BGR NumPy array containing the person's face —
                       ideally a clear, front-facing, well-lit photo.
        registry:       A data_store.FaceRegistryDB (or anything with a
                         matching `.enroll(person_name, embedding, source)`
                         method) to store the resulting embedding in.
        detector / embedder: Reused across multiple enrollment calls if
                               provided, otherwise created fresh each call.
        source:                  Free-text note on where this photo came from.

    Returns:
        The new row id in the registry.

    Raises:
        FacePrivacyError: if no face (or more than one face, ambiguous
                            which to enroll) is found in the photo.
    """
    detector = detector or FaceDetector()
    embedder = embedder or FaceEmbedder()

    faces = detector.detect(photo)
    if not faces:
        raise FacePrivacyError(
            f"No face found in the enrollment photo for '{person_name}'. "
            f"Use a clear, front-facing, well-lit photo."
        )
    if len(faces) > 1:
        logger.warning(
            "Found %d faces in the enrollment photo for '%s' — using the largest one. "
            "For best results, use a photo with only one person in it.",
            len(faces), person_name,
        )

    # If multiple faces were found, assume the largest is the intended subject.
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = (int(v) for v in largest.bbox)
    face_crop = photo[y1:y2, x1:x2]

    embedding = embedder.embed(face_crop)
    row_id = registry.enroll(person_name, embedding.tolist(), source=source)
    logger.info("Enrolled '%s' (row id=%d).", person_name, row_id)
    return row_id
