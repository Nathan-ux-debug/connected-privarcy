"""
data_store

Persistent storage for two things the review/redaction pipeline needs
across restarts:

    D2 - Face Registry:      enrolled people's face embedding vectors
                              (who the face_privacy package should NOT blur)
    D4 - Correction Log:       human reviewer decisions on flagged detections
                                (accept/reject, with the original bbox + context)

Backed by SQLite (stdlib `sqlite3` — no server, no extra dependency) so
this runs anywhere Python does. `api.py` exposes the same functions as a
small FastAPI app, ready for a frontend to call directly.
"""

from .exceptions import DataStoreError
from .models import FaceEmbeddingRecord, CorrectionRecord
from .face_registry import FaceRegistryDB
from .correction_log import CorrectionLogDB

__all__ = [
    "FaceRegistryDB",
    "CorrectionLogDB",
    "FaceEmbeddingRecord",
    "CorrectionRecord",
    "DataStoreError",
]

__version__ = "1.0.0"
