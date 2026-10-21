"""D2 - Face Registry: persistent storage for enrolled face embeddings."""

import datetime
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from .exceptions import DataStoreError
from .models import FaceEmbeddingRecord

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name TEXT NOT NULL,
    embedding TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_person ON face_embeddings(person_name);
"""


class FaceRegistryDB:
    """SQLite-backed store for enrolled face embeddings (D2).

    One row per exemplar embedding — a person can be enrolled with
    several (different angles/lighting), all matched against at lookup
    time by face_privacy.

    Parameters:
        db_path: Path to the SQLite file. Created if it doesn't exist.
                  Pass ":memory:" for an ephemeral in-process database
                  (handy for tests).
    """

    def __init__(self, db_path: str | Path = "face_registry.db") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def enroll(self, person_name: str, embedding: List[float], source: str = "") -> int:
        """Store one embedding for a person. Returns the new row id.

        Raises:
            DataStoreError: if person_name is empty or embedding is empty.
        """
        if not person_name or not person_name.strip():
            raise DataStoreError("person_name must be non-empty.")
        if not embedding:
            raise DataStoreError("embedding must be a non-empty vector.")

        record = FaceEmbeddingRecord(
            person_name=person_name.strip(),
            embedding=list(embedding),
            source=source,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        try:
            cursor = self._conn.execute(
                "INSERT INTO face_embeddings (person_name, embedding, source, created_at) VALUES (?, ?, ?, ?)",
                (record.person_name, record.embedding_json(), record.source, record.created_at),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise DataStoreError(f"Failed to enroll face for '{person_name}': {exc}") from exc

        logger.info("Enrolled embedding for '%s' (row id=%d, source='%s').", person_name, cursor.lastrowid, source)
        return cursor.lastrowid

    def all_embeddings(self) -> List[FaceEmbeddingRecord]:
        """Return every enrolled embedding across every person — the
        full set face_privacy's matcher compares an incoming face against."""
        try:
            rows = self._conn.execute(
                "SELECT id, person_name, embedding, source, created_at FROM face_embeddings"
            ).fetchall()
        except sqlite3.Error as exc:
            raise DataStoreError(f"Failed to read face embeddings: {exc}") from exc

        return [
            FaceEmbeddingRecord(
                id=row[0], person_name=row[1],
                embedding=FaceEmbeddingRecord.embedding_from_json(row[2]),
                source=row[3], created_at=row[4],
            )
            for row in rows
        ]

    def embeddings_for(self, person_name: str) -> List[FaceEmbeddingRecord]:
        """All embeddings enrolled for one specific person."""
        try:
            rows = self._conn.execute(
                "SELECT id, person_name, embedding, source, created_at FROM face_embeddings WHERE person_name = ?",
                (person_name,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise DataStoreError(f"Failed to read embeddings for '{person_name}': {exc}") from exc

        return [
            FaceEmbeddingRecord(
                id=row[0], person_name=row[1],
                embedding=FaceEmbeddingRecord.embedding_from_json(row[2]),
                source=row[3], created_at=row[4],
            )
            for row in rows
        ]

    def list_people(self) -> List[str]:
        """Distinct list of everyone currently enrolled."""
        rows = self._conn.execute("SELECT DISTINCT person_name FROM face_embeddings ORDER BY person_name").fetchall()
        return [r[0] for r in rows]

    def remove_person(self, person_name: str) -> int:
        """Delete every embedding for a person (revoke enrollment).
        Returns the number of rows deleted."""
        cursor = self._conn.execute("DELETE FROM face_embeddings WHERE person_name = ?", (person_name,))
        self._conn.commit()
        logger.info("Removed %d embedding(s) for '%s'.", cursor.rowcount, person_name)
        return cursor.rowcount

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FaceRegistryDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
