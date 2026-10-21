"""FastAPI endpoints exposing D2 (Face Registry) and D4 (Correction Log)
to a frontend.

This is a thin wrapper — every endpoint just validates the request shape
and calls straight into FaceRegistryDB / CorrectionLogDB. Run with:

    uvicorn data_store.api:app --reload

Then, e.g.:
    POST /corrections   {"frame_id": "clip1_042", "bbox": [10,20,100,200],
                          "decision": "accepted", "detected_class": "credit_card",
                          "confidence": 0.81, "reviewer": "alice"}
    GET  /corrections?decision=accepted&limit=50
    POST /faces          {"person_name": "Alice", "embedding": [0.1, 0.2, ...]}
    GET  /faces
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .correction_log import CorrectionLogDB
from .face_registry import FaceRegistryDB
from .exceptions import DataStoreError

app = FastAPI(title="Redaction Pipeline Data Store (D2 + D4)")

# A single shared connection per store for the life of the process. In a
# real deployment, swap these paths for a persistent location / mount.
_face_db = FaceRegistryDB("face_registry.db")
_correction_db = CorrectionLogDB("correction_log.db")


class EnrollFaceRequest(BaseModel):
    person_name: str
    embedding: List[float]
    source: str = ""


class RecordDecisionRequest(BaseModel):
    frame_id: str
    bbox: List[float]
    decision: str  # "accepted" | "rejected"
    detected_class: str = ""
    confidence: float = 0.0
    context: str = ""
    reviewer: str = ""


@app.post("/faces")
def enroll_face(req: EnrollFaceRequest) -> dict:
    try:
        row_id = _face_db.enroll(req.person_name, req.embedding, req.source)
    except DataStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"id": row_id, "person_name": req.person_name}


@app.get("/faces")
def list_faces() -> dict:
    return {"people": _face_db.list_people(), "total_embeddings": _face_db.count()}


@app.delete("/faces/{person_name}")
def remove_face(person_name: str) -> dict:
    deleted = _face_db.remove_person(person_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No enrollment found for '{person_name}'")
    return {"person_name": person_name, "embeddings_removed": deleted}


@app.post("/corrections")
def record_correction(req: RecordDecisionRequest) -> dict:
    try:
        row_id = _correction_db.record_decision(
            frame_id=req.frame_id, bbox=req.bbox, decision=req.decision,
            detected_class=req.detected_class, confidence=req.confidence,
            context=req.context, reviewer=req.reviewer,
        )
    except DataStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"id": row_id}


@app.get("/corrections")
def get_corrections(frame_id: Optional[str] = None, decision: Optional[str] = None, limit: int = 100) -> dict:
    try:
        records = _correction_db.get_decisions(frame_id=frame_id, decision=decision, limit=limit)
    except DataStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"count": len(records), "corrections": [r.__dict__ for r in records]}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "faces_enrolled": _face_db.count(), "corrections_logged": _correction_db.count()}
