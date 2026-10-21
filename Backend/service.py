"""BackendService - the single facade the frontend talks to.

No Qt in here. Everything the UI shows must come from a method on this class,
so the UI can never display numbers the backend did not produce.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2

from Backend.data_store import CorrectionLogDB, FaceRegistryDB
from Backend.face_privacy import FaceEmbedder, FaceMatcher, build_face_detector, enroll_person
from Backend.face_privacy.consent import FaceConsentMatcher
from Backend.ocr.provider import OCRProvider
from Backend.pipeline.config import PipelineConfig, REDACTION_METHODS
from Backend.pipeline.job_runner import JobResult, JobRunner, probe_video
from Backend.pipeline.live_session import LiveSession
from Backend.pipeline.review_apply import apply_review_redactions
from Backend.pipeline.video_pipeline import VideoPipeline
from Backend.rule_refinement import RuleStore, evidence_from_log, propose_rules, validate_proposal
from Backend.storage import StorageManager

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm")


class BackendService:
    def __init__(self, backend_dir: Optional[Path] = None, model=None) -> None:
        self.backend_dir = Path(backend_dir) if backend_dir else Path(__file__).resolve().parent
        self.storage = StorageManager(self.backend_dir)
        self.startup_report = self.storage.startup_cleanup()

        self.config = PipelineConfig()
        self.rules = RuleStore(self.storage.rules_dir)
        self._apply_active_rules()

        self.face_registry = FaceRegistryDB(self.storage.d2_path)
        self.correction_db = CorrectionLogDB(self.storage.d4_path)
        self.face_embedder = FaceEmbedder()
        # 0.75 (library default) does not separate HOG vectors: unrelated textures score ~0.94 and
        # different natural images ~0.55-0.72. 0.90 is a conservative starting point: a registered person who is not
        # matched is simply blurred (safe); a stranger wrongly matched would be exposed (unsafe).
        self.face_matcher = FaceMatcher(similarity_threshold=0.90)
        self.ocr_provider = OCRProvider(model_name=self.config.ocr_model)

        self.pipeline = VideoPipeline(model=model, config=self.config, ocr_provider=self.ocr_provider)
        self.face_detector, self.face_detector_notes = build_face_detector(self.pipeline.model)
        self.consent = FaceConsentMatcher(self.face_registry, self.face_embedder, self.face_matcher)
        self.pipeline.consent = self.consent
        self.pipeline.face_detector = self.face_detector

        self.job_runner: Optional[JobRunner] = None
        self.live_session: Optional[LiveSession] = None
        self.last_live_recording: Optional[str] = None
        self.registry_dir = self.storage.d2_photos

    # ------------------------------------------------------------------ rules
    def _apply_active_rules(self) -> None:
        self.config.class_auto_redact_confidence = dict(self.rules.active().class_auto_redact_confidence)

    def update_settings(self, **values) -> None:
        if "ocr_model" in values and values["ocr_model"] != self.config.ocr_model:
            self.ocr_provider = OCRProvider(model_name=values["ocr_model"])
            self.pipeline.ocr_provider = self.ocr_provider
        self.config.update(**values)
        if "registered_face_policy" in values and values["registered_face_policy"] not in ("exempt", "redact_all"):
            self.config.registered_face_policy = "exempt"

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self) -> dict:
        names = self.pipeline.class_map
        torch = sys.modules.get("torch")      # already imported by ultralytics when the real model is loaded
        try:
            device = ("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else "n/a"
        except Exception:
            device = "unknown"
        ocr = self.ocr_provider.status()
        registry_count = self.face_registry.count()
        return {
            "detector": {
                "loaded": self.pipeline.model is not None, "device": device,
                "classes": names["names"], "class_map_ok": names["ok"], "warnings": names["warnings"],
            },
            "ocr": {
                "available": ocr["available"], "loaded": ocr["loaded"], "model": ocr["model"], "reason": ocr["reason"],
                "integrated": True, "localizer": "OpenCV morphological text-line localiser",
            },
            "face_recognition": {
                "loaded": self.face_detector is not None,
                "backends": getattr(self.face_detector, "backends", []),
                "notes": self.face_detector_notes, "registered_embeddings": registry_count,
                "embedding": "HOG (baseline, not deep-learning)", "last_error": self.consent.last_error,
            },
            "redaction": {"methods": list(REDACTION_METHODS), "active": self.config.redaction_method},
            "storage": self.storage.layout(),
            "rules": {"active_version": self.rules.active_version_number(),
                      "overrides": dict(self.config.class_auto_redact_confidence)},
            "opencv": {"version": getattr(cv2, "__version__", "?"), "file": getattr(cv2, "__file__", "?"),
                       "has_cascade": hasattr(cv2, "CascadeClassifier")},
            "startup": self.startup_report,
        }

    def overview(self) -> dict:
        jobs = self.storage.list_jobs(500)
        return {
            "completed_jobs": sum(1 for j in jobs if j.status == "completed"),
            "total_jobs": len(jobs),
            "registered_faces": len(self.list_faces()),
            "pending_review": sum(j.pending_review for j in jobs if j.status in ("completed", "cancelled")),
            "corrections_logged": self.correction_db.count(),
        }

    def get_history(self, limit: int = 50) -> list:
        return self.storage.list_jobs(limit)

    # ------------------------------------------------------------------ video
    @staticmethod
    def validate_video(path: str) -> dict:
        p = Path(path)
        if not p.is_file():
            return {"ok": False, "error": "File not found."}
        if p.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            return {"ok": False, "error": f"Unsupported extension '{p.suffix}'. Supported: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"}
        try:
            info = probe_video(p)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        info.update(ok=True, size_bytes=p.stat().st_size, name=p.name)
        return info

    def run_job(self, video_path: str, on_frame=None, on_detections=None, on_flagged=None, on_redactions=None) -> JobResult:
        self.job_runner = JobRunner(self.pipeline, self.storage)
        try:
            return self.job_runner.run(video_path, config=self.config, on_frame=on_frame, on_detections=on_detections,
                                       on_flagged=on_flagged, on_redactions=on_redactions)
        finally:
            self.job_runner = None

    def cancel_job(self) -> None:
        if self.job_runner is not None:
            self.job_runner.cancel()

    # ----------------------------------------------------------------- review
    def _log_decision(self, item: dict, decision: str, note: str = "") -> int:
        context = {
            "job_id": item.get("job_id"), "source": item.get("source"), "frame": item.get("frame"),
            "model_class": item.get("label"), "status_at_review": item.get("status"),
            "source_class": item.get("source_class"), "ocr_patterns": item.get("ocr_patterns"),
            "note": note, "rules_version": self.rules.active_version_number(),
        }
        return self.correction_db.record_decision(
            frame_id=str(item.get("id", "")), bbox=[float(v) for v in item.get("bbox", [0, 0, 0, 0])],
            decision=decision, detected_class=item.get("label") or item.get("classification") or "",
            confidence=float(item.get("confidence") or 0.0), context=json.dumps(context),
        )

    def confirm_detection(self, item: dict, note: str = "") -> dict:
        """Confirm = 'yes, redact this'. Applies the redaction to the output video, then logs."""
        frames = apply_review_redactions(item)
        row = self._log_decision(item, "accepted", note)
        return {"frames_redacted": frames, "log_id": row}

    def reject_detection(self, item: dict, note: str = "") -> dict:
        """Reject = 'false positive, leave visible'. Logged; the video is untouched."""
        return {"log_id": self._log_decision(item, "rejected", note)}

    def pending_review_items(self) -> list:
        """Every unresolved review item across jobs, newest job first."""
        items = []
        for job in self.storage.list_jobs(200):
            if job.pending_review <= 0:
                continue
            for item in self.storage.load_review_items(job.job_id):
                if item.get("status") == "pending":
                    item.setdefault("job_id", job.job_id)
                    item.setdefault("output_path", job.output_path)
                    item["job_name"] = job.source_name
                    items.append(item)
        return items

    def resolve_review_item(self, item: dict, decision: str, note: str = "") -> dict:
        """decision: 'confirmed' (redact) or 'rejected' (leave visible)."""
        result = self.confirm_detection(item, note) if decision == "confirmed" else self.reject_detection(item, note)
        job_id = item.get("job_id")
        if job_id:
            remaining = self.storage.set_review_status(job_id, item["id"], decision)
            job = self.storage.load_job(job_id)
            if job:
                job.pending_review = remaining
                self.storage.save_job(job)
            if remaining == 0:
                self.storage.release_job_frames(job_id)      # review finished -> temporary crops no longer needed
        return result

    def release_job_review_frames(self, job_id: str) -> None:
        job = self.storage.load_job(job_id)
        self.storage.release_job_frames(job_id)
        if job:
            job.pending_review = 0
            self.storage.save_job(job)

    def list_corrections(self, limit: int = 500) -> list:
        return self.correction_db.get_decisions(limit=limit)

    # ------------------------------------------------------------------- rules
    def rule_evidence(self):
        return evidence_from_log(self.correction_db)

    def rule_proposals(self) -> dict:
        evidence = self.rule_evidence()
        current = dict(self.config.class_auto_redact_confidence)
        out = []
        for proposal in propose_rules(evidence, current):
            report = validate_proposal(proposal, evidence, current)
            out.append({"proposal": proposal, "validation": report})
        return {"evidence": evidence, "items": out}

    def approve_proposal(self, proposal, report, approver: str = "reviewer"):
        overrides = dict(self.config.class_auto_redact_confidence)
        overrides[proposal.class_name] = proposal.proposed_threshold
        version = self.rules.activate(overrides, approver, proposal.summary(), self.rule_evidence().to_dict(), report.to_dict())
        self._apply_active_rules()
        return version

    def reject_proposal(self, proposal, approver: str = "reviewer") -> None:
        self.rules.reject(proposal.summary(), approver)

    def rollback_rules(self, version: int, approver: str = "reviewer"):
        v = self.rules.rollback(version, approver)
        self._apply_active_rules()
        return v

    # -------------------------------------------------------------------- faces
    def list_faces(self) -> list:
        records = {}
        for record in self.face_registry.all_embeddings():
            meta = self._face_metadata(record.source)
            records.setdefault(record.person_name, {
                "code": record.person_name, "name": meta.get("name", record.person_name),
                "image_path": meta.get("image_path"), "registered_at": record.created_at,
                "consent": meta.get("consent", "recorded at registration"),
            })
        return list(records.values())

    def register_face(self, code: str, name: str, image_path: str):
        if self.face_detector is None:
            raise RuntimeError("No face detector backend is available; see Settings > System status.")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read face image: {image_path}")
        source = json.dumps({"name": name, "image_path": str(image_path), "consent": f"registered {time.strftime('%Y-%m-%d %H:%M')}"})
        return enroll_person(code, image, self.face_registry, detector=self.face_detector,
                             embedder=self.face_embedder, source=source)

    def remove_face(self, code: str) -> int:
        return self.face_registry.remove_person(code)

    def verify_face(self, code: str) -> dict:
        records = self.face_registry.embeddings_for(code)
        if not records:
            return {"matched": False, "confidence": 0.0, "message": "Participant is not registered."}
        image_path = self._face_metadata(records[0].source).get("image_path")
        image = cv2.imread(image_path) if image_path else None
        if image is None:
            return {"matched": False, "confidence": 0.0, "message": "The registered photo could not be read."}
        faces = self.face_detector.detect(image) if self.face_detector else []
        if not faces:
            return {"matched": False, "confidence": 0.0, "message": "No face was found in the registered photo."}
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        emb = self.face_embedder.embed(image[y1:y2, x1:x2])
        res = self.face_matcher.match(face, emb, [(r.person_name, r.embedding) for r in records])
        return {"matched": res.is_known, "confidence": max(0.0, res.similarity) * 100,
                "message": "The registered photo matches the stored face." if res.is_known else "The registered photo does not match."}

    @staticmethod
    def _face_metadata(source) -> dict:
        try:
            meta = json.loads(source or "{}")
        except (TypeError, ValueError):
            meta = {}
        return meta if isinstance(meta, dict) else {}

    # --------------------------------------------------------------------- live
    def discover_cameras(self, max_index: int = 5) -> list:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
        names: list = []
        try:
            import importlib

            names = importlib.import_module("pygrabber.dshow_graph").FilterGraph().get_input_devices()
        except Exception:
            pass
        cameras = []
        for index in range(max_index):
            cap = None
            try:
                cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        cameras.append({"index": index, "label": names[index] if index < len(names) else f"Camera {index}"})
            finally:
                if cap is not None:
                    cap.release()
        return cameras

    def start_live(self, camera_index: int, width: int, height: int, fps: float) -> LiveSession:
        self.stop_live()
        path = self.storage.outputs_dir / f"live_{int(time.time())}.mp4"
        cfg = self.config.copy()
        ocr = self.ocr_provider if cfg.live_ocr_enabled else None
        session = LiveSession(self.pipeline.model, cfg, camera_index, width, height, fps, output_path=path,
                              consent=self.consent, ocr_provider=ocr)
        self.live_session = session
        return session

    def stop_live(self) -> Optional[str]:
        session, self.live_session = self.live_session, None
        if session is None:
            return None
        session.stop()
        self._record_live_measurement(session)
        if session.output_path and session.output_path.exists() and session.stats["frames"] > 0:
            self.last_live_recording = str(session.output_path)
        elif session.output_path:
            session.output_path.unlink(missing_ok=True)
        return self.last_live_recording

    def _record_live_measurement(self, session: LiveSession) -> None:
        """Keep REAL measurements from real sessions (>= 50 frames) as verification evidence."""
        status = session.status()
        if session.stats["frames"] < 50 or session.actual is None:
            return
        entry = {
            "when": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": session.stats["frames"],
            "avg_fps": round(status["fps"], 2), "latency_ms": round(status["latency_ms"] or 0, 1),
            "inference_ms": round(status["inference_ms"] or 0, 1), "resolution": list(session.actual),
            "detection_interval": session.config.detection_interval, "ocr": bool(session.config.live_ocr_enabled),
            "ended_with_error": session.error,
        }
        path = self.storage.live_measurements_path
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except Exception:
            data = []
        data.append(entry)
        path.write_text(json.dumps(data[-50:], indent=2), encoding="utf-8")

    def live_measurements(self) -> list:
        path = self.storage.live_measurements_path
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except Exception:
            return []

    def shutdown(self) -> None:
        self.stop_live()
        self.cancel_job()
