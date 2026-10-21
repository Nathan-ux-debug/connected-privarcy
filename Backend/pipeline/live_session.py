"""Live camera session.

A background thread captures frames, runs the SAME FrameEngine as recorded
video, and publishes only REDACTED frames. The un-redacted frame never leaves
this thread, so neither the preview nor the recording can show raw pixels.

Measured, not assumed: ``status()`` reports the real processing FPS (rolling),
capture->preview latency and per-frame inference time.

Errors are surfaced, never swallowed: camera cannot open, camera stops
delivering frames / is unplugged (-> state 'error'), recording cannot start.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .config import PipelineConfig
from .frame_engine import FrameEngine

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_READ_FAILURES = 45


class LiveSession:
    def __init__(
        self,
        model,
        config: PipelineConfig,
        camera_index: int,
        width: int,
        height: int,
        fps: float,
        output_path: Optional[str | Path] = None,
        consent=None,
        ocr_provider=None,
        capture_factory: Optional[Callable] = None,
    ) -> None:
        self.config = config.copy()
        self.requested = (width, height, fps)
        self.actual = None
        self.state = "starting"            # starting | live | stopped | error
        self.error: Optional[str] = None
        self.output_path = Path(output_path) if output_path else None
        self.stats = {"frames": 0, "redacted": 0, "flagged": 0, "held": 0, "low_confidence": 0}
        self.audit_log: deque = deque(maxlen=100)

        self._latest: Optional[np.ndarray] = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._proc_times: deque = deque(maxlen=60)       # seconds per processed frame (inference+redaction)
        self._frame_stamps: deque = deque(maxlen=60)     # wall-clock of processed frames
        self._latencies: deque = deque(maxlen=60)        # capture -> redacted-available
        self._writer = None

        self.engine = FrameEngine(model, self.config, source_name=f"live-{camera_index}", consent=consent,
                                  ocr_provider=ocr_provider, live=True)

        factory = capture_factory or self._default_capture
        self._capture = factory(camera_index)
        if self._capture is None or not self._capture.isOpened():
            self._fail("Could not open the camera. It may be in use by another application, unplugged, or blocked "
                       "by Windows camera privacy settings.")
            return
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        self.actual = (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or width,
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height,
            float(self._capture.get(cv2.CAP_PROP_FPS) or fps),
        )
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(
                str(self.output_path), cv2.VideoWriter_fourcc(*"mp4v"), min(self.actual[2], 30.0) or 15.0,
                (self.actual[0], self.actual[1]),
            )
            if not self._writer.isOpened():
                self._capture.release()
                self._fail("Could not create the recording file.")
                return
        self.state = "live"
        self._thread = threading.Thread(target=self._run, name="privarcy-live", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    @staticmethod
    def _default_capture(index: int):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
        return cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)

    def _fail(self, message: str) -> None:
        self.state, self.error = "error", message
        logger.error("Live session error: %s", message)

    @property
    def is_live(self) -> bool:
        return self.state == "live"

    # ------------------------------------------------------------------
    def _run(self) -> None:
        failures = 0
        frame_number = 0
        interval = max(1, int(self.config.detection_interval))
        try:
            while not self._stop.is_set():
                ok, frame = self._capture.read()
                captured_at = time.perf_counter()
                if not ok or frame is None:
                    failures += 1
                    if failures >= MAX_CONSECUTIVE_READ_FAILURES:
                        self._fail("The camera stopped delivering frames (disconnected or taken by another app).")
                        break
                    time.sleep(0.02)
                    continue
                failures = 0
                frame_number += 1
                started = time.perf_counter()
                try:
                    out = self.engine.process(frame, frame_number, self.actual[2] if self.actual else 30.0,
                                              run_detection=(frame_number % interval == 1 or interval == 1))
                except Exception as exc:
                    # Fail CLOSED: never show the raw frame if processing failed.
                    logger.exception("Live frame processing failed")
                    self._fail(f"Processing failed, preview stopped to avoid showing unredacted video: {exc}")
                    break
                done = time.perf_counter()
                self._proc_times.append(done - started)
                self._latencies.append(done - captured_at)
                self._frame_stamps.append(done)

                self.stats["frames"] += 1
                self.stats["redacted"] += out.redactions
                self.stats["flagged"] += out.flagged_new
                self.stats["held"] += out.held
                self.stats["low_confidence"] += sum(out.low_confidence.values())
                for record in out.records:
                    if record["status"] in ("auto_redacted", "pending"):
                        self.audit_log.append({"description": f"{record['label']} {record['status'].replace('_', ' ')} "
                                                              f"({record['confidence']:.0%})", "kind": record["label"].upper()})
                with self._latest_lock:
                    self._latest = out.frame
                if self._writer is not None:
                    self._writer.write(out.frame)
        finally:
            if self.state == "live":
                self.state = "stopped"

    # ------------------------------------------------------------------
    def latest_frame(self) -> Optional[np.ndarray]:
        """The most recent REDACTED frame (a copy), or None."""
        with self._latest_lock:
            return None if self._latest is None else self._latest.copy()

    def status(self) -> dict:
        n = len(self._frame_stamps)
        fps = 0.0
        if n >= 2:
            span = self._frame_stamps[-1] - self._frame_stamps[0]
            fps = (n - 1) / span if span > 0 else 0.0
        return {
            "state": self.state, "error": self.error, "fps": fps,
            "latency_ms": (sum(self._latencies) / len(self._latencies) * 1000.0) if self._latencies else None,
            "inference_ms": (sum(self._proc_times) / len(self._proc_times) * 1000.0) if self._proc_times else None,
            "frames": self.stats["frames"], "resolution": self.actual,
            "target_fps_met": bool(fps >= 15.0) if n >= 10 else None,
        }

    def stop(self) -> None:
        self._stop.set()
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        try:
            if self._capture is not None:
                self._capture.release()
        finally:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            if self.state != "error":
                self.state = "stopped"
