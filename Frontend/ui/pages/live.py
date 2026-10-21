"""Live Camera: preview of REDACTED frames only, real FPS/latency, honest status."""

from __future__ import annotations

from PyQt5.QtCore import QTimer, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QListWidget, QVBoxLayout, QWidget

from ..widgets import Banner, Card, Page, Pill, PreviewFrame, StatTile, bgr_to_pixmap, button, label
from ..workers import FuncWorker

RESOLUTIONS = [("640 × 480", 640, 480), ("1280 × 720", 1280, 720), ("1920 × 1080", 1920, 1080)]


class LivePage(Page):
    def __init__(self, app) -> None:
        super().__init__("Live Camera", "Webcam monitoring with the same detection and redaction rules as recorded video. "
                         "Only redacted frames are ever shown or recorded.")
        self.app = app
        self.scan_worker = None
        self.start_worker = None
        self.session = None
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)

        self.banner = Banner()
        self.body.addWidget(self.banner)

        cols = QHBoxLayout()
        cols.setSpacing(18)
        left = QVBoxLayout()
        left.setSpacing(12)
        self.preview = PreviewFrame("Camera is off", 380)
        left.addWidget(self.preview, 1)
        ctl = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(150)
        self.camera_combo.addItem("Scanning for cameras…")
        ctl.addWidget(self.camera_combo)
        self.res_combo = QComboBox()
        for text, *_ in RESOLUTIONS:
            self.res_combo.addItem(text)
        ctl.addWidget(self.res_combo)
        self.rescan_btn = button("", "", "refresh", callback=self.scan)
        self.rescan_btn.setToolTip("Rescan cameras")
        ctl.addWidget(self.rescan_btn)
        ctl.addStretch()
        left.addLayout(ctl)
        ctl2 = QHBoxLayout()
        self.start_btn = button("Start", "Primary", "play", callback=self.start)
        self.stop_btn = button("Stop", "Danger", "stop", "#D92D20", callback=self.stop)
        self.stop_btn.setEnabled(False)
        ctl2.addWidget(self.start_btn)
        ctl2.addWidget(self.stop_btn)
        ctl2.addSpacing(10)
        self.ocr_box = QCheckBox("Read text (OCR) — slow")
        self.ocr_box.setChecked(app.service.config.live_ocr_enabled)
        self.ocr_box.toggled.connect(lambda v: app.service.update_settings(live_ocr_enabled=v))
        ctl2.addWidget(self.ocr_box)
        ctl2.addStretch()
        left.addLayout(ctl2)
        cols.addLayout(left, 7)

        right = QVBoxLayout()
        right.setSpacing(12)
        status = Card("Status")
        st_row = QHBoxLayout()
        self.state_pill = Pill("Stopped", "neutral")
        st_row.addWidget(self.state_pill)
        self.state_text = label("Start the camera to begin.", "Muted", wrap=True)
        st_row.addWidget(self.state_text, 1)
        status.add(st_row)
        grid = QHBoxLayout()
        self.t_fps = StatTile("Measured FPS")
        self.t_lat = StatTile("Latency (ms)")
        grid.addWidget(self.t_fps)
        grid.addWidget(self.t_lat)
        status.add(grid)
        grid2 = QHBoxLayout()
        self.t_inf = StatTile("Inference / frame (ms)")
        self.t_red = StatTile("Redactions")
        grid2.addWidget(self.t_inf)
        grid2.addWidget(self.t_red)
        status.add(grid2)
        self.target_label = label("", "Faint", wrap=True)
        status.add(self.target_label)
        self.res_label = label("", "Faint")
        status.add(self.res_label)
        right.addWidget(status)

        log = Card("Recent events")
        self.events = QListWidget()
        self.events.setMinimumHeight(110)
        self.events.setMaximumHeight(170)
        log.add(self.events)
        right.addWidget(log)

        ev = Card("Verification on this computer")
        self.verify_label = label("", "Muted", wrap=True)
        ev.add(self.verify_label)
        row = QHBoxLayout()
        self.open_rec = button("Open last recording", "", "folder", callback=self._open_recording)
        row.addWidget(self.open_rec)
        row.addStretch()
        ev.add(row)
        right.addWidget(ev)
        right.addStretch()
        cols.addLayout(right, 4)
        self.body.addLayout(cols)

    # ------------------------------------------------------------- cameras
    def scan(self) -> None:
        if self.session is not None or self.scan_worker is not None:
            return
        self.camera_combo.clear()
        self.camera_combo.addItem("Scanning for cameras…")
        self.start_btn.setEnabled(False)
        self.scan_worker = FuncWorker(self.app.service.discover_cameras)
        self.scan_worker.done.connect(self._scan_done)
        self.scan_worker.error.connect(lambda m: self._scan_done([]))
        self.scan_worker.finished.connect(lambda: setattr(self, "scan_worker", None))
        self.scan_worker.start()

    def _scan_done(self, cameras) -> None:
        self.camera_combo.clear()
        if not cameras:
            self.camera_combo.addItem("No camera found")
            self.camera_combo.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.banner.show_message("No camera was detected. Check it is plugged in, not used by another app (Teams/Zoom/Camera), "
                                     "and that Windows > Privacy > Camera allows desktop apps. Then press the refresh button.", "warn")
            return
        self.banner.hide()
        self.camera_combo.setEnabled(True)
        for cam in cameras:
            self.camera_combo.addItem(cam["label"], cam["index"])
        self.start_btn.setEnabled(True)

    # ------------------------------------------------------------- session
    def start(self) -> None:
        if self.session is not None or self.camera_combo.currentData() is None or self.start_worker is not None:
            return
        self.banner.hide()
        _text, w, h = RESOLUTIONS[self.res_combo.currentIndex()]
        index = self.camera_combo.currentData()
        self.start_btn.setEnabled(False)
        for wdg in (self.camera_combo, self.res_combo, self.rescan_btn, self.ocr_box):
            wdg.setEnabled(False)
        self.state_pill.set("Starting", "info")
        self.state_text.setText("Opening the camera…")
        # Opening a camera can block for seconds on Windows: never on the UI thread.
        self.start_worker = FuncWorker(self.app.service.start_live, index, w, h, 30)
        self.start_worker.done.connect(self._started)
        self.start_worker.error.connect(lambda m: self._start_failed(m))
        self.start_worker.finished.connect(lambda: setattr(self, "start_worker", None))
        self.start_worker.start()

    def _start_failed(self, message: str) -> None:
        self.banner.show_message(message, "danger")
        self.state_pill.set("Error", "danger")
        self.state_text.setText("Camera could not be started.")
        self.start_btn.setEnabled(self.camera_combo.currentData() is not None)
        for wdg in (self.camera_combo, self.res_combo, self.rescan_btn, self.ocr_box):
            wdg.setEnabled(True)

    def _started(self, session) -> None:
        if session.state == "error":
            msg = session.error
            self.app.service.stop_live()
            self._start_failed(msg)
            return
        self.session = session
        self.stop_btn.setEnabled(True)
        self.events.clear()
        self.app.set_busy(True)
        self.timer.start()

    def stop(self) -> None:
        self.timer.stop()
        if self.session is not None:
            self.app.service.stop_live()
            self.session = None
        self.preview.image.clear_image("Camera is off")
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(self.camera_combo.currentData() is not None)
        for wdg in (self.camera_combo, self.res_combo, self.rescan_btn, self.ocr_box):
            wdg.setEnabled(True)
        self.state_pill.set("Stopped", "neutral")
        self.state_text.setText("Camera released.")
        self.app.set_busy(False)
        self._refresh_verification()

    def _tick(self) -> None:
        s = self.session
        if s is None:
            return
        frame = s.latest_frame()
        if frame is not None:
            self.preview.image.set_pixmap(bgr_to_pixmap(frame))
        st = s.status()
        if st["state"] == "error":
            self.banner.show_message(st["error"] or "Camera error.", "danger")
            self.stop()
            self.state_pill.set("Error", "danger")
            self.state_text.setText("The session ended because of an error.")
            return
        self.state_pill.set("Live", "ok")
        self.state_text.setText("Showing redacted video only.")
        self.t_fps.set_value(f"{st['fps']:.1f}")
        self.t_lat.set_value("—" if st["latency_ms"] is None else f"{st['latency_ms']:.0f}")
        self.t_inf.set_value("—" if st["inference_ms"] is None else f"{st['inference_ms']:.0f}")
        self.t_red.set_value(str(s.stats["redacted"]))
        if st["resolution"]:
            self.res_label.setText(f"Camera delivering {st['resolution'][0]}×{st['resolution'][1]}")
        met = st["target_fps_met"]
        self.target_label.setText("Measuring…" if met is None else
                                  ("Meets the ~15 FPS target on this machine." if met else "Below the ~15 FPS target. Try 640×480 or a larger detection interval in Settings."))
        self.events.clear()
        for e in list(s.audit_log)[-8:][::-1]:
            self.events.addItem(e["description"])

    def _refresh_verification(self) -> None:
        runs = self.app.service.live_measurements()
        if not runs:
            self.verify_label.setText("No real camera session (50+ frames) has been measured on this computer yet. "
                                      "Live capture is unverified until you run one.")
            return
        last = runs[-1]
        self.verify_label.setText(
            f"Last measured session: {last['when']}\n{last['frames']} frames at {last['resolution'][0]}×{last['resolution'][1]}  ·  "
            f"{last['avg_fps']} FPS  ·  {last['latency_ms']} ms latency  ·  OCR {'on' if last['ocr'] else 'off'}")

    def _open_recording(self) -> None:
        path = self.app.service.last_live_recording
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            self.banner.show_message("No recording yet. A recording is kept when a session captures frames.", "info")

    def on_show(self) -> None:
        self._refresh_verification()
        if self.session is None and self.scan_worker is None and self.camera_combo.currentData() is None:
            self.scan()

    def shutdown(self) -> None:
        self.timer.stop()
        if self.session is not None:
            self.app.service.stop_live()
            self.session = None
