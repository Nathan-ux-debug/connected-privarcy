"""Video Processing: select -> validate -> configure -> process -> results."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QProgressBar, QRadioButton, QVBoxLayout, QWidget,
)

from ..widgets import (
    Banner, Card, Page, Pill, PreviewFrame, StatTile, bgr_to_pixmap, button, clear_layout, human_bytes, human_duration, label,
)
from ..workers import ProcessingWorker
from Backend.service import SUPPORTED_VIDEO_EXTENSIONS

CLASS_LABELS = [("face", "Faces"), ("id_card", "ID cards"), ("credit_card", "Payment cards"),
                ("screen", "Screens"), ("license_plate", "License plates")]


class VideoPage(Page):
    def __init__(self, app) -> None:
        super().__init__("Video Processing", "Upload a recorded video. Sensitive regions above 80% confidence are redacted automatically; "
                         "50–80% go to the review queue.")
        self.app = app
        self.video_path: str | None = None
        self.info: dict | None = None
        self.worker: ProcessingWorker | None = None
        self.last_result = None

        self.banner = Banner()
        self.body.addWidget(self.banner)

        cols = QHBoxLayout()
        cols.setSpacing(18)

        # ---------------- left: source + options
        left = QVBoxLayout()
        left.setSpacing(14)
        src = Card("Source video")
        self.preview = PreviewFrame("No video selected", 220)
        src.add(self.preview)
        self.file_label = label("", "Muted", wrap=True)
        src.add(self.file_label)
        self.meta_label = label("", "Faint", wrap=True)
        src.add(self.meta_label)
        row = QHBoxLayout()
        self.choose_btn = button("Choose video…", "", "folder", callback=self.choose)
        row.addWidget(self.choose_btn)
        row.addStretch()
        src.add(row)
        src.add(label("Supported: " + ", ".join(e.lstrip(".").upper() for e in SUPPORTED_VIDEO_EXTENSIONS), "Faint", wrap=True))
        left.addWidget(src)

        opts = Card("Redaction options")
        opts.add(label("Style", "Muted"))
        self.method_buttons = {}
        mrow = QHBoxLayout()
        for m in ("Blur", "Pixelate", "Solid Mask"):
            rb = QRadioButton(m)
            rb.toggled.connect(self._options_changed)
            self.method_buttons[m] = rb
            mrow.addWidget(rb)
        mrow.addStretch()
        opts.add(mrow)
        opts.add(label("Detect", "Muted"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        self.class_boxes = {}
        for i, (key, text) in enumerate(CLASS_LABELS):
            cb = QCheckBox(text)
            cb.toggled.connect(self._options_changed)
            self.class_boxes[key] = cb
            grid.addWidget(cb, i // 3, i % 3)
        opts.add(grid)
        self.ocr_box = QCheckBox("Find and redact sensitive text (OCR)")
        self.ocr_box.toggled.connect(self._options_changed)
        opts.add(self.ocr_box)
        self.ocr_note = label("", "Faint", wrap=True)
        opts.add(self.ocr_note)
        left.addWidget(opts)
        left.addStretch()
        cols.addLayout(left, 5)

        # ---------------- right: run + results
        right = QVBoxLayout()
        right.setSpacing(14)
        run = Card("Processing")
        top = QHBoxLayout()
        self.status_pill = Pill("Idle", "neutral")
        top.addWidget(self.status_pill)
        self.status_text = label("Choose a video to begin.", "Muted")
        top.addWidget(self.status_text, 1)
        run.add(top)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        run.add(self.progress)
        grid2 = QGridLayout()
        grid2.setSpacing(10)
        self.tiles = {k: StatTile(c) for k, c in (("frames", "Frames"), ("detections", "Detections"), ("redactions", "Redactions"),
                                                  ("flagged", "For review"), ("fps", "Speed (fps)"), ("eta", "Time left"))}
        for i, t in enumerate(self.tiles.values()):
            grid2.addWidget(t, i // 3, i % 3)
        run.add(grid2)
        btns = QHBoxLayout()
        self.start_btn = button("Start processing", "Primary", "play", callback=self.start)
        self.cancel_btn = button("Cancel", "Danger", "stop", "#D92D20", callback=self.cancel)
        self.cancel_btn.setEnabled(False)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.cancel_btn)
        btns.addStretch()
        run.add(btns)
        right.addWidget(run)

        self.result_card = Card("Result")
        self.result_body = QVBoxLayout()
        self.result_card.add(self.result_body)
        self.result_card.hide()
        right.addWidget(self.result_card)
        right.addStretch()
        cols.addLayout(right, 6)
        self.body.addLayout(cols)
        self.body.addStretch()
        self._load_options()

    # ---------------------------------------------------------------- options
    def _load_options(self) -> None:
        cfg = self.app.service.config
        self._loading = True
        for m, rb in self.method_buttons.items():
            rb.setChecked(m == cfg.redaction_method)
        for key, cb in self.class_boxes.items():
            cb.setChecked(key in cfg.enabled_classes)
        self.ocr_box.setChecked(cfg.ocr_enabled)
        self._loading = False
        self._refresh_ocr_note()

    def _refresh_ocr_note(self) -> None:
        st = self.app.service.ocr_provider.status()
        if st["available"]:
            self.ocr_note.setText("OCR reads text inside document, screen and card regions; only text matching sensitive patterns is redacted.")
        else:
            self.ocr_note.setText("OCR is unavailable in this environment, so sensitive text on documents will NOT be detected. " + st["reason"])
            self.ocr_note.setStyleSheet("color: #C77700;")

    def _options_changed(self) -> None:
        if getattr(self, "_loading", False):
            return
        method = next((m for m, rb in self.method_buttons.items() if rb.isChecked()), "Blur")
        classes = {k for k, cb in self.class_boxes.items() if cb.isChecked()}
        self.app.service.update_settings(redaction_method=method, enabled_classes=classes, ocr_enabled=self.ocr_box.isChecked())
        self.app.settings_changed()

    # ------------------------------------------------------------------ source
    def choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a video", "",
                                              "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.wmv);;All files (*)")
        if path:
            self.set_video(path)

    def set_video(self, path: str) -> None:
        if self.worker is not None:
            self.banner.show_message("A job is running. Wait for it to finish or cancel it first.", "warn")
            return
        self.banner.hide()
        self.result_card.hide()
        info = self.app.service.validate_video(path)
        if not info.get("ok"):
            self.video_path, self.info = None, None
            self.preview.image.clear_image("Cannot use this file")
            self.file_label.setText(Path(path).name)
            self.meta_label.setText("")
            self.banner.show_message(f"This file can't be processed: {info.get('error')}\n"
                                     "Try re-exporting it as MP4 (H.264), then select it again.", "danger")
            self.start_btn.setEnabled(False)
            self._set_status("Idle", "neutral", "Choose a valid video to begin.")
            return
        self.video_path, self.info = path, info
        self.file_label.setText(Path(path).name)
        self.meta_label.setText(f"{info['width']}×{info['height']}  ·  {info['fps']:.2f} fps  ·  {info['total_frames']} frames  ·  "
                                f"{human_duration(info['duration'])}  ·  {human_bytes(info['size_bytes'])}")
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        self.preview.image.set_pixmap(bgr_to_pixmap(frame) if ok else None)
        self.start_btn.setEnabled(True)
        self._set_status("Ready", "accent", "Ready to process.")
        for t in self.tiles.values():
            t.set_value("—")
        self.progress.setValue(0)

    # --------------------------------------------------------------------- run
    def _set_status(self, text: str, kind: str, detail: str) -> None:
        self.status_pill.set(text, kind)
        self.status_text.setText(detail)

    def start(self) -> None:
        if not self.video_path or self.worker is not None:
            return
        self.banner.hide()
        self.result_card.hide()
        self.worker = ProcessingWorker(self.app.service, self.video_path)
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        for w in (self.start_btn, self.choose_btn, self.ocr_box, *self.method_buttons.values(), *self.class_boxes.values()):
            w.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._set_status("Running", "info", "Loading models and processing frames…")
        self.app.set_busy(True)
        self.worker.start()

    def cancel(self) -> None:
        if self.worker:
            self.cancel_btn.setEnabled(False)
            self._set_status("Cancelling", "warn", "Finishing the current frame…")
            self.worker.cancel()

    def _on_progress(self, p: dict) -> None:
        total = max(p["total"], 1)
        self.progress.setValue(int(p["current"] / total * 1000))
        self.tiles["frames"].set_value(f"{p['current']}/{p['total']}")
        self.tiles["detections"].set_value(str(p["detections"]))
        self.tiles["redactions"].set_value(str(p["redactions"]))
        self.tiles["flagged"].set_value(str(p["flagged"]))
        self.tiles["fps"].set_value(f"{p['fps']:.1f}")
        self.tiles["eta"].set_value(human_duration(p["eta"]))
        self.status_text.setText(f"Processing frame {p['current']} of {p['total']}")

    def _on_done(self, result) -> None:
        self.last_result = result
        if result.cancelled:
            self._set_status("Cancelled", "warn", f"Stopped at frame {result.frames}. Process the same file again to resume.")
        else:
            self._set_status("Completed", "ok", f"Finished in {human_duration(result.processing_time)} ({result.avg_fps:.1f} fps average).")
            self.progress.setValue(1000)
        self._show_result(result)

    def _on_failed(self, message: str) -> None:
        self._set_status("Failed", "danger", "Processing stopped.")
        self.banner.show_message(f"Processing failed: {message}\nNo partial file was kept. Check the video plays in another player, "
                                 "free some disk space, or try a different file.", "danger")

    def _on_finished(self) -> None:
        self.worker = None
        for w in (self.start_btn, self.choose_btn, self.ocr_box, *self.method_buttons.values(), *self.class_boxes.values()):
            w.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.app.set_busy(False)
        self.app.refresh_badges()

    def _show_result(self, r) -> None:
        clear_layout(self.result_body)
        self.result_card.show()
        pending = sum(1 for d in r.detections if d.get("status") == "pending")
        auto = r.redactions
        summary = r.summary
        self.result_body.addWidget(label(
            f"{r.frames} frames written  ·  {auto} redaction operations  ·  {pending} item(s) need review", "", wrap=True))
        extra = []
        if summary.get("held_redactions"):
            extra.append(f"{summary['held_redactions']} redactions came from held tracks (object missed for ≤5 frames)")
        if summary.get("text_redactions"):
            extra.append(f"{summary['text_redactions']} sensitive-text redactions")
        if extra:
            self.result_body.addWidget(label("  ·  ".join(extra), "Faint", wrap=True))
        for w in r.warnings:
            b = Banner()
            b.show_message(w, "warn")
            self.result_body.addWidget(b)
        grid = QGridLayout()
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        grid.addWidget(button("Open redacted video", "Primary", callback=lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(r.output_path))), 0, 0)
        grid.addWidget(button("Show in folder", "", "folder", callback=lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(r.output_path).parent)))), 0, 1)
        if pending:
            grid.addWidget(button(f"Review {pending} item(s)", "", "review", callback=lambda: self.app.navigate("Review Queue")), 1, 0)
        grid.setColumnStretch(2, 1)
        w = QWidget()
        w.setLayout(grid)
        self.result_body.addWidget(w)

    def on_show(self) -> None:
        if self.worker is None:
            self._load_options()

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(5000)
