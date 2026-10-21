"""Dashboard: hero + upload action, quick actions, real status, recent jobs, pending review."""

from __future__ import annotations

import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QVBoxLayout, QWidget

from .. import theme
from ..widgets import Card, EmptyState, Page, Pill, StatusDot, button, clear_layout, hline, label


class DashboardPage(Page):
    def __init__(self, app) -> None:
        super().__init__("", "")
        self.app = app
        # Dashboard replaces the standard header with a centred hero (like the reference).
        clear_layout(self.header_row)
        self.body.removeItem(self.header_row)
        self.body.setSpacing(22)

        hero = QVBoxLayout()
        hero.setSpacing(8)
        hero.setContentsMargins(0, 26, 0, 4)
        title = label("What are you protecting today?", "HeroTitle")
        title.setAlignment(Qt.AlignCenter)
        sub = label("Detect and redact faces, ID and payment cards, screens, plates and sensitive text in recorded "
                    "video or a live camera. Everything runs locally.", "HeroSub", wrap=True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setFixedWidth(620)
        sub.setMinimumHeight(46)
        hero.addWidget(title)
        hero.addWidget(sub, alignment=Qt.AlignHCenter)
        self.body.addLayout(hero)

        # prompt-style upload box
        prompt = QFrame()
        prompt.setObjectName("Prompt")
        prompt.setMaximumWidth(740)
        pl = QVBoxLayout(prompt)
        pl.setContentsMargins(14, 14, 14, 14)
        inner = QFrame()
        inner.setObjectName("PromptInner")
        il = QVBoxLayout(inner)
        il.setContentsMargins(16, 14, 14, 12)
        il.setSpacing(14)
        self.prompt_text = label("Choose a video to redact…  (MP4, MOV, AVI, MKV, M4V, WEBM, WMV)", "Muted")
        self.prompt_text.setStyleSheet(f"color: {theme.TEXT_FAINT}; font-size: 15px;")
        il.addWidget(self.prompt_text)
        row = QHBoxLayout()
        row.addWidget(button("Browse files", "", "folder", callback=self._browse))
        row.addStretch()
        self.send = button("→", "Send", callback=self._browse)
        row.addWidget(self.send)
        il.addLayout(row)
        pl.addWidget(inner)
        self.body.addWidget(prompt, alignment=Qt.AlignHCenter)

        chips = QHBoxLayout()
        chips.setSpacing(8)
        chips.addStretch()
        for text, icon, target in (("Process a video", "video", "Video Processing"), ("Start live camera", "camera", "Live Camera"),
                                   ("Open review queue", "review", "Review Queue"), ("Register a face", "face", "Face Registry")):
            b = button(text, "Chip", icon, theme.TEXT_MUTED, callback=lambda t=target: app.navigate(t))
            chips.addWidget(b)
        chips.addStretch()
        self.body.addLayout(chips)
        self.body.addSpacing(8)

        # lower panel: recent jobs | status
        panel = Card(padding=22)
        cols = QHBoxLayout()
        cols.setSpacing(24)
        left = QVBoxLayout()
        lh = QHBoxLayout()
        lh.addWidget(label("Recent jobs", "SectionTitle"))
        lh.addStretch()
        lh.addWidget(button("See all", "Ghost", callback=lambda: app.navigate("History & Outputs")))
        left.addLayout(lh)
        self.jobs_box = QVBoxLayout()
        self.jobs_box.setSpacing(0)
        left.addLayout(self.jobs_box)
        left.addStretch()
        cols.addLayout(left, 3)
        div = QFrame()
        div.setObjectName("VDivider")
        cols.addWidget(div)
        right = QVBoxLayout()
        right.setSpacing(10)
        right.addWidget(label("Pipeline status", "SectionTitle"))
        self.status_box = QVBoxLayout()
        self.status_box.setSpacing(9)
        right.addLayout(self.status_box)
        right.addSpacing(6)
        right.addWidget(hline())
        right.addSpacing(2)
        rh = QHBoxLayout()
        rh.addWidget(label("Pending review", "SectionTitle"))
        rh.addStretch()
        rh.addWidget(button("Open", "Ghost", callback=lambda: app.navigate("Review Queue")))
        right.addLayout(rh)
        self.review_label = label("", "Muted", wrap=True)
        right.addWidget(self.review_label)
        right.addStretch()
        cols.addLayout(right, 2)
        panel.add(cols)
        self.body.addWidget(panel)
        self.body.addStretch()

    # ------------------------------------------------------------------
    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a video", "",
                                              "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.wmv);;All files (*)")
        if path:
            self.app.open_video(path)

    def on_show(self) -> None:
        svc = self.app.service
        # ---- recent jobs (real manifests)
        clear_layout(self.jobs_box)
        jobs = svc.get_history(5)
        if not jobs:
            self.jobs_box.addWidget(EmptyState("video", "No jobs yet", "Process a video and it will appear here with its real results."))
        for job in jobs:
            row = QFrame()
            row.setObjectName("Row")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 9, 0, 9)
            col = QVBoxLayout()
            col.setSpacing(1)
            name = label(job.source_name)
            name.setStyleSheet("font-weight: 500;")
            col.addWidget(name)
            when = time.strftime("%b %d, %H:%M", time.localtime(job.created_at))
            col.addWidget(label(f"{when}  ·  {job.frames_processed} frames  ·  {job.redactions} redactions", "Faint"))
            rl.addLayout(col, 1)
            rl.addWidget(Pill(job.status.capitalize(), theme.STATUS_KIND.get(job.status, "neutral")))
            self.jobs_box.addWidget(row)

        # ---- real status from diagnostics
        diag = svc.diagnostics()
        clear_layout(self.status_box)
        det, ocr, face, red = diag["detector"], diag["ocr"], diag["face_recognition"], diag["redaction"]
        rows = [
            ("YOLO detector", "ok" if det["loaded"] else "danger", f"{len(det['classes'])} classes · {det['device']}"),
            ("OCR (TrOCR)", "ok" if ocr["available"] else "warn", ("ready" if ocr["loaded"] else "loads on first use") if ocr["available"] else "unavailable"),
            ("Face detection", "ok" if face["loaded"] else "danger", ", ".join(b.replace("FaceDetector", "").replace("Yolo", "YOLO") or "Haar" for b in face["backends"]) or "none"),
            ("Redaction", "ok", red["active"]),
        ]
        for name, kind, detail in rows:
            r = QHBoxLayout()
            r.setSpacing(8)
            r.addWidget(StatusDot(kind))
            r.addWidget(label(name))
            r.addStretch()
            r.addWidget(label(detail, "Faint"))
            w = QWidget()
            w.setLayout(r)
            self.status_box.addWidget(w)

        ov = svc.overview()
        if ov["pending_review"]:
            self.review_label.setText(f"{ov['pending_review']} detection(s) between 50% and 80% confidence are waiting for a decision.")
        else:
            self.review_label.setText("Nothing waiting. Uncertain detections from your jobs will appear here.")
