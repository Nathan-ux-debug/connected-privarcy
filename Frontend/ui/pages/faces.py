"""Face Registry (D2): consented participants. Registration requires explicit consent."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QVBoxLayout, QWidget

from ..widgets import Banner, Card, EmptyState, Page, Pill, button, clear_layout, label
from ..workers import FuncWorker


class FacesPage(Page):
    def __init__(self, app) -> None:
        super().__init__("Face Registry", "People who have consented to appear unredacted. Everyone else's face is redacted.")
        self.app = app
        self.photo_path: str | None = None
        self.worker = None
        self.banner = Banner()
        self.body.addWidget(self.banner)
        self.policy_banner = Banner()
        self.body.addWidget(self.policy_banner)

        cols = QHBoxLayout()
        cols.setSpacing(18)
        form = Card("Register a participant")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Full name")
        form.add(label("Name", "Muted"))
        form.add(self.name_edit)
        self.photo_preview = QLabel("No photo")
        self.photo_preview.setAlignment(Qt.AlignCenter)
        self.photo_preview.setFixedHeight(170)
        self.photo_preview.setStyleSheet("background: #F4F4F8; border: 1px dashed #D4D4DD; border-radius: 8px; color: #9097A3;")
        form.add(self.photo_preview)
        form.add(button("Choose photo…", "", "upload", callback=self.choose_photo))
        form.add(label("Use one person, good light, face unobstructed. Larger, sharper photos register better.", "Faint", wrap=True))
        self.consent_box = QCheckBox("Consent confirmed")
        self.consent_box.setStyleSheet("QCheckBox { spacing: 8px; }")
        form.add(self.consent_box)
        form.add(label("I confirm this person gave informed consent to be registered and shown unredacted.", "Faint", wrap=True))
        self.register_btn = button("Register", "Primary", callback=self.register)
        row = QHBoxLayout()
        row.addWidget(self.register_btn)
        row.addStretch()
        form.add(row)
        cols.addWidget(form, 4)

        right = QVBoxLayout()
        right.setSpacing(14)
        self.list_card = Card("Registered participants")
        self.list_box = QVBoxLayout()
        self.list_box.setSpacing(0)
        self.list_card.add(self.list_box)
        right.addWidget(self.list_card)
        self.info_card = Card("How matching works")
        self.info_label = label("", "Muted", wrap=True)
        self.info_card.add(self.info_label)
        right.addWidget(self.info_card)
        right.addStretch()
        cols.addLayout(right, 6)
        self.body.addLayout(cols)

    def choose_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a photo", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if path:
            self.photo_path = path
            pm = QPixmap(path)
            self.photo_preview.setPixmap(pm.scaledToHeight(160, Qt.SmoothTransformation))

    def register(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            self.banner.show_message("Enter the person's name.", "warn")
            return
        if not self.photo_path:
            self.banner.show_message("Choose a photo first.", "warn")
            return
        if not self.consent_box.isChecked():
            self.banner.show_message("Consent is required. Tick the consent box to continue.", "warn")
            return
        svc = self.app.service
        code = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "participant"
        existing = {f["code"] for f in svc.list_faces()}
        base, n = code, 2
        while code in existing:
            code, n = f"{base}_{n}", n + 1
        dest = svc.registry_dir / f"{code}{Path(self.photo_path).suffix.lower() or '.jpg'}"
        try:
            shutil.copy2(self.photo_path, dest)
        except OSError as exc:
            self.banner.show_message(f"Could not store the reference photo: {exc}", "danger")
            return
        self.register_btn.setEnabled(False)
        self.banner.show_message("Detecting the face and creating its signature…", "info")
        self.worker = FuncWorker(svc.register_face, code, name, str(dest))
        self.worker.done.connect(lambda _r, d=dest: self._registered(name))
        self.worker.error.connect(lambda m, d=dest: self._reg_failed(m, d))
        self.worker.finished.connect(lambda: (self.register_btn.setEnabled(True), setattr(self, "worker", None)))
        self.worker.start()

    def _registered(self, name: str) -> None:
        self.banner.show_message(f"{name} is registered.", "ok")
        self.name_edit.clear()
        self.consent_box.setChecked(False)
        self.photo_path = None
        self.photo_preview.setPixmap(QPixmap())
        self.photo_preview.setText("No photo")
        self.on_show()

    def _reg_failed(self, message: str, dest: Path) -> None:
        dest.unlink(missing_ok=True)       # don't keep a reference photo for a failed registration
        self.banner.show_message(f"Registration failed: {message}", "danger")

    def verify(self, code: str) -> None:
        res = self.app.service.verify_face(code)
        self.banner.show_message(f"{res['message']} (similarity {res['confidence']:.0f}%)", "ok" if res["matched"] else "warn")

    def remove(self, code: str, name: str) -> None:
        if QMessageBox.question(self, "Remove participant", f"Remove {name}? Their face signature and photo will be deleted.") != QMessageBox.Yes:
            return
        svc = self.app.service
        meta = next((f for f in svc.list_faces() if f["code"] == code), None)
        svc.remove_face(code)
        if meta and meta.get("image_path"):
            Path(meta["image_path"]).unlink(missing_ok=True)
        self.banner.show_message(f"{name} was removed.", "ok")
        self.on_show()

    def on_show(self) -> None:
        svc = self.app.service
        clear_layout(self.list_box)
        faces = svc.list_faces()
        if not faces:
            self.list_box.addWidget(EmptyState("face", "No one is registered", "With nobody registered, every detected face is redacted."))
        for f in faces:
            row = QFrame()
            row.setObjectName("Row")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 9, 0, 9)
            thumb = QLabel()
            thumb.setFixedSize(44, 44)
            thumb.setStyleSheet("background: #F1F1F5; border-radius: 22px;")
            pm = QPixmap(f.get("image_path") or "")
            if not pm.isNull():
                thumb.setPixmap(pm.scaled(44, 44, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
            rl.addWidget(thumb)
            col = QVBoxLayout()
            col.setSpacing(1)
            nm = label(f["name"])
            nm.setStyleSheet("font-weight: 500;")
            col.addWidget(nm)
            col.addWidget(label(f"{f['code']}  ·  {f['registered_at'][:10]}  ·  {f.get('consent', '')}", "Faint", wrap=True))
            rl.addLayout(col, 1)
            rl.addWidget(button("Verify", "Ghost", callback=lambda c=f["code"]: self.verify(c)))
            rl.addWidget(button("Remove", "Ghost", callback=lambda c=f["code"], n=f["name"]: self.remove(c, n)))
            self.list_box.addWidget(row)

        cfg = svc.config
        if cfg.registered_face_policy == "exempt":
            self.policy_banner.show_message("Policy: registered faces are NOT redacted. Change this in Settings to redact everyone.", "info")
        else:
            self.policy_banner.show_message("Policy: every face is redacted, including registered people (registration does not exempt).", "warn")
        d = svc.diagnostics()["face_recognition"]
        self.info_label.setText(
            "Each detected face is compared with the registered signatures. If it does not match closely enough — or anything goes wrong — "
            "the face stays redacted.\n\n"
            f"Face detectors in use: {', '.join(d['backends']) or 'none'}.  Matching: {d['embedding']}, strict threshold "
            f"{svc.face_matcher.similarity_threshold:.2f}.\n\n"
            "Limitation: this baseline matcher is much weaker than a deep face-recognition model. Expect registered people to be "
            "redacted sometimes (safe), especially with a different angle or lighting.")
