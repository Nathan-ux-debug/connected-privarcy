"""UI integration tests (offscreen Qt).

REAL: Qt widgets, BackendService, storage, pipeline, correction log, rule store.
MOCK: the YOLO model (scripted detections) and the camera (synthetic frames).
Screenshots are written to $PRIVARCY_SHOTS (default: Frontend/docs/screenshots_test).
"""
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget

from Backend.tests.fakes import FakeYolo, gradient_frame
from Backend.tests.test_live_session import FakeCap

SHOTS = Path(os.environ.get("PRIVARCY_SHOTS", Path(__file__).resolve().parents[1] / "docs" / "screenshots_test"))
SHOTS.mkdir(parents=True, exist_ok=True)


def pump(app, seconds=0.0, until=None, timeout=15.0):
    end = time.time() + (timeout if until else seconds)
    while time.time() < end:
        app.processEvents()
        if until and until():
            return True
        time.sleep(0.01)
    return until() if until else True


def make_video(path, frames=40, w=480, h=320):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for i in range(frames):
        f = gradient_frame(w, h)
        cv2.putText(f, "PrivARcy test", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        wr.write(f)
    wr.release()
    return path


@pytest.fixture(scope="module")
def qapp():
    from ui import theme
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(theme.stylesheet())
    return app


@pytest.fixture(scope="module")
def window(qapp, tmp_path_factory):
    import ui.shell as shell
    from Backend.service import BackendService

    tmp = tmp_path_factory.mktemp("proj")
    script = ([[("face", 0.95, (60, 60, 140, 150)), ("id_card", 0.62, (250, 120, 400, 230)), ("credit_card", 0.9, (200, 30, 300, 90))]] * 40
              + [[("face", 0.93, (60, 60, 140, 150))]] * 20000)
    service = BackendService(tmp / "Backend", model=FakeYolo(script))
    service.update_settings(ocr_enabled=False)

    class FakeLoader(QObject):
        loaded = pyqtSignal(object)
        failed = pyqtSignal(str)
        def start(self):
            self.loaded.emit(service)
    shell.BackendLoader = FakeLoader
    win = shell.MainWindow("")
    win.resize(1360, 820)
    win.show()
    win.tmp = tmp
    win.service_ref = service
    pump(qapp, 0.3)
    yield win
    win.close()


def shot(win, name):
    QApplication.processEvents()
    win.grab().save(str(SHOTS / f"{name}.png"))


def test_starts_and_dashboard_is_honest_when_empty(qapp, window):
    assert window.top_stack.currentIndex() == 2
    shot(window, "01_dashboard_empty")
    page = window.pages["Dashboard"]
    assert page.review_label.text().startswith("Nothing waiting")


def test_all_sections_navigable(qapp, window):
    for name in ["Dashboard", "Video Processing", "Live Camera", "Review Queue", "Face Registry",
                 "Correction Log", "History & Outputs", "Settings"]:
        window.navigate(name)
        pump(qapp, 0.05)
        assert window.page_stack.currentWidget() is window.pages[name]
        assert window.nav_buttons[name].isChecked()
    shot(window, "08_settings")


def test_invalid_video_gives_clear_error(qapp, window, tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"nope")
    window.open_video(str(bad))
    page = window.pages["Video Processing"]
    assert page.banner.isVisible() and "can't be processed" in page.banner.text.text()
    assert not page.start_btn.isEnabled()
    txt = tmp_path / "x.txt"; txt.write_text("hi")
    page.set_video(str(txt))
    assert "Unsupported extension" in page.banner.text.text()


def test_video_processing_end_to_end(qapp, window, tmp_path):
    src = make_video(tmp_path / "clip.mp4")
    window.open_video(str(src))
    page = window.pages["Video Processing"]
    assert page.start_btn.isEnabled() and "480×320" in page.meta_label.text()
    shot(window, "02_video_ready")
    page.start()
    assert not page.start_btn.isEnabled() and page.cancel_btn.isEnabled()
    shot(window, "03_video_running")
    assert pump(qapp, until=lambda: page.worker is None, timeout=60)
    assert page.status_pill.text() == "Completed"
    assert page.tiles["frames"].value_label.text() == "40/40"
    assert page.result_card.isVisible()
    assert page.start_btn.isEnabled()
    shot(window, "04_video_done")
    job = window.service.get_history(1)[0]
    assert job.status == "completed" and job.frames_processed == 40 and job.pending_review == 1
    out = cv2.VideoCapture(job.output_path); n = 0
    while out.read()[0]: n += 1
    assert n == 40


def test_dashboard_and_history_show_real_job(qapp, window):
    window.navigate("Dashboard"); pump(qapp, 0.1)
    d = window.pages["Dashboard"]
    assert "clip.mp4" in [w.text() for w in d.findChildren(type(d.prompt_text)) if w.text() == "clip.mp4"]
    assert d.review_label.text().startswith("1 detection")
    shot(window, "05_dashboard_data")
    window.navigate("History & Outputs"); pump(qapp, 0.1)
    shot(window, "07_history")
    assert "Review Queue" in window.nav_buttons and window.nav_buttons["Review Queue"].text().strip().endswith("1")


def test_review_confirm_logs_and_updates_video(qapp, window):
    window.navigate("Review Queue"); pump(qapp, 0.2)
    page = window.pages["Review Queue"]
    assert len(page.items) == 1 and page.current["label"] == "id_card"
    assert page.confirm_btn.isEnabled()
    shot(window, "06_review")
    page.note.setText("looks like a real ID")
    page.confirm_btn.click()
    assert pump(qapp, until=lambda: page.worker is None and not page.items, timeout=30)
    assert "Confirmed: redacted 40" in page.banner.text.text()
    recs = window.service.list_corrections()
    assert recs[0].decision == "accepted" and recs[0].confidence > 0.6 and recs[0].bbox[2] > recs[0].bbox[0]
    assert window.service.overview()["pending_review"] == 0


def test_faces_require_consent_and_register(qapp, window, tmp_path):
    window.navigate("Face Registry"); pump(qapp, 0.1)
    page = window.pages["Face Registry"]
    photo = tmp_path / "p.png"
    cv2.imwrite(str(photo), gradient_frame(300, 300))
    page.name_edit.setText("Alice Example"); page.photo_path = str(photo)
    page.register()
    assert "Consent is required" in page.banner.text.text()
    page.consent_box.setChecked(True)
    page.register()
    assert pump(qapp, until=lambda: "is registered" in page.banner.text.text() or "failed" in page.banner.text.text(), timeout=10)
    assert "Alice Example is registered" in page.banner.text.text()
    assert [f["name"] for f in window.service.list_faces()] == ["Alice Example"]
    shot(window, "09_faces")
    page.remove("Alice_Example", "Alice Example") if False else None


def test_corrections_page_proposal_flow(qapp, window):
    svc = window.service
    for i in range(12):
        svc.correction_db.record_decision(f"s:{i}", [1, 2, 3, 4], "accepted", "credit_card", 0.62 + i * 0.01)
    window.navigate("Correction Log"); pump(qapp, 0.2)
    page = window.pages["Correction Log"]
    shot(window, "10_corrections")
    data = svc.rule_proposals()
    assert [i["proposal"].class_name for i in data["items"]] == ["credit_card"]
    v = svc.approve_proposal(data["items"][0]["proposal"], data["items"][0]["validation"])
    assert svc.config.class_auto_redact_confidence == {"credit_card": 0.6}
    svc.rollback_rules(0)
    assert svc.config.class_auto_redact_confidence == {}
    page.on_show()


def test_settings_save_changes_backend_and_persist(qapp, window):
    window.navigate("Settings"); pump(qapp, 0.1)
    p = window.pages["Settings"]
    p.method.setCurrentText("Solid Mask")
    p.face_policy.setCurrentIndex(p.face_policy.findData("redact_all"))
    p.smooth.setValue(5)
    p.save()
    c = window.service.config
    assert c.redaction_method == "Solid Mask" and c.registered_face_policy == "redact_all" and c.temporal_smoothing == 5
    assert str(window.settings.value("registered_face_policy")) == "redact_all"
    p.face_policy.setCurrentIndex(p.face_policy.findData("exempt")); p.method.setCurrentText("Blur"); p.smooth.setValue(3); p.save()


def test_live_page_lifecycle_with_mock_camera(qapp, window, monkeypatch):
    from Backend.pipeline.live_session import LiveSession
    monkeypatch.setattr(LiveSession, "_default_capture", staticmethod(lambda i: FakeCap()))
    monkeypatch.setattr(window.service, "discover_cameras", lambda: [{"index": 0, "label": "Mock camera"}])
    window.navigate("Live Camera")
    page = window.pages["Live Camera"]
    assert pump(qapp, until=lambda: page.camera_combo.currentData() == 0, timeout=5)
    for cycle in range(3):                                   # repeated start / stop
        page.start_btn.click()
        assert pump(qapp, until=lambda: page.session is not None, timeout=5)
        assert pump(qapp, until=lambda: page.session.stats["frames"] > 8, timeout=10)
        pump(qapp, 0.3)
        assert page.state_pill.text() == "Live" and page.t_fps.value_label.text() != "—"
        if cycle == 0:
            shot(window, "11_live")
        page.stop_btn.click()
        assert page.session is None and window.service.live_session is None
        assert page.start_btn.isEnabled()
    assert window.service.last_live_recording and Path(window.service.last_live_recording).exists()


def test_live_camera_error_is_shown(qapp, window, monkeypatch):
    from Backend.pipeline.live_session import LiveSession
    monkeypatch.setattr(LiveSession, "_default_capture", staticmethod(lambda i: FakeCap(opened=False)))
    page = window.pages["Live Camera"]
    window.navigate("Live Camera")
    page.start_btn.setEnabled(True)
    page.start_btn.click()
    assert pump(qapp, until=lambda: page.banner.isVisible(), timeout=5)
    assert "Could not open the camera" in page.banner.text.text() and page.start_btn.isEnabled()


def test_window_resizes_without_horizontal_scroll(qapp, window):
    from PyQt5.QtCore import Qt
    for w, h in ((1100, 700), (1366, 768), (1920, 1040)):
        window.resize(w, h)
        for name in window.pages:
            window.navigate(name); pump(qapp, 0.03)
            bar = window.pages[name].scroll.horizontalScrollBar()
            if bar.maximum() != 0:
                col = window.pages[name].scroll.column
                wide = sorted(((c.minimumSizeHint().width(), type(c).__name__, c.objectName(), c.text()[:30] if hasattr(c, "text") and callable(c.text) else "")
                               for c in col.findChildren(QWidget)), reverse=True)[:6]
                raise AssertionError(f"{name} scrolls horizontally at {w}px by {bar.maximum()}: {wide}")
    window.resize(1360, 820)
    window.navigate("Dashboard")
    shot(window, "12_dashboard_final")
