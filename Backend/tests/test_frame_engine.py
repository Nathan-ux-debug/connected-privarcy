import numpy as np
import pytest

from Backend.pipeline.config import PipelineConfig, verify_class_map, normalize_method, model_class_names
from Backend.pipeline.frame_engine import FrameEngine
from .fakes import FakeYolo, REAL_NAMES, gradient_frame


def cfg(**kw):
    c = PipelineConfig(redaction_method="Solid Mask", box_padding={}, cover_raw_box=False)
    c.update(**kw)
    return c


def run(script, config=None, frames=None, consent=None, n=None):
    model = FakeYolo(script)
    eng = FrameEngine(model, config or cfg(), source_name="t.mp4", consent=consent)
    outs = []
    for i in range(n or len(script)):
        f = gradient_frame() if frames is None else frames[i]
        outs.append(eng.process(f.copy(), i + 1, 30.0))
    return eng, outs


def masked(frame, box):
    x1, y1, x2, y2 = box
    return bool((frame[y1:y2, x1:x2] == 0).all())


# ---------------- class mapping ------------------------------------------------
def test_class_map_of_shipped_weights_is_six_class_and_reported():
    """Names taken from best.pt metadata (read from the file, see audit)."""
    rep = verify_class_map(REAL_NAMES)
    assert rep["ok"] and rep["extra"] == ["document"] and rep["order_differs"]
    assert any("document" in w for w in rep["warnings"])


def test_class_names_come_from_model_not_index():
    assert model_class_names(FakeYolo([]))[3] == "face"


def test_redaction_method_normalisation():
    assert normalize_method("pixelate") == "Pixelate"
    assert normalize_method("solid mask") == "Solid Mask"
    assert normalize_method("blur") == "Blur"
    assert normalize_method(None) == "Blur"


# ---------------- confidence gating (boundaries) -----------------------------------
@pytest.mark.parametrize("conf,status", [(0.80, "auto_redacted"), (0.7999, "pending"), (0.50, "pending")])
def test_confidence_boundaries(conf, status):
    box = (50, 50, 120, 120)
    eng, outs = run([[("credit_card", conf, box)]])
    assert outs[0].records[0]["status"] == status
    assert masked(outs[0].frame, box) == (status == "auto_redacted")


def test_below_review_floor_is_not_redacted_but_counted_as_uncertain():
    box = (50, 50, 120, 120)
    eng, outs = run([[("face", 0.4999, box)]])
    assert outs[0].records == [] and not masked(outs[0].frame, box)
    assert outs[0].low_confidence == {"face": 1}
    assert eng.summary()["low_confidence_unreviewed"] == {"face": 1}


def test_document_is_never_redacted_whole():
    box = (20, 20, 200, 200)
    eng, outs = run([[("document", 0.95, box)]])
    assert outs[0].redactions == 0 and outs[0].records == []


def test_disabled_class_ignored():
    box = (50, 50, 120, 120)
    eng, outs = run([[("license_plate", 0.95, box)]], cfg(enabled_classes={"face"}))
    assert outs[0].redactions == 0


def test_per_class_rule_override_lowers_auto_threshold_but_never_below_floor():
    c = cfg(class_auto_redact_confidence={"credit_card": 0.6, "face": 0.2})
    assert c.auto_threshold_for("credit_card") == 0.6
    assert c.auto_threshold_for("face") == pytest.approx(0.55)      # clamped above the review floor
    box = (50, 50, 120, 120)
    eng, outs = run([[("credit_card", 0.65, box)]], c)
    assert outs[0].records[0]["status"] == "auto_redacted"


# ---------------- temporal smoothing & track lifecycle -------------------------------
def test_ema_smoothing_uses_alpha_two_over_n_plus_one():
    c = cfg(temporal_smoothing=3)      # alpha = 0.5
    a, b = (100, 100, 140, 140), (110, 100, 150, 140)     # IoU = 30*40/(2*1600-1200)=0.6
    eng, outs = run([[("credit_card", 0.9, a)], [("credit_card", 0.9, b)]], c)
    assert outs[1].records[0]["bbox"] == [105, 100, 145, 140]


def test_iou_below_half_starts_a_new_track():
    a, b = (100, 100, 140, 140), (130, 100, 170, 140)     # IoU = 10*40/(3200-400)=0.14
    eng, outs = run([[("credit_card", 0.9, a)], [("credit_card", 0.9, b)]])
    assert outs[1].records[0]["bbox"] == list(b)           # not smoothed
    assert len(eng.tracks) == 2


def test_track_expires_after_five_unmatched_frames_and_is_held_until_then():
    box = (100, 100, 140, 140)
    script = [[("credit_card", 0.9, box)]] + [[]] * 6
    eng, outs = run(script)
    held = [o.held for o in outs]
    assert held == [0, 1, 1, 1, 1, 1, 0]                   # frames 2..6 held, frame 7 expired
    for o in outs[:6]:
        assert masked(o.frame, box)
    assert not masked(outs[6].frame, box)
    assert eng.tracks == []


def test_hold_can_be_disabled():
    box = (100, 100, 140, 140)
    eng, outs = run([[("credit_card", 0.9, box)], []], cfg(hold_missed_detections=False))
    assert not masked(outs[1].frame, box)


def test_redaction_covers_raw_box_even_when_ema_lags():
    c = cfg(cover_raw_box=True, temporal_smoothing=7)       # strong lag
    a, b = (100, 100, 140, 140), (112, 100, 152, 140)
    eng, outs = run([[("face", 0.9, a)], [("face", 0.9, b)]], c)
    assert masked(outs[1].frame, b)


# ---------------- review queue ---------------------------------------------------
def test_medium_confidence_tracked_and_deduplicated(tmp_path):
    a, b = (100, 100, 140, 140), (104, 100, 144, 140)
    model = FakeYolo([[("id_card", 0.6, a)], [("id_card", 0.7, b)], [("id_card", 0.65, a)]])
    eng = FrameEngine(model, cfg(), source_name="v.mp4", review_dir=tmp_path)
    outs = [eng.process(gradient_frame(), i + 1, 30.0) for i in range(3)]
    statuses = [o.records[0]["status"] for o in outs]
    assert statuses == ["pending", "tracked_duplicate", "tracked_duplicate"]
    pending = eng.records[0]
    assert len(pending["track_observations"]) == 3 and pending["confidence"] == 0.7
    assert eng.total_flagged == 1
    assert len(list(tmp_path.glob("*.jpg"))) == 1            # only the best crop kept
    assert not masked(outs[0].frame, a)                       # not redacted until a human confirms


# ---------------- faces / consent policy ------------------------------------------
class AlwaysKnown:
    def known_indices(self, frame, boxes):
        return set(range(len(boxes)))


def test_registered_face_exempt_policy_default():
    box = (100, 100, 140, 140)
    eng, outs = run([[("face", 0.95, box)]], consent=AlwaysKnown())
    assert outs[0].records[0]["status"] == "consented" and not masked(outs[0].frame, box)


def test_registered_face_redact_all_policy():
    box = (100, 100, 140, 140)
    eng, outs = run([[("face", 0.95, box)]], cfg(registered_face_policy="redact_all"), consent=AlwaysKnown())
    assert outs[0].records[0]["status"] == "auto_redacted" and masked(outs[0].frame, box)


def test_consent_matcher_fails_closed_on_error():
    from Backend.face_privacy.consent import FaceConsentMatcher

    class BadRegistry:
        def count(self): raise RuntimeError("db locked")
    m = FaceConsentMatcher(BadRegistry(), object(), object())
    assert m.known_indices(gradient_frame(), [(0, 0, 50, 50)]) == set()
    assert "db locked" in m.last_error


def test_consent_matcher_matches_and_caches():
    skimage_data = pytest.importorskip("skimage.data")
    import cv2
    from Backend.data_store import FaceRegistryDB
    from Backend.face_privacy import FaceEmbedder, FaceMatcher
    from Backend.face_privacy.consent import FaceConsentMatcher
    frame = cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR)
    emb = FaceEmbedder()
    reg = FaceRegistryDB(":memory:")
    reg.enroll("alice", emb.embed(frame[30:140, 150:260]).tolist())
    m = FaceConsentMatcher(reg, emb, FaceMatcher(similarity_threshold=0.90))
    assert m.known_indices(frame, [(150, 30, 260, 140)]) == {0}
    assert m.known_indices(frame, [(300, 250, 410, 360)]) == set()   # different region: not consented
    assert m.known_indices(frame, [(150, 30, 154, 34)]) == set()     # too small to identify -> stays redacted


def test_hog_default_threshold_is_not_discriminative_documented():
    """Regression guard for the audit finding: unrelated textures exceed the library default 0.75."""
    from Backend.face_privacy import FaceEmbedder
    from Backend.face_privacy.matcher import FaceMatcher
    rs = np.random.RandomState(1)
    e = FaceEmbedder()
    a, b = (e.embed((rs.rand(100, 100, 3) * 255).astype(np.uint8)) for _ in range(2))
    assert FaceMatcher._cosine_similarity(a, b) > 0.75
