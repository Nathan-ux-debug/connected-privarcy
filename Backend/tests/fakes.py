"""Test doubles. Anything fake is labelled MOCK in the test names/docstrings."""

import numpy as np

REAL_NAMES = {0: "id_card", 1: "document", 2: "screen", 3: "face", 4: "credit_card", 5: "license_plate"}


class _T:
    def __init__(self, v): self.v = v
    def item(self): return self.v
    def cpu(self): return self
    def numpy(self): return np.asarray(self.v, dtype=float)
    def __getitem__(self, i): return self


class _Box:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [_T(cls_id)]
        self.conf = [_T(conf)]
        self.xyxy = [_T(list(xyxy))]


class _Result:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeYolo:
    """MOCK YOLO: returns scripted detections per predict() call.

    script: list (one entry per call) of lists of (class_name, conf, (x1,y1,x2,y2)).
    Calls past the end of the script return no detections.
    """

    def __init__(self, script, names=None):
        self.names = dict(names or REAL_NAMES)
        self._ids = {v: k for k, v in self.names.items()}
        self.script = script
        self.calls = 0

    def predict(self, source=None, conf=0.25, imgsz=640, verbose=False):
        items = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        boxes = [_Box(self._ids[n], c, b) for (n, c, b) in items if c >= conf]
        return [_Result(boxes, self.names)]


def gradient_frame(w=320, h=240):
    """Non-uniform frame so blur / mask is detectable."""
    x = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    y = np.tile(np.linspace(255, 0, h, dtype=np.uint8).reshape(-1, 1), (1, w))
    noise = (np.random.RandomState(0).rand(h, w) * 60).astype(np.uint8)
    return np.dstack([x, y, noise]).astype(np.uint8)
