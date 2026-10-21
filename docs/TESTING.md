# Test methodology and results

Run date of this audit: sandbox, Python 3.12, OpenCV 4.13, PyQt5 5.15 (offscreen). **No torch/YOLO weights executed; no webcam; no TrOCR.**

| Suite | Result | Notes |
|---|---|---|
| `Backend/tests` (47) | 47 passed | FrameEngine, OCR localisation, storage, rules, jobs, live. **MOCK**: YOLO (scripted), TrOCR (scripted), camera (synthetic). |
| `Frontend/tests` (12) | 12 passed | Real widgets + real BackendService; mock YOLO/camera. Screenshots in `Frontend/docs/screenshots/`. |
| Original suites (decision_engine, face_privacy, ocr, redaction, data_store, yolo_detection, …) | 142 passed, 18 failed | The same 18 fail on the **untouched original** code here (no `fastapi`, no CUDA libs in the sandbox): environment, not regressions. Re-run on your machine. |

## What the mock tests do and do not prove
They prove the **logic** (thresholds, tracking, coordinates, storage, decisions, lifecycle, UI wiring). They do **not** prove detection
quality, OCR accuracy, face-matching accuracy, webcam behaviour or real FPS. Those need `baseline_check.py`, `webcam_check.py`
and a held-out evaluation set (see STATUS.md).
