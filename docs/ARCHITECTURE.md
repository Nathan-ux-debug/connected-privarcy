# PrivARcy — Architecture (as implemented)

```
Frontend (PyQt5)            Backend
  main.py                     service.py  BackendService  ← the only thing the UI calls
  ui/shell.py  sidebar/nav      ├─ pipeline/video_pipeline.py   VideoPipeline  (video I/O, cancel/resume)
  ui/pages/*   8 sections       ├─ pipeline/job_runner.py       JobRunner      (D1/D3 bookkeeping, manifests)
  ui/workers.py QThreads        ├─ pipeline/frame_engine.py     FrameEngine    ← per-frame decisions (video AND live)
                                ├─ pipeline/live_session.py     LiveSession    (threaded camera, measured FPS)
                                ├─ pipeline/text_stage.py       localise → OCR → PII → decision
                                ├─ ocr/{localizer,provider,engine}.py   text lines, lazy TrOCR
                                ├─ face_privacy/{detector,consent,embedder,matcher}.py
                                ├─ decision_engine/             HIGH/MEDIUM/LOW routing + PII rules
                                ├─ storage/manager.py           D1 / D3 lifecycle
                                ├─ data_store/                  D2 face registry, D4 correction log (SQLite)
                                └─ rule_refinement/             proposals · validation · versions · rollback
```

## Per-frame flow (`FrameEngine.process`)
1. YOLO predict at conf ≥ 0.25 (`low_floor`); class **names** from the model.
2. < 0.50 → not redacted, counted per class as an *uncertainty warning* (low confidence ≠ safe).
3. EMA smoothing + tracking (IoU ≥ 0.50, expiry after 5 unmatched detection runs).
4. Faces: consent check against D2 (policy `exempt` or `redact_all`); any failure ⇒ stays redacted.
5. `DecisionEngine`: ≥ 0.80 auto-redact · 0.50–0.80 review (de-duplicated per track, crop saved to D3) · below: ignore.
6. Regions of class `document` / `screen` / `id_card` / `credit_card` → text stage: locate lines → read **new** lines (cached per line) → PII rules → auto / review. Recognised text is never stored.
7. Held tracks: an auto-redacted object missed this frame stays redacted until its track expires.
8. Redactions applied **last**, so every read sees original pixels. Preview/recording receive only redacted frames.

## Stores
| Store | Location | Lifecycle |
|---|---|---|
| D1 Video File Store | `Backend/storage/d1_video_store/jobs/*.json` (+ `Backend/outputs/` for redacted videos) | Manifest per job; sources referenced, not copied (opt-in copy). |
| D2 Face Registry | `Backend/face_registry.db`, `Backend/uploads/faces/` | Unchanged location. Removing a person deletes embedding + photo. |
| D3 Processed Frames | `Backend/storage/d3_processed_frames/<job>/` | Review crops + `review_items.json`. Deleted when review is resolved, on job delete, or after 7 days. |
| D4 Correction Log | `Backend/correction_log.db`; rules in `Backend/storage/rules/` | Append-only decisions; rule versions immutable, audit in `audit.jsonl`. |

## Confidence policy
HIGH ≥ 0.80 auto · MEDIUM 0.50–0.80 review · LOW < 0.50 not redacted alone (reported). Per-class overrides exist only through validated, approved rule versions (min 0.55).

## Review & refinement — three different things
Logging (every decision) → Rule refinement (evidence-gated, validated, approved, reversible; lowers one class's auto threshold) → Retraining (**not implemented**).

## Launch
```powershell
cd "Connected PrivARcy"
.\Backend\venv\Scripts\python.exe Frontend\main.py
```
Tests: `python -m pytest Backend\tests Frontend\tests -q` (Frontend tests set `QT_QPA_PLATFORM=offscreen`).
