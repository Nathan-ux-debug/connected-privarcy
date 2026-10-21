# PrivARcy — Feature status (audited against the source, not previous reports)

Legend: **Verified** = an automated test or measurement ran against the real code. **Mock-tested** = logic verified with a
stand-in (labelled) for the YOLO model / camera / TrOCR. **Unverified** = code exists but has not been run on real input.

| Area | Status | Evidence / what is still needed |
|---|---|---|
| YOLO class mapping | **Verified (metadata)** | `best.pt` has **6 classes**: 0 id_card, 1 document, 2 screen, 3 face, 4 credit_card, 5 license_plate. This differs from the 5-class order in the brief. The pipeline resolves classes **by name**, never by index; `document` is an OCR region only (never redacted whole). Retrain/re-export is NOT needed. |
| YOLO inference on real footage | **Unverified here** | Could not run torch in the audit sandbox. Run `Backend\tools\baseline_check.py <video>` on your machine (prints frames in/out, detections, redactions). |
| Recorded-video pipeline | **Mock-tested** | 30 frames in → 30 out, manifest, cancel→resume to full length, failure cleanup. Real-model regression (your previous 215→215 test) must be re-run with `baseline_check.py`. |
| Confidence gating | **Verified** | Boundary tests: 0.80→auto, 0.7999→review, 0.50→review, 0.4999→not redacted and counted as an uncertainty warning. |
| Temporal smoothing / tracks | **Verified** | EMA α=2/(n+1); IoU ≥ 0.50 matches, < 0.50 new track; track expires after 5 unmatched runs; **new:** an auto-redacted track is kept redacted until it expires; redaction covers union(raw, smoothed). |
| Review confirm / reject / log | **Mock-tested end to end** | Persisted queue survives restart; confirm rewrites the output video; decisions logged with class, bbox, confidence. |
| Correction logging (old data) | **Defect found** | All 74 existing rows have `confidence=0.0`, `bbox=[0,0,0,0]` — the old UI never passed them. They are ignored as evidence. New rows are complete. |
| Face detection (YOLO + Haar) | **Mock-tested** | App now starts without `cv2.CascadeClassifier` (tested by removing it). YOLO face boxes drive both redaction and consent matching. Moving/blurred-face recall must be evaluated on real footage. |
| Face registration / matching | **Partial — risk found** | HOG embeddings barely separate faces: unrelated random textures score ≈0.94 cosine; a different natural image 0.56–0.72. The library default 0.75 would mis-identify strangers. Default raised to **0.90**; failures keep the face redacted. **Needs a deep embedding model + evaluation with real people.** |
| OCR text localisation | **Implemented; localiser verified on synthetic text** | OpenCV line localiser finds lines in original-frame coordinates (crop offsets tested). Weak on blur, tiny or low-contrast text. |
| OCR recognition (TrOCR) | **Unverified** | `transformers` is **not installed** in your venv, and the model downloads on first use. Pipeline/decision/redaction logic is tested with a **mock** recogniser. OCR/YOLO confidence calibration is NOT done (same 0.80/0.50 thresholds applied). |
| OCR → PII → redaction integration | **Mock-tested** | Sensitive lines redacted, harmless line untouched, whole document not blanked, recognised text never stored (only pattern names). |
| Decision engine | **Connected** | One `FrameEngine` used by video and live. Note: `DecisionEngine`'s own default `t_high` is 0.85; the pipeline passes 0.80 explicitly. |
| Live camera | **Mock-tested; physical webcam UNVERIFIED** | Lifecycle, repeated start/stop, disconnect, open failure, fail-closed on processing error, redacted-only preview — all with a **mock camera**. Run `Backend\tools\webcam_check.py`; the UI also records measured FPS/latency of real sessions. |
| Signature detection | **Out of scope (decision)** | Not in the model's classes or the 5-class brief; no reliable localiser. Shown as unavailable. |
| Storage D1–D4 | **Verified** | D1 manifests (reference sources, no duplication), D3 per-job temp frames freed after review / 7 days, interrupted-job marking, legacy `outputs/review_frames` cleanup. D2/D4 stay where they were (no data moved). |
| Correction → rule refinement | **Implemented; no usable evidence yet** | Proposal → validation → approval → versioned activation → audit → rollback, tested. Only lowers a per-class auto-redact threshold, never below 0.55. Not model retraining. |
| OpenCV startup error | **Not reproducible from the ZIP** | The shipped venv has exactly one OpenCV (`opencv-python 4.12.0.88`), which provides `CascadeClassifier`; no `cv2` shadowing file. So the error came from a different interpreter/OpenCV install. Run `Backend\tools\diagnose_opencv.py` with the failing interpreter. The app no longer depends on it. |
| Frontend | **Rebuilt; UI-tested offscreen** | 12 integration tests (all sections, full video job, review, faces, corrections, settings, live with mock camera, resize without horizontal scroll). Not yet viewed on your real display. |

## Remaining work
1. Run `baseline_check.py` with the real model and your sample videos; record frames in/out.
2. `pip install transformers` in the venv, then evaluate OCR on real documents/screens; calibrate OCR confidence.
3. Run `webcam_check.py` on the Acer Nitro webcam; keep the report.
4. Replace/augment HOG with a deep face embedding (e.g. ONNX SFace) and evaluate on registered vs unregistered people.
5. Collect fresh review decisions (new logging works) before expecting rule proposals.
6. Build a held-out evaluation set (not training data) for detection recall, false positives and redaction coverage.
