# face_privacy

"Blur everyone except the people I've registered." A two-stage system:
**detect** every face in a frame, then **recognize** which ones (if any)
match someone enrolled — anything that doesn't match gets handed to
`redaction.FrameRedactor` to blur.

```
frame → FaceDetector (find faces) → FaceEmbedder (per face) → FaceMatcher
     (compare against data_store.FaceRegistryDB) → unknown faces → redaction.FrameRedactor
```

---

## 1. Layout

```
face_privacy/
├── __init__.py       # public API
├── exceptions.py       # FacePrivacyError
├── models.py             # FaceBox, MatchResult
├── detector.py             # FaceDetector — Haar cascade face detection
├── embedder.py                # FaceEmbedder — HOG-based feature vector per face
├── matcher.py                    # FaceMatcher — cosine similarity vs. enrolled registry
├── enrollment.py                    # enroll_person() — one-call detect+embed+store
├── cli.py                              # `enroll` and `run` (video/live) subcommands
└── tests/
    └── test_face_privacy.py
```

## 2. How it works — verified on a real photo, not just synthetic data

Unlike most of this pipeline, face detection genuinely needs real face
statistics to test meaningfully — a procedurally-drawn "cartoon face"
won't reliably trigger a Haar cascade. So instead of a synthetic test
image, this was verified against a real photo (via `skimage.data.astronaut()`,
a real NASA portrait bundled with scikit-image):

- **Detection**: found exactly one face, tight bounding box around it, confidence 0.58.
- **Recognition — same face**: enrolled that face, matched it again → similarity **1.00**.
- **Recognition — same face, shifted crop** (simulating a slightly different
  frame of the same person): similarity **0.92** — still well above the
  0.75 match threshold.
- **Recognition — unrelated image region**: similarity **0.63** — correctly
  stays below threshold, correctly flagged as "unknown."

That's the actual mechanic your vlogger use case depends on: enroll once,
recognize across different frames/angles of the same person, reject
everyone else.

### 2.1 Detection (`detector.py`)

Uses OpenCV's bundled Haar cascade (`haarcascade_frontalface_default.xml`
— ships with `opencv-python`, no download). Fast, CPU-friendly, good
enough for frontal/near-frontal faces. Weaker on extreme angles, low
light, or partial occlusion than a modern deep-learning face detector.

### 2.2 Embedding (`embedder.py`)

Each detected face crop is resized to 96×96, contrast-normalized
(histogram equalization — reduces sensitivity to lighting differences),
and converted to a HOG (Histogram of Oriented Gradients) feature vector,
then L2-normalized. This is a classical computer-vision technique, not a
deep neural embedding — see §4 below for what that trade-off means.

### 2.3 Matching (`matcher.py`)

Cosine similarity between a query embedding and every embedding in the
registry (a person can have multiple enrolled photos/exemplars — the
closest one wins). Above `similarity_threshold` (default 0.75) → known,
leave alone. Below → unknown, redact.

### 2.4 Enrollment (`enrollment.py`, `cli.py enroll`)

```bash
python -m face_privacy.cli enroll --name "Alice" --photo alice.jpg --registry-db faces.db
```

Detects the (largest, if multiple) face in the photo, embeds it, and
stores it via `data_store.FaceRegistryDB` (D2) — the same persistent
store the correction-log system uses, so enrollments survive restarts.

### 2.5 Running it (`cli.py run`)

```bash
# Batch: redact a saved video
python -m face_privacy.cli run --video uploads/clip.mp4 --output redacted.mp4 --registry-db faces.db

# Live: redact a webcam feed in real time
python -m face_privacy.cli run --live 0 --registry-db faces.db
```

Every unmatched face in every frame gets redacted via `redaction.FrameRedactor`
(same blur/pixelate/solid options as the rest of the pipeline). Live mode
reuses `redaction.ThreadedVideoCapture` for the same low-latency capture
behavior described in the `redaction` package's README.

---

## 3. Combining with object redaction

`face_privacy` and `redaction`'s class-based redaction (credit cards,
documents) are complementary, not competing — run both passes on the
same frame if you need both:

```python
from redaction import FrameRedactor, RedactionConfig

# Pass 1: blur unenrolled faces (face_privacy's own flow)
frame = _process_frame(frame, detector, embedder, matcher, enrolled, face_redactor)

# Pass 2: also blur credit cards / documents (yolo_detection's classes)
object_result = yolo_detector.predict(frame)
frame = FrameRedactor(RedactionConfig(classes_to_redact={"credit_card", "document"})).redact(
    frame, object_result.detections
)
```

---

## 4. Accuracy expectations and the upgrade path

Be direct about what this baseline is and isn't:

- **What it's good for**: a working end-to-end demonstration of the
  "enroll → recognize → redact everyone else" architecture, verified
  correct on real photo data (§2 above), no heavy install, no GPU needed.
- **Where it's weaker than production face recognition**: HOG embeddings
  are noticeably more sensitive to pose, lighting, and expression changes
  than a deep-learning face embedding (FaceNet, ArcFace, etc.). For a
  vlogger scenario — real people, moving, varied lighting, different
  angles across a video — expect more false "unknown" results (an
  enrolled person occasionally getting blurred because a frame didn't
  match well enough) than a production system would give you. Enrolling
  **multiple exemplar photos per person** (different angles/lighting) is
  the built-in mitigation — `FaceRegistryDB` already supports this — and
  is worth doing before relying on this for real footage.
- **Upgrade path, if accuracy here isn't sufficient**: swap `FaceEmbedder`
  for a deep embedding model (e.g. `insightface`/ArcFace, or the `face_recognition`
  library built on `dlib`). Both are meaningfully heavier installs
  (dlib compilation or ONNX runtime + model weights) — the same kind of
  disk/download tradeoff already flagged for `ultralytics` and would need
  the same real-environment install and validation this sandbox couldn't do.

---

## 5. A note on storing children's biometric data

Since the stated use case includes recognizing specific children: face
embeddings are biometric identifiers. Several jurisdictions regulate
biometric data specifically and more strictly when it belongs to minors
(e.g. Illinois' BIPA, GDPR's provisions on children's data, COPPA in the
U.S. for anything collecting data from children). This is a legitimate,
privacy-protective use case (blurring bystanders by default is the
opposite of surveillance), but worth a real legal/compliance check —
consent requirements, retention limits, deletion rights — before storing
real children's embeddings in `D2`, separate from the engineering.

---

## 6. Install

```bash
pip install "opencv-python>=4.8,<5" numpy
# data_store, redaction, frame_extraction packages for the full CLI flows
```

No extra download beyond `opencv-python` itself — the Haar cascade and
HOG descriptor are both built in.

## 7. Tests

```bash
python -m unittest face_privacy.tests.test_face_privacy -v
```

21 tests covering: detector error handling and config validation,
embedder consistency (same input → same output, normalized, grayscale
input accepted), matcher correctness (identical embedding matches,
orthogonal embedding doesn't, closest-of-multiple-people picked
correctly, shape-mismatch guarded), and enrollment (stores correctly,
no-face-found raises, multiple-faces picks the largest). Detection and
matching were additionally verified against a real photo — see §2.
