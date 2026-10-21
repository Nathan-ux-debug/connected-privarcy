"""Command-line entry point for face_privacy.

Examples:
    # Enroll someone from a photo
    python -m face_privacy.cli enroll --name "Alice" --photo alice.jpg --registry-db faces.db

    # Blur every unenrolled face in a saved video
    python -m face_privacy.cli run --video uploads/clip.mp4 --output redacted.mp4 \\
        --registry-db faces.db --method pixelate

    # Blur every unenrolled face live from a webcam
    python -m face_privacy.cli run --live 0 --registry-db faces.db
"""

import argparse
import logging
import sys

import cv2

from .detector import FaceDetector
from .embedder import FaceEmbedder
from .matcher import FaceMatcher
from .enrollment import enroll_person
from .exceptions import FacePrivacyError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _cmd_enroll(args) -> int:
    from data_store import FaceRegistryDB, DataStoreError

    photo = cv2.imread(args.photo)
    if photo is None:
        logger.error("Could not read image: %s", args.photo)
        return 1

    registry = FaceRegistryDB(args.registry_db)
    try:
        row_id = enroll_person(args.name, photo, registry, source=args.photo)
    except (FacePrivacyError, DataStoreError) as exc:
        logger.error("Enrollment failed: %s", exc)
        return 1
    finally:
        registry.close()

    logger.info("Enrolled '%s' (row id=%d) into %s", args.name, row_id, args.registry_db)
    return 0


def _load_enrolled(registry_db: str):
    from data_store import FaceRegistryDB
    import numpy as np

    registry = FaceRegistryDB(registry_db)
    try:
        records = registry.all_embeddings()
    finally:
        registry.close()
    return [(r.person_name, np.array(r.embedding)) for r in records]


def _process_frame(frame, detector, embedder, matcher, enrolled, redactor):
    faces = detector.detect(frame)
    unknown_faces = []
    for face in faces:
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        embedding = embedder.embed(crop)
        result = matcher.match(face, embedding, enrolled)
        if not result.is_known:
            unknown_faces.append(face)
    return redactor.redact(frame, unknown_faces)


def _cmd_run(args) -> int:
    from redaction import FrameRedactor, RedactionConfig, VideoWriter

    detector = FaceDetector()
    embedder = FaceEmbedder()
    matcher = FaceMatcher(similarity_threshold=args.similarity_threshold)
    redactor = FrameRedactor(RedactionConfig(classes_to_redact=None, method=args.method))
    enrolled = _load_enrolled(args.registry_db)
    logger.info("Loaded %d enrolled embedding(s) for %d known people.",
                len(enrolled), len({name for name, _ in enrolled}))

    if args.video:
        from frame_extraction import FrameExtractor
        extractor = FrameExtractor(video_path=args.video, output_dir=None, target_fps=args.fps or 1_000_000)
        output_fps = extractor.source_fps / extractor.frame_interval
        writer = VideoWriter(args.output, fps=output_fps)
        count = 0
        try:
            for frame_meta in extractor.extract():
                redacted = _process_frame(frame_meta.frame, detector, embedder, matcher, enrolled, redactor)
                writer.write_frame(redacted)
                count += 1
        finally:
            writer.release()
        logger.info("Done. %d frame(s) written to %s", count, args.output)
        return 0

    else:
        from redaction.live_stream import ThreadedVideoCapture
        live_source = int(args.live) if args.live.isdigit() else args.live
        capture = ThreadedVideoCapture(live_source).start()
        writer = VideoWriter(args.output, fps=30) if args.output else None
        logger.info("Live mode started. Press 'q' in the preview window to stop.")
        try:
            while True:
                frame = capture.read()
                if frame is None:
                    continue
                redacted = _process_frame(frame, detector, embedder, matcher, enrolled, redactor)
                if writer:
                    writer.write_frame(redacted)
                if not args.no_display:
                    cv2.imshow("face_privacy (press 'q' to quit)", redacted)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            capture.stop()
            if writer:
                writer.release()
            if not args.no_display:
                cv2.destroyAllWindows()
        return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enroll known faces, and blur every unenrolled face in a video/feed.")
    sub = parser.add_subparsers(dest="command", required=True)

    enroll_p = sub.add_parser("enroll", help="Register a person's face from a photo.")
    enroll_p.add_argument("--name", type=str, required=True)
    enroll_p.add_argument("--photo", type=str, required=True)
    enroll_p.add_argument("--registry-db", type=str, default="face_registry.db")

    run_p = sub.add_parser("run", help="Blur every unenrolled face in a video or live feed.")
    source = run_p.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=str)
    source.add_argument("--live", type=str)
    run_p.add_argument("--output", type=str, default=None)
    run_p.add_argument("--registry-db", type=str, default="face_registry.db")
    run_p.add_argument("--method", type=str, default="gaussian", choices=["gaussian", "pixelate", "solid"])
    run_p.add_argument("--similarity-threshold", type=float, default=0.75)
    run_p.add_argument("--fps", type=float, default=None, help="[--video only] Sampling fps; omit for every frame.")
    run_p.add_argument("--no-display", action="store_true", help="[--live only] Don't open a preview window.")

    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.command == "enroll":
        return _cmd_enroll(args)
    if args.video and not args.output:
        logger.error("--output is required when using --video.")
        return 1
    try:
        return _cmd_run(args)
    except FacePrivacyError as exc:
        logger.error("Failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
