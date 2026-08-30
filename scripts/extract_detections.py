#!/usr/bin/env python
"""Generate ``detections.h5`` from echo studies with the RT-DETR detector.

Same inputs as ``extract_echoprime_embeddings.py`` -- a study directory, a
directory of studies -- and the same 16 frames per clip, so
the two output files line up clip for clip.

    python scripts/extract_detections.py \
        --study my_study/ \
        --checkpoint checkpoints/rtdetr_cardiac_7cls.pt \
        --out-dir outputs/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echosonar.encoders.detector import CardiacDetector  # noqa: E402
from echosonar.preprocess import FRAME_SAMPLING_STRATEGIES, sample_frame_indices  # noqa: E402
from echosonar.study import discover_studies, load_manifest  # noqa: E402
from echosonar.writer import existing_keys  # noqa: E402

logger = logging.getLogger("echosonar")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract RT-DETR boxes and structure embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--study", help="one study directory of echo clips")
    source.add_argument("--root", help="directory of study directories")
    source.add_argument("--manifest", help="JSONL manifest of studies")
    p.add_argument("--multi", action="store_true", help="with --root: one study per sub-directory")

    p.add_argument(
        "--checkpoint",
        default="checkpoints/rtdetr_cardiac_7cls.pt",
        help="RT-DETR cardiac detector weights",
    )
    p.add_argument("--out-dir", default="outputs", help="where to write detections.h5")
    p.add_argument("--split-name", help="suffix, e.g. 'train' -> detections_train.h5")

    p.add_argument("--frame-sampling", default="first32", choices=FRAME_SAMPLING_STRATEGIES)
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--gpu", type=int, default=0, help="GPU index")
    p.add_argument("--overwrite", action="store_true", help="rewrite instead of resuming")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def collect_studies(args):
    if args.manifest:
        return load_manifest(args.manifest)
    if args.study:
        return discover_studies(args.study, multi=False)
    return discover_studies(args.root, multi=args.multi)


def write_clip(h5, study_id, clip, detections):
    """Store one clip's per-frame detections."""
    group = h5.create_group(clip.clip_id)
    group.attrs["study_uuid"] = study_id
    group.attrs["n_frames_processed"] = len(detections)
    if clip.view:
        group.attrs["view_name"] = clip.view

    for frame_idx, frame in enumerate(detections):
        frame_group = group.create_group(f"frame_{frame_idx}")
        frame_group.create_dataset("boxes_xyxy", data=frame.boxes_xyxy)
        frame_group.create_dataset("classes", data=frame.classes)
        frame_group.create_dataset("confidences", data=frame.confidences)
        frame_group.create_dataset("embeddings", data=frame.embeddings)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    studies = collect_studies(args)
    logger.info("%d studies, %d clips", len(studies), sum(len(s.clips) for s in studies))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.split_name}" if args.split_name else ""
    out_path = out_dir / f"detections{suffix}.h5"

    done = set() if args.overwrite else set(existing_keys(out_path))
    if done:
        logger.info("resuming: %d clips already done", len(done))

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731

    failed = []
    with CardiacDetector(
        args.checkpoint, device=args.gpu, conf=args.conf, imgsz=args.imgsz
    ) as detector, h5py.File(out_path, "w" if args.overwrite else "a") as h5:
        pending = [(s, c) for s in studies for c in s.clips if c.clip_id not in done]
        for i, (study, clip) in enumerate(tqdm(pending, desc="clips", unit="clip"), 1):
            try:
                frames = clip.read()
                indices = sample_frame_indices(len(frames), strategy=args.frame_sampling)
                write_clip(h5, study.study_id, clip, detector.detect_clip(frames[indices]))
            except Exception as exc:
                logger.error("clip %s failed: %s", clip.clip_id, exc)
                failed.append(clip.clip_id)
            if i % 100 == 0:
                h5.flush()
        n_clips = len(h5.keys())

    print(json.dumps({"output": str(out_path), "clips": n_clips, "failed": failed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
