#!/usr/bin/env python
"""Generate ``clip_tokens.h5``: pre-pooling EchoPrime tokens for report generation.

Same inputs as ``extract_echoprime_embeddings.py`` -- a study directory, a
directory of studies, or a JSONL manifest -- and the same 16 frames per clip.

    python scripts/extract_echoprime_tokens.py \
        --study my_study/ \
        --weights-dir /path/to/EchoPrime/model_data/weights \
        --out-dir outputs/
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echosonar.encoders.echoprime import set_worker_device  # noqa: E402
from echosonar.pipeline import extract_clip_tokens  # noqa: E402
from echosonar.preprocess import FRAME_SAMPLING_STRATEGIES  # noqa: E402
from echosonar.study import discover_studies, load_manifest  # noqa: E402

logger = logging.getLogger("echosonar")

DEFAULT_VIEW_CLASSIFIER = "checkpoints/view_classifier.pt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract EchoPrime spatiotemporal tokens for report generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--study", help="one study directory of echo clips")
    source.add_argument("--root", help="directory of study directories")
    source.add_argument("--manifest", help="JSONL manifest of studies")
    p.add_argument(
        "--multi",
        action="store_true",
        help="with --root: treat each sub-directory as a separate study",
    )

    p.add_argument(
        "--weights-dir",
        required=True,
        help="directory holding echo_prime_encoder.pt and view_classifier.pt",
    )
    p.add_argument(
        "--view-classifier",
        default=DEFAULT_VIEW_CLASSIFIER,
        help="fine-grained view classifier checkpoint (matches the report "
        "model's training vocabulary; distinct from EchoPrime's own view "
        "classifier). Pass '' to skip and only store EchoPrime's coarse view.",
    )
    p.add_argument("--out-dir", default="outputs", help="where to write clip_tokens.h5")
    p.add_argument(
        "--split-name",
        help="suffix for the output file, e.g. 'train' -> clip_tokens_train.h5",
    )
    p.add_argument(
        "--frame-sampling",
        default="first32",
        choices=FRAME_SAMPLING_STRATEGIES,
        help="which 16 frames to feed the encoder (must match extract_detections.py "
        "and extract_echoprime_embeddings.py for a consistent dataset)",
    )

    p.add_argument("--device", help="e.g. cuda:0 or cpu (default: cuda if available)")
    p.add_argument("--gpu", type=int, help="shorthand for --device cuda:GPU")
    p.add_argument("--batch-size", type=int, default=16, help="clips per forward pass")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite the output file instead of resuming into it",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def collect_studies(args):
    if args.manifest:
        return load_manifest(args.manifest)
    if args.study:
        return discover_studies(args.study, multi=False)
    return discover_studies(args.root, multi=args.multi)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    device = args.device
    if args.gpu is not None:
        device = str(set_worker_device(args.gpu))

    view_classifier_checkpoint = args.view_classifier or None
    if view_classifier_checkpoint and not Path(view_classifier_checkpoint).is_file():
        raise FileNotFoundError(
            f"view classifier checkpoint not found: {view_classifier_checkpoint}\n"
            "The report model was trained on this classifier's view labels, not "
            "EchoPrime's own coarser ones, so it's required by default -- get it "
            "the same way as the RT-DETR checkpoint (see README.md), or pass "
            "--view-classifier '' to store only EchoPrime's coarse view instead."
        )

    studies = collect_studies(args)
    n_clips = sum(len(s.clips) for s in studies)
    logger.info("%d studies, %d clips", len(studies), n_clips)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.split_name}" if args.split_name else ""
    tokens_path = out_dir / f"clip_tokens{suffix}.h5"

    summary = extract_clip_tokens(
        studies,
        weights_dir=args.weights_dir,
        tokens_h5_path=tokens_path,
        view_classifier_checkpoint=view_classifier_checkpoint,
        device=device,
        frame_sampling=args.frame_sampling,
        batch_size=args.batch_size,
        resume=not args.overwrite,
    )
    logger.info("%s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
