#!/usr/bin/env python
"""Generate ``clip_embeddings.h5`` and ``study_embeddings.h5`` from echo studies.

Examples
--------
One study directory of videos::

    python scripts/extract_echoprime_embeddings.py \
        --study examples/demo_study \
        --weights-dir /path/to/EchoPrime/model_data/weights \
        --out-dir outputs/

A dataset of many studies (one sub-directory each), on 4 GPUs::

    python scripts/extract_echoprime_embeddings.py \
        --root /data/echo_studies --multi --num-gpus 4 \
        --weights-dir /path/to/EchoPrime/model_data/weights \
        --out-dir outputs/ --split-name train
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echosonar.encoders.echoprime import set_worker_device  # noqa: E402
from echosonar.pipeline import extract_studies  # noqa: E402
from echosonar.preprocess import FRAME_SAMPLING_STRATEGIES  # noqa: E402
from echosonar.study import Study, discover_studies, load_manifest  # noqa: E402
from echosonar.writer import existing_keys  # noqa: E402

logger = logging.getLogger("echosonar")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract EchoPrime clip and study embeddings",
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
        help="with --root: treat each sub-directory as a separate study "
        "(default: --root is itself one study)",
    )

    p.add_argument(
        "--weights-dir",
        required=True,
        help="directory holding echo_prime_encoder.pt and view_classifier.pt",
    )
    p.add_argument(
        "--mil-weights",
        help="path to MIL_weights.csv (default: found near --weights-dir, "
        "else assets/MIL_weights.csv if you dropped a local copy in)",
    )

    p.add_argument("--out-dir", default="outputs", help="where to write the H5 files")
    p.add_argument(
        "--split-name",
        help="suffix for the output files, e.g. 'train' -> "
        "clip_embeddings_train.h5 (default: no suffix)",
    )

    p.add_argument(
        "--frame-sampling",
        default="first32",
        choices=FRAME_SAMPLING_STRATEGIES,
        help="which 16 frames to feed the encoder (default: first32, the rule "
        "used for the released embeddings)",
    )
    p.add_argument("--device", help="e.g. cuda:0 or cpu (default: cuda if available)")
    p.add_argument("--num-gpus", type=int, default=1, help="shard studies over N GPUs")
    p.add_argument("--batch-size", type=int, default=16, help="clips per forward pass")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite the output files instead of resuming into them",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def collect_studies(args):
    if args.manifest:
        return load_manifest(args.manifest)
    if args.study:
        return discover_studies(args.study, multi=False)
    return discover_studies(args.root, multi=args.multi)


def output_paths(out_dir: Path, split_name, shard=None):
    suffix = f"_{split_name}" if split_name else ""
    shard_suffix = f"_gpu{shard}" if shard is not None else ""
    return (
        out_dir / f"clip_embeddings{suffix}{shard_suffix}.h5",
        out_dir / f"study_embeddings{suffix}{shard_suffix}.h5",
    )


def _worker(gpu_id, studies, args, out_dir):
    """One GPU's share of the studies, written to its own shard files."""
    set_worker_device(gpu_id)
    clip_path, study_path = output_paths(out_dir, args.split_name, shard=gpu_id)
    summary = extract_studies(
        studies,
        weights_dir=args.weights_dir,
        clip_h5_path=clip_path,
        study_h5_path=study_path,
        device=f"cuda:{gpu_id}",
        frame_sampling=args.frame_sampling,
        batch_size=args.batch_size,
        resume=not args.overwrite,
        progress=True,
    )
    logger.info("[gpu %d] %s", gpu_id, summary)


def merge_shards(out_dir: Path, split_name, num_gpus):
    """Fold the per-GPU shard files into the final pair, then delete them."""
    import h5py

    for prefix in ("clip_embeddings", "study_embeddings"):
        suffix = f"_{split_name}" if split_name else ""
        merged_path = out_dir / f"{prefix}{suffix}.h5"
        with h5py.File(merged_path, "a") as merged:
            for gpu_id in range(num_gpus):
                shard_path = out_dir / f"{prefix}{suffix}_gpu{gpu_id}.h5"
                if not shard_path.exists():
                    continue
                with h5py.File(shard_path, "r") as shard:
                    for key in shard.keys():
                        if key not in merged:
                            shard.copy(shard[key], merged, name=key)
                shard_path.unlink()
        logger.info("merged %s (%d entries)", merged_path, len(existing_keys(merged_path)))


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    studies = collect_studies(args)
    n_clips = sum(len(s.clips) for s in studies)
    logger.info("%d studies, %d clips", len(studies), n_clips)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = max(1, min(args.num_gpus, len(studies)))
    if num_gpus == 1:
        clip_path, study_path = output_paths(out_dir, args.split_name)
        summary = extract_studies(
            studies,
            weights_dir=args.weights_dir,
            clip_h5_path=clip_path,
            study_h5_path=study_path,
            device=args.device,
            frame_sampling=args.frame_sampling,
            batch_size=args.batch_size,
            resume=not args.overwrite,
        )
    else:
        import torch.multiprocessing as mp

        # Round-robin so each GPU gets a mix of large and small studies.
        chunks = [studies[i::num_gpus] for i in range(num_gpus)]
        mp.set_start_method("spawn", force=True)
        processes = []
        for gpu_id in range(num_gpus):
            proc = mp.Process(target=_worker, args=(gpu_id, chunks[gpu_id], args, out_dir))
            proc.start()
            processes.append(proc)
        for proc in processes:
            proc.join()

        merge_shards(out_dir, args.split_name, num_gpus)
        clip_path, study_path = output_paths(out_dir, args.split_name)
        summary = {
            "studies_written": len(existing_keys(study_path)),
            "clips_written": len(existing_keys(clip_path)),
            "failed": "see per-GPU logs above",
        }

    print(json.dumps({"outputs": [str(clip_path), str(study_path)], **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
