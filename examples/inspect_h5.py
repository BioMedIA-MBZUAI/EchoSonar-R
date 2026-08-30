#!/usr/bin/env python
"""Print the structure of a clip/study embeddings file.

    python examples/inspect_h5.py examples/sample_outputs/study_embeddings.h5
"""

import argparse
import json

import h5py
import numpy as np


def describe(path, limit):
    with h5py.File(path, "r") as h5:
        print(f"{path}")
        if h5.attrs:
            root = ", ".join(f"{k}={v}" for k, v in h5.attrs.items())
            print(f"  root attrs: {root}")
        keys = list(h5.keys())
        print(f"  {len(keys)} top-level entries\n")

        for key in keys[:limit]:
            group = h5[key]
            print(f"  /{key}")
            for name, dataset in group.items():
                values = dataset[:]
                print(
                    f"    {name:<14} {str(values.shape):<12} {values.dtype}  "
                    f"min={np.min(values):+.4f} max={np.max(values):+.4f} "
                    f"norm={np.linalg.norm(values):.4f}"
                )
            for name, value in group.attrs.items():
                shown = value
                if isinstance(value, str) and value.startswith("["):
                    shown = json.loads(value)
                print(f"    @{name:<13} {shown}")
            print()

        if len(keys) > limit:
            print(f"  ... {len(keys) - limit} more (use --limit)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--limit", type=int, default=3, help="entries to print")
    args = p.parse_args()
    describe(args.path, args.limit)


if __name__ == "__main__":
    main()
