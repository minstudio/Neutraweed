"""Matched-size control (CLAUDE.md §4 Stage D, §8 — non-negotiable).

For every hybrid set of N_total training images, build a real-only set of the
SAME size by duplicating + classically augmenting the real train images. If the
hybrid set beats this control, the gain comes from synthetic *diversity*, not
merely from having more images.

We don't pre-render augmentations to disk: we repeat real image paths to hit the
target count and rely on the detector's train-time augmentation (Ultralytics
applies mosaic/flip/HSV per epoch, so repeats are augmented differently). This
keeps the control honest and disk-cheap.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import yaml

from ..common import classes, paths
from .build_hybrid import _real_split_images, _write_list


def build(name: str, target_count: int, seed: int) -> Path:
    real_train = _real_split_images("train")
    if not real_train:
        raise SystemExit("No real train images — run Stage A first.")

    rng = random.Random(seed)
    reps = math.ceil(target_count / len(real_train))
    pool = (real_train * reps)
    rng.shuffle(pool)
    train = pool[:target_count]

    out_dir = paths.DATASETS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_list(out_dir / "train.txt", train)
    _write_list(out_dir / "val.txt", _real_split_images("val"))
    _write_list(out_dir / "test.txt", _real_split_images("test"))

    descriptor = {
        "path": str(out_dir.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {i: n for i, n in enumerate(classes.CLASS_NAMES)},
        "meta": {
            "composition": "matched-size-control",
            "target_count": target_count,
            "n_real_unique": len(real_train),
            "duplication_factor": round(target_count / len(real_train), 3),
            "seed": seed,
        },
    }
    with open(out_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(descriptor, f, sort_keys=False)

    print(f"Matched-size control '{name}': {target_count} imgs "
          f"({len(real_train)} unique x{descriptor['meta']['duplication_factor']}) -> {out_dir}")
    return out_dir / "dataset.yaml"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Matched-size real-only control (Stage D).")
    ap.add_argument("--name", required=True)
    ap.add_argument("--target-count", type=int, required=True,
                    help="match this to the paired hybrid set's train size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.name, args.target_count, args.seed)
