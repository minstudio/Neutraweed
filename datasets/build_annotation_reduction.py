"""Annotation-reduction experiment datasets (CLAUDE.md 2 RQ5, 4 Stage D).

Question: can composite synthetic data substitute for expensive real-2022
annotation? Gomez 2025 showed the 2021+2022 mix beats 2021-only; nobody asked
whether *synthetic* 2022-style data recovers that benefit at zero labeling cost.

Four train arms, all evaluated on the frozen real val/test (Parcela B / Parcela
C, 2022) identical to every other experiment (no leakage):

  A  2021-only            real TOMATO_1 tiles only             (lower bound)
  B  2021 + real Finca    the full field-holdout train         (oracle / upper bound)
  T  2021 + composite     TOMATO_1 + composite, sized to == B  (synthetic substitute)
  M  matched-size control 2021 duplicated up to |B|            (quantity-only baseline)

Readout: recovery = (T - A) / (B - A) in map50_95. T > M proves the gain is the
2022 domain information the composites carry, not merely image count (CLAUDE.md
8: always run the matched-size control).

Year is read off the tiled filename prefix: TOMATO_1__* = 2021, TOMATO_2__* =
2022. In the train split every TOMATO_2 image is Finca Santa Amalia (Parcela
B -> val, Parcela C -> test), so the prefix cleanly separates 2021 from real-2022.

Run:
  python -m src.datasets.build_annotation_reduction --arm all --seed 0
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import yaml

from ..common import classes, paths
from .build_hybrid import _real_split_images, _sample, _synthetic_pool, _write_list

YEAR_PREFIX = {2021: "TOMATO_1__", 2022: "TOMATO_2__"}


def _split_by_year(images: list[Path]) -> tuple[list[Path], list[Path]]:
    y2021 = [p for p in images if p.name.startswith(YEAR_PREFIX[2021])]
    y2022 = [p for p in images if p.name.startswith(YEAR_PREFIX[2022])]
    covered = set(y2021) | set(y2022)
    other = [p for p in images if p not in covered]
    if other:
        raise SystemExit(
            f"{len(other)} train images match no known year prefix "
            f"(e.g. {other[0].name}); expected TOMATO_1__* or TOMATO_2__*."
        )
    return y2021, y2022


def build(arm: str, seed: int = 0, generator: str = "composite") -> Path:
    arm = arm.upper()
    real_train = _real_split_images("train")
    if not real_train:
        raise SystemExit(
            "No real train images run Stage A (prepare_data) then tile_dataset first."
        )
    y2021, y2022 = _split_by_year(real_train)
    n_B = len(y2021) + len(y2022)

    n_synth = 0
    if arm == "A":
        train = list(y2021)
    elif arm == "B":
        train = y2021 + y2022
    elif arm == "T":
        pool = _synthetic_pool(generator)
        if not pool:
            raise SystemExit(
                f"Composite pool '{generator}' empty or unlabelled run Stage C first."
            )
        n_synth = len(y2022)                       # match T's addition to B exactly
        train = y2021 + _sample(pool, n_synth, seed)
    elif arm == "M":
        rng = random.Random(seed)
        reps = math.ceil(n_B / len(y2021))
        pool = y2021 * reps
        rng.shuffle(pool)
        train = pool[:n_B]
    else:
        raise SystemExit(f"Unknown arm {arm!r}; expected A|B|T|M")

    # Arm T is the only generator-dependent arm; suffix non-default pools so a
    # 2022-background rerun (annotred_T_composite2022) never clobbers the original
    # 2021-domain annotred_T and both can be compared side by side.
    suffix = f"_{generator}" if (arm == "T" and generator != "composite") else ""
    name = f"annotred_{arm}{suffix}"
    out_dir = paths.DATASETS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_list(out_dir / "train.txt", train)
    # INVARIANT: val/test are always the real field-holdout split (2022 Parcela B/C).
    _write_list(out_dir / "val.txt", _real_split_images("val"))
    _write_list(out_dir / "test.txt", _real_split_images("test"))

    composition = {
        "A": "real-2021-only",
        "B": "real-2021+2022",
        "T": "real-2021+composite",
        "M": "matched-size-2021",
    }[arm]
    descriptor = {
        "path": str(out_dir.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {i: n for i, n in enumerate(classes.CLASS_NAMES)},
        "meta": {
            "experiment": "annotation_reduction",
            "arm": arm,
            "composition": composition,
            "generator": generator if arm == "T" else None,
            "seed": seed,
            "n_train": len(train),
            "n_real_2021": len(y2021),
            "n_real_2022": len(y2022),
            "n_synthetic": n_synth,
            "target_size_B": n_B,
        },
    }
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(descriptor, f, sort_keys=False)

    m = descriptor["meta"]
    print(
        f"[{name}] train={m['n_train']} "
        f"(2021 {m['n_real_2021']} + real-2022 {m['n_real_2022']} + synth {m['n_synthetic']}) "
        f"| val={len(_real_split_images('val'))} test={len(_real_split_images('test'))} "
        f"-> {yaml_path}"
    )
    return yaml_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build annotation-reduction datasets (arms A/B/T/M)."
    )
    ap.add_argument("--arm", default="all", help="A | B | T | M | all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--generator", default="composite",
        help="synthetic pool name under data/synthetic/ for arm T "
             "(e.g. composite, composite2022, composite2022strict, sd35, flux, gan)",
    )
    args = ap.parse_args()
    arms = ["A", "B", "T", "M"] if args.arm.lower() == "all" else [args.arm]
    for a in arms:
        build(a, args.seed, args.generator)
