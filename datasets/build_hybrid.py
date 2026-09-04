"""Build a real / synthetic / hybrid YOLO dataset (CLAUDE.md §4 Stage D).

A hybrid set = all real train images + a sampled fraction of a generator's
synthetic pool. The fraction (`synthetic_ratio`) is expressed relative to the
real-train image count, swept 0.1 -> 2.0 in 0.1 steps for the ratio curve.

Outputs data/datasets/<name>/{train.txt,val.txt,test.txt,dataset.yaml}.
val/test always reference the real split (enforced below).

Run (example):
  python -m src.datasets.build_hybrid --name hybrid_sd35_r50 \
      --composition hybrid --generator sd35 --ratio 0.5 --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from ..common import classes, paths

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_images(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def _real_split_images(split: str) -> list[Path]:
    return _list_images(paths.REAL / "images" / split)


def _synthetic_pool(generator: str) -> list[Path]:
    pool = paths.SYNTHETIC / generator / "images"
    # only keep images that actually have a label file (post-QC, post-annotate)
    return [p for p in _list_images(pool) if _label_for(p).exists()]


def _label_for(image_path: Path) -> Path:
    """Mirror Ultralytics' convention: swap the last 'images' path part for 'labels'."""
    parts = list(image_path.with_suffix(".txt").parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts)


def _sample(pool: list[Path], k: int, seed: int,
            strict_unique: bool = False) -> list[Path]:
    if k <= 0 or not pool:
        return []
    rng = random.Random(seed)
    if k <= len(pool):
        return rng.sample(pool, k)
    if strict_unique:
        raise SystemExit(
            f"--strict-unique: this ratio needs {k} synthetic images but the pool "
            f"holds only {len(pool)}. The {k - len(pool)} extra would be duplicates, "
            f"i.e. upweighting the same images rather than adding information.\n"
            f"  Either generate a pool of at least {k} scenes, or lower the ratio.")
    # ratio > pool size: take all, then sample the remainder with replacement
    extra = [rng.choice(pool) for _ in range(k - len(pool))]
    return pool + extra


def _crowding(label: Path, iou_thr: float = 0.10) -> float:
    """Fraction of a tile's boxes that overlap at least one other box.

    Same definition ap_analysis uses, so a tile excluded from training is
    excluded from scoring by the same rule. Counting boxes measures busy-ness;
    this measures clustering, which is what makes a plant ambiguous to delimit
    and is outside the scope of detecting semi-isolated plants.
    """
    if not label.exists():
        return 0.0
    b = []
    for line in label.read_text(encoding="utf-8").splitlines():
        q = line.split()
        if len(q) == 5:
            _, cx, cy, w, h = (float(v) for v in q)
            b.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    if len(b) < 2:
        return 0.0
    import numpy as np
    a = np.asarray(b, dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], a[None, :, 0])
    y1 = np.maximum(a[:, None, 1], a[None, :, 1])
    x2 = np.minimum(a[:, None, 2], a[None, :, 2])
    y2 = np.minimum(a[:, None, 3], a[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    iou = inter / np.maximum(ar[:, None] + ar[None, :] - inter, 1e-9)
    np.fill_diagonal(iou, 0.0)
    return float((iou.max(axis=1) >= iou_thr).mean())


def _drop_crowded(images: list[Path], max_overlap: float,
                  tag: str = "real") -> list[Path]:
    """Remove tiles where more than `max_overlap` of the plants touch a neighbour.

    Applied to TRAINING data. The evaluation-side equivalent lives in
    ap_analysis (--drop-overlap); filtering only one side would train on dense
    clusters and then decline to score them, which is not the same experiment.
    """
    if max_overlap >= 1.0:
        return images
    import json
    cache_f = paths.DATASETS / f".crowding_{tag}.json"
    cache = {}
    if cache_f.exists():
        try:
            cache = json.loads(cache_f.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    keep, dirty = [], False
    for p in images:
        key = str(p)
        if key not in cache:
            lp = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
            cache[key] = _crowding(lp)
            dirty = True
        if cache[key] <= max_overlap:
            keep.append(p)
    if dirty:
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache_f.write_text(json.dumps(cache), encoding="utf-8")
        except Exception:
            pass
    print(f"  crowding filter ({tag}, max_overlap={max_overlap}): "
          f"kept {len(keep)} / {len(images)} "
          f"({len(keep) / max(len(images), 1) * 100:.0f}%)")
    return keep


def build(
    name: str,
    composition: str,
    generator: str | None,
    ratio: float,
    seed: int,
    max_overlap: float = 1.0,
    strict_unique: bool = False,
) -> Path:
    real_train = _real_split_images("train")
    if max_overlap < 1.0:
        real_train = _drop_crowded(real_train, max_overlap, "real_train")
    if not real_train and composition != "synthetic-only":
        raise SystemExit("No real train images — run Stage A (prepare_data) first.")

    if composition == "real-only":
        train = real_train
    elif composition == "synthetic-only":
        if not generator:
            raise SystemExit("synthetic-only needs --generator")
        train = _synthetic_pool(generator)
        if not train:
            raise SystemExit(
                f"Synthetic pool for '{generator}' is empty or unlabelled — "
                "run Stage B/C first."
            )
    elif composition == "hybrid":
        if not generator:
            raise SystemExit("hybrid needs --generator")
        pool = _synthetic_pool(generator)
        if not pool:
            raise SystemExit(f"Synthetic pool for '{generator}' empty — run Stage B/C.")
        if max_overlap < 1.0:
            pool = _drop_crowded(pool, max_overlap, f"synth_{generator}")
        n_synth = round(ratio * len(real_train))
        train = real_train + _sample(pool, n_synth, seed, strict_unique)
    else:
        raise SystemExit(f"Unknown composition: {composition}")

    out_dir = paths.DATASETS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_list(out_dir / "train.txt", train)
    # INVARIANT: val/test are always the real split.
    _write_list(out_dir / "val.txt", _real_split_images("val"))
    _write_list(out_dir / "test.txt", _real_split_images("test"))

    descriptor = {
        "path": str(out_dir.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {i: n for i, n in enumerate(classes.CLASS_NAMES)},
        # provenance, so a run is reproducible from its dataset.yaml alone
        "meta": {
            "composition": composition,
            "generator": generator,
            "synthetic_ratio": ratio,
            "seed": seed,
            "n_train": len(train),
            "n_real_train": len(real_train),
            "n_synthetic": len(train) - len(real_train) if composition == "hybrid" else (
                len(train) if composition == "synthetic-only" else 0
            ),
        },
    }
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(descriptor, f, sort_keys=False)

    print(f"Dataset '{name}' -> {yaml_path}")
    print(f"  train={len(train)} (real {len(real_train)} + synth {descriptor['meta']['n_synthetic']})"
          f"  val={len(_real_split_images('val'))}  test={len(_real_split_images('test'))}")
    return yaml_path


def _write_list(path: Path, images: list[Path]) -> None:
    path.write_text("\n".join(str(p.resolve()) for p in images) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Assemble a YOLO dataset (Stage D).")
    ap.add_argument("--name", required=True)
    ap.add_argument("--composition", required=True,
                    choices=["real-only", "synthetic-only", "hybrid"])
    ap.add_argument("--generator", default=None,
                    help="synthetic pool name under data/synthetic/ (any: composite, sd35cut, ...)")
    ap.add_argument("--ratio", type=float, default=0.0)
    ap.add_argument("--max-overlap", type=float, default=1.0,
                    help="drop TRAINING tiles where more than this fraction of "
                         "the plants overlap another at IoU>=0.10. Applies to both "
                         "the real and the synthetic half. 1.0 = off. Match the "
                         "value used with ap_analysis --drop-overlap.")
    ap.add_argument("--strict-unique", action="store_true",
                    help="refuse to build if the ratio would require reusing "
                         "synthetic images. Beyond pool_size/13110 the extra "
                         "images are duplicates, i.e. upweighting rather than "
                         "new data.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.name, args.composition, args.generator, args.ratio, args.seed,
          args.max_overlap, args.strict_unique)
