#!/usr/bin/env python3
"""Per-instance high-frequency content inside label boxes, and its SPREAD.

The claim under test is that v2 beats v8 because its instances span a wider range
of apparent sharpness, not because any one of them looks better. So the quantity
of interest is the p10-p90 span, not the median.

Metric: variance of the Laplacian inside the box, divided by the intensity
variance inside the same box. The ratio is exposure- and contrast-independent, so
a dark instance and a bright one are comparable.

    python scripts/sharpness_spread.py --src real=data/real --src v2=<pool> --src v8=<pool>

Each --src is NAME=PATH over a dir holding images/ and labels/ (or a split dir).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg"}
DEFAULT_NAMES = ["SOLNI", "POROL", "SETVE", "CYPRO", "ECHCG"]


def _layout(root: Path) -> tuple[Path, Path]:
    cands = [("images", "labels"), ("images/train", "labels/train"), (".", "labels")]
    for img, lab in cands:
        i, l = root / img, root / lab
        if i.is_dir() and l.is_dir():
            return i, l
    raise SystemExit(f"[sharp] no images/labels layout under {root}")


def _pairs(root: Path) -> list[tuple[Path, Path]]:
    img_dir, lab_dir = _layout(root)
    out = []
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        lab = lab_dir / p.relative_to(img_dir).with_suffix(".txt")
        if lab.exists():
            out.append((p, lab))
    return out


def _instances(img_path: Path, lab_path: Path, min_side: int):
    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        return
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    for line in lab_path.read_text().split("\n"):
        f = line.split()
        if len(f) < 5:
            continue
        c, xc, yc, bw, bh = int(f[0]), *(float(v) for v in f[1:5])
        x1 = int(max(0, (xc - bw / 2) * W))
        y1 = int(max(0, (yc - bh / 2) * H))
        x2 = int(min(W, (xc + bw / 2) * W))
        y2 = int(min(H, (yc + bh / 2) * H))
        if min(x2 - x1, y2 - y1) < min_side:
            continue
        crop = g[y1:y2, x1:x2]
        iv = float(crop.var())
        if iv < 1.0:
            continue
        lv = float(cv2.Laplacian(crop, cv2.CV_32F).var())
        yield c, lv / iv, float(np.sqrt((x2 - x1) * (y2 - y1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--min-side", type=int, default=16)
    ap.add_argument("--scale-band", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="only instances whose sqrt-area falls in [LO, HI) px, so "
                         "sources are compared at matched apparent size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for spec in args.src:
        if "=" not in spec:
            raise SystemExit(f"[sharp] --src must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        pairs = _pairs(Path(path))
        if not pairs:
            raise SystemExit(f"[sharp] no image/label pairs under {path}")
        if len(pairs) > args.max_images:
            pairs = rng.sample(pairs, args.max_images)

        acc: dict[int, list[float]] = {}
        for img, lab in pairs:
            for c, s, side in _instances(img, lab, args.min_side):
                if args.scale_band and not (args.scale_band[0] <= side < args.scale_band[1]):
                    continue
                acc.setdefault(c, []).append(s)

        print(f"\n=== {name}  ({len(pairs)} images)")
        print(f"{'class':<8}{'n':>7}{'p10':>9}{'p50':>9}{'p90':>9}{'p90/p10':>10}")
        allv = []
        for c in sorted(acc):
            v = np.asarray(acc[c])
            allv.append(v)
            p10, p50, p90 = np.percentile(v, [10, 50, 90])
            cn = args.names[c] if c < len(args.names) else str(c)
            print(f"{cn:<8}{len(v):>7}{p10:>9.4f}{p50:>9.4f}{p90:>9.4f}"
                  f"{p90 / max(p10, 1e-9):>10.2f}")
            rows.append((name, cn, len(v), p10, p50, p90, p90 / max(p10, 1e-9)))
        if allv:
            v = np.concatenate(allv)
            p10, p50, p90 = np.percentile(v, [10, 50, 90])
            print(f"{'ALL':<8}{len(v):>7}{p10:>9.4f}{p50:>9.4f}{p90:>9.4f}"
                  f"{p90 / max(p10, 1e-9):>10.2f}")
            rows.append((name, "ALL", len(v), p10, p50, p90, p90 / max(p10, 1e-9)))

    print("\nRead the p90/p10 column, not p50. Prediction: real wide, v2 wide, "
          "v8 narrow and shifted sharp.")

    if args.csv:
        import csv as _csv
        with args.csv.open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["source", "class", "n", "p10", "p50", "p90", "spread"])
            w.writerows(rows)
        print(f"-> {args.csv}")


if __name__ == "__main__":
    main()
