#!/usr/bin/env python3
"""Build a mixed synthetic pool by symlinking a 50/50 draw from two existing pools.

    python scripts/mix_pools.py --a <poolA> --b <poolB> --out <newpool> --n 2500
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


IMG_EXT = {".png", ".jpg", ".jpeg"}


def _layout(pool: Path) -> tuple[Path, Path]:
    for img, lab in (("images", "labels"), (".", "labels")):
        i, l = pool / img, pool / lab
        if i.is_dir() and l.is_dir():
            return i.resolve(), l.resolve()
    raise SystemExit(f"[mix] no images/labels layout under {pool}")


def _pairs(pool: Path) -> list[tuple[Path, Path]]:
    img_dir, lab_dir = _layout(pool)
    out = []
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        lab = lab_dir / p.relative_to(img_dir).with_suffix(".txt")
        if lab.exists():
            out.append((p.resolve(), lab.resolve()))
    if not out:
        raise SystemExit(f"[mix] no image/label pairs under {pool}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0,
                    help="total images; 0 = size of the smaller parent")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pa, pb = _pairs(args.a), _pairs(args.b)
    n = args.n or min(len(pa), len(pb))
    half = n // 2
    if half > len(pa) or n - half > len(pb):
        raise SystemExit(f"[mix] need {half}/{n - half}, have {len(pa)}/{len(pb)}")

    rng = random.Random(args.seed)
    draw = [("a", p) for p in rng.sample(pa, half)] + \
           [("b", p) for p in rng.sample(pb, n - half)]

    img_out, lab_out = args.out / "images", args.out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    for tag, (img, lab) in draw:
        stem = f"{tag}_{img.stem}"
        for src, dst in ((img, img_out / f"{stem}{img.suffix}"),
                         (lab, lab_out / f"{stem}.txt")):
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)

    print(f"[mix] {args.out}: {half} from {args.a.name}, {n - half} from {args.b.name}")
    print(f"[mix] parents held {len(pa)} and {len(pb)} pairs")


if __name__ == "__main__":
    main()
