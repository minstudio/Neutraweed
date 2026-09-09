#!/usr/bin/env python3
"""Blur a synthetic pool until its instance sharpness matches real imagery.

Sharpness = var(Laplacian) / var(intensity) inside label boxes, the measure in
scripts/sharpness_spread.py. Every synthetic pool sits 2-6x above real on it, and
the ordering predicts detection across v2 / v9 / v8. This script removes the gap
so the claim can be tested causally instead of by rank correlation.

Per image, bisects a Gaussian sigma so that the median instance sharpness in that
image lands on the target. The whole frame is blurred, not just the boxes, so no
discontinuity is introduced at box edges. Labels are symlinked unchanged.

    python scripts/blur_to_target.py --pool data/synthetic/sd35cut_v8 \
        --ref data/real --out data/synthetic/sd35cut_v8_blur
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg"}


def _layout(root: Path) -> tuple[Path, Path]:
    for img, lab in (("images", "labels"), ("images/train", "labels/train"), (".", "labels")):
        i, l = root / img, root / lab
        if i.is_dir() and l.is_dir():
            return i, l
    raise SystemExit(f"[blur] no images/labels layout under {root}")


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


def _boxes(lab_path: Path, W: int, H: int, min_side: int) -> list[tuple[int, int, int, int]]:
    out = []
    for line in lab_path.read_text().split("\n"):
        f = line.split()
        if len(f) < 5:
            continue
        xc, yc, bw, bh = (float(v) for v in f[1:5])
        x1 = int(max(0, (xc - bw / 2) * W))
        y1 = int(max(0, (yc - bh / 2) * H))
        x2 = int(min(W, (xc + bw / 2) * W))
        y2 = int(min(H, (yc + bh / 2) * H))
        if min(x2 - x1, y2 - y1) >= min_side:
            out.append((x1, y1, x2, y2))
    return out


def _sharp(gray: np.ndarray, boxes) -> float:
    vals = []
    for x1, y1, x2, y2 in boxes:
        c = gray[y1:y2, x1:x2]
        iv = float(c.var())
        if iv < 1.0:
            continue
        vals.append(float(cv2.Laplacian(c, cv2.CV_32F).var()) / iv)
    return float(np.median(vals)) if vals else float("nan")


def _measure(args):
    img_path, lab_path, min_side = args
    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        return float("nan")
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return _sharp(g, _boxes(lab_path, g.shape[1], g.shape[0], min_side))


def _process(args):
    img_path, lab_path, out_img, out_lab, target, min_side, max_sigma, iters = args
    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        return None
    H, W = im.shape[:2]
    boxes = _boxes(lab_path, W, H, min_side)

    sigma = 0.0
    if boxes:
        g0 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if _sharp(g0, boxes) > target:
            lo, hi = 0.0, max_sigma
            for _ in range(iters):
                mid = (lo + hi) / 2
                b = cv2.GaussianBlur(im, (0, 0), mid)
                s = _sharp(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32), boxes)
                if not np.isfinite(s) or s > target:
                    lo = mid
                else:
                    hi = mid
            sigma = (lo + hi) / 2

    out = cv2.GaussianBlur(im, (0, 0), sigma) if sigma > 1e-3 else im
    cv2.imwrite(str(out_img), out)
    if out_lab.is_symlink() or out_lab.exists():
        out_lab.unlink()
    out_lab.symlink_to(lab_path.resolve())
    return sigma


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=None,
                    help="real dataset root; its median instance sharpness becomes "
                         "the target. Overridden by --target.")
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--ref-images", type=int, default=400)
    ap.add_argument("--min-side", type=int, default=16)
    ap.add_argument("--max-sigma", type=float, default=4.0)
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    target = args.target
    if target is None:
        if args.ref is None:
            raise SystemExit("[blur] give --ref or --target")
        ref = _pairs(args.ref)[: args.ref_images]
        with mp.Pool(args.workers) as pool:
            vals = pool.map(_measure, [(i, l, args.min_side) for i, l in ref])
        vals = [v for v in vals if np.isfinite(v)]
        target = float(np.median(vals))
        print(f"[blur] target from {len(vals)} ref images: {target:.4f}")

    pairs = _pairs(args.pool)
    if not pairs:
        raise SystemExit(f"[blur] no image/label pairs under {args.pool}")
    img_out, lab_out = args.out / "images", args.out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    jobs = [(i, l, img_out / i.name, lab_out / f"{i.stem}.txt",
             target, args.min_side, args.max_sigma, args.iters) for i, l in pairs]
    with mp.Pool(args.workers) as pool:
        sig = [s for s in pool.map(_process, jobs) if s is not None]

    sig = np.asarray(sig)
    print(f"[blur] {len(sig)} images -> {args.out}")
    print(f"[blur] sigma  p10 {np.percentile(sig, 10):.2f}  "
          f"p50 {np.median(sig):.2f}  p90 {np.percentile(sig, 90):.2f}  "
          f"max {sig.max():.2f}")
    n_cap = int((sig > args.max_sigma - 0.05).sum())
    if n_cap:
        print(f"[blur] WARNING {n_cap} images hit the sigma cap — raise --max-sigma")


if __name__ == "__main__":
    main()
