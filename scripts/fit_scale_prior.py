"""Fit the real box-size prior that the compositor should reproduce.

The detector sees every training image at `detector.imgsz`. A real tile of side
T px is letterboxed by imgsz/T, so a box of normalised side s appears at
s * T * imgsz / max(W, H) pixels. That apparent size — not the native size — is
the scale statistic a synthetic pool has to match.

Writes data/real/scale_prior.json: per-class quantiles of the apparent LONGEST
SIDE (in imgsz pixels) plus instance counts. CompositeGenerator samples from it.

Pure file reads + PIL headers, so it is safe on the login node.

  python scripts/fit_scale_prior.py
  python scripts/fit_scale_prior.py --split train --nq 101
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common import classes, paths  # noqa: E402


def _image_for(stem: str, img_dir: Path):
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".PNG"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def collect(split: str, imgsz: int, tile_size: int = 0):
    """Apparent longest/geometric-mean box side, in imgsz pixels.

    `tile_size > 0` means data/real is still un-tiled here (the local mirror):
    predict the post-tiling appearance by treating the source frame as if it
    were already cut into tile_size windows. On the cluster data/real IS tiled,
    so pass 0 and the frame's own side is used.
    """
    from PIL import Image

    img_dir = paths.REAL / "images" / split
    lbl_dir = paths.REAL / "labels" / split
    if not lbl_dir.exists():
        raise SystemExit(f"missing {lbl_dir}")

    per_long = {i: [] for i in range(len(classes.CLASS_NAMES))}
    per_geom = {i: [] for i in range(len(classes.CLASS_NAMES))}
    sides = []
    n_img = 0

    for lp in sorted(lbl_dir.glob("*.txt")):
        ip = _image_for(lp.stem, img_dir)
        if ip is None:
            continue
        with Image.open(ip) as im:
            W, H = im.size
        n_img += 1
        sides.append(max(W, H))
        k = imgsz / (tile_size or max(W, H))
        for line in lp.read_text(encoding="utf-8").splitlines():
            q = line.split()
            if len(q) != 5:
                continue
            cid = int(q[0])
            bw = float(q[3]) * W * k
            bh = float(q[4]) * H * k
            if bw <= 0 or bh <= 0:
                continue
            per_long[cid].append(max(bw, bh))
            per_geom[cid].append((bw * bh) ** 0.5)
    return per_long, per_geom, n_img, sides


def main() -> None:
    from src.common.config import load_config

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--nq", type=int, default=101)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config("base.yaml")
    imgsz = args.imgsz or int(cfg["detector"]["imgsz"])
    tile_size = int(cfg.get("tiling", {}).get("size", 0) or 0)

    probe_long, _, _, sides = collect(args.split, imgsz, 0)
    med_side = int(np.median(sides)) if sides else 0
    tiled = bool(tile_size) and med_side <= tile_size + 2
    correction = 0 if (tiled or not tile_size) else tile_size

    per_long, per_geom, n_img, sides = collect(args.split, imgsz, correction)
    total = sum(len(v) for v in per_long.values())
    if not total:
        raise SystemExit("no boxes found")
    del probe_long

    qs = np.linspace(0.0, 1.0, args.nq)
    out = {
        "split": args.split,
        "imgsz": imgsz,
        "tiling_size": tile_size,
        "source_looks_tiled": tiled,
        "tiling_correction_px": correction,
        "median_source_side_px": med_side,
        "n_images": n_img,
        "n_boxes": total,
        "quantile_levels": [round(float(q), 4) for q in qs],
        "classes": {},
    }

    print(f"[scale_prior] split={args.split} imgsz={imgsz} images={n_img} boxes={total}")
    print(f"[scale_prior] median source side={med_side}px "
          f"({'already tiled' if tiled else 'un-tiled'}; tiling.size={tile_size}"
          + (f"; applying tiling correction x{imgsz / correction:.3f}" if correction else "")
          + ")")
    print(f"{'class':8s}{'n':>7s}{'p10':>7s}{'p25':>7s}{'med':>7s}"
          f"{'p75':>7s}{'p90':>7s}{'p99':>7s}")

    for cid, name in enumerate(classes.CLASS_NAMES):
        v = np.asarray(per_long[cid], dtype=np.float64)
        g = np.asarray(per_geom[cid], dtype=np.float64)
        if v.size == 0:
            out["classes"][name] = {"n": 0, "longest_side_q": [], "geom_mean_q": []}
            continue
        out["classes"][name] = {
            "n": int(v.size),
            "share": round(float(v.size / total), 4),
            "longest_side_q": [round(float(x), 2) for x in np.quantile(v, qs)],
            "geom_mean_q": [round(float(x), 2) for x in np.quantile(g, qs)],
            "median": round(float(np.median(v)), 1),
        }
        p = np.percentile(v, [10, 25, 50, 75, 90, 99])
        print(f"{name:8s}{v.size:7d}" + "".join(f"{x:7.0f}" for x in p))

    allv = np.concatenate([np.asarray(per_long[c]) for c in per_long if per_long[c]])
    out["all"] = {
        "n": int(allv.size),
        "longest_side_q": [round(float(x), 2) for x in np.quantile(allv, qs)],
        "median": round(float(np.median(allv)), 1),
    }
    p = np.percentile(allv, [10, 25, 50, 75, 90, 99])
    print(f"{'ALL':8s}{allv.size:7d}" + "".join(f"{x:7.0f}" for x in p))

    dst = args.out or (paths.REAL / "scale_prior.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[scale_prior] -> {dst}")


if __name__ == "__main__":
    main()
