"""Derive a per-frame valid region from the horizontal span of the annotations.

Emits the same JSON as scripts/draw_valid_region.py, so it feeds
map_regions_to_tiles.py and ap_analysis.py --valid-regions unchanged.

WHAT THIS IS. Two vertical lines per frame, at the leftmost and rightmost
ground-truth box edge (plus an optional margin). Everything between them is
valid; detections outside are ignored.

WHAT IT IS NOT. An unbiased measurement. The region is defined from where the
boxes are, so a detection outside the span is discounted *because* nothing was
annotated there — which is the quantity under test. It also under-estimates the
true annotated region whenever the annotators covered strip area that happens to
contain no weeds, so it over-corrects. Treat the result as an UPPER BOUND on the
scope-corrected AP, and report it beside the uncorrected number as a lower bound.

The margin controls how much of that bias you accept: --margin 0 is the tightest
and most optimistic, a large margin approaches the uncorrected number.

  python scripts/auto_region_from_gt.py --split test --out regions_auto.json --margin 200
  python scripts/map_regions_to_tiles.py --in regions_auto.json --out regions_auto_tiles.json \
      --images data/real/images/test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import paths  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"}


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("regions_auto.json"))
    ap.add_argument("--margin", type=float, default=0.0,
                    help="pixels added outside the box span on each side. Larger = "
                         "less optimistic. Try 0, 200, 500 to bracket.")
    ap.add_argument("--axis", default="x", choices=["x", "y", "both"],
                    help="which extent to bound. Crop rows run top-to-bottom here, "
                         "so 'x' is the default.")
    args = ap.parse_args()

    img_dir = args.images or (paths.REAL / "images" / args.split)
    lbl_dir = args.labels
    if lbl_dir is None:
        guess = img_dir.parent.parent / "labels" / img_dir.name
        lbl_dir = guess if guess.exists() else (paths.REAL / "labels" / args.split)
    if not img_dir.exists():
        raise SystemExit(f"no such directory: {img_dir}")

    regions, stats = {}, []
    for ip in sorted(p for p in img_dir.iterdir() if p.suffix in IMG_EXT):
        lp = lbl_dir / f"{ip.stem}.txt"
        with Image.open(ip) as im:
            W, H = im.size
        xs, ys, n = [], [], 0
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                q = line.split()
                if len(q) != 5:
                    continue
                _, cx, cy, bw, bh = int(q[0]), *(float(v) for v in q[1:])
                xs += [(cx - bw / 2) * W, (cx + bw / 2) * W]
                ys += [(cy - bh / 2) * H, (cy + bh / 2) * H]
                n += 1
        if n == 0:
            regions[ip.name] = {"mode": "none"}
            stats.append((ip.name, 0, 0.0))
            continue

        x0 = max(0.0, min(xs) - args.margin) if args.axis in ("x", "both") else 0.0
        x1 = min(float(W), max(xs) + args.margin) if args.axis in ("x", "both") else float(W)
        y0 = max(0.0, min(ys) - args.margin) if args.axis in ("y", "both") else 0.0
        y1 = min(float(H), max(ys) + args.margin) if args.axis in ("y", "both") else float(H)

        frac = ((x1 - x0) * (y1 - y0)) / float(W * H)
        if frac >= 0.995:
            regions[ip.name] = {"mode": "all"}
        else:
            regions[ip.name] = {
                "mode": "poly",
                "polys": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]],
                "lines": [[[x0, y0], [x0, y1]], [[x1, y0], [x1, y1]]],
                "flip": False,
            }
        stats.append((ip.name, n, frac))

    args.out.write_text(json.dumps(
        {"regions": regions,
         "meta": {"tool": "auto_region_from_gt", "margin": args.margin,
                  "axis": args.axis, "split": args.split,
                  "warning": "region derived from annotation extent — upper bound, "
                             "not an unbiased measurement"}},
        indent=1), encoding="utf-8")

    cov = [f for _, n, f in stats if n]
    print(f"[auto] {len(stats)} frames, margin={args.margin:.0f}px, axis={args.axis}")
    if cov:
        cov_sorted = sorted(cov)
        print(f"[auto] valid area as a fraction of the frame: "
              f"min {cov_sorted[0]:.2f}  median {cov_sorted[len(cov)//2]:.2f}  "
              f"max {cov_sorted[-1]:.2f}")
    empty = sum(1 for _, n, _ in stats if n == 0)
    if empty:
        print(f"[auto] {empty} frames had no annotations -> marked 'none'")
    print(f"[auto] -> {args.out}")
    print("[auto] REMINDER: this is an upper bound. Report it next to the "
          "uncorrected number, and say how it was derived.")


if __name__ == "__main__":
    main()
