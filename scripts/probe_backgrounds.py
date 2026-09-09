"""Diagnose why extract_backgrounds finds no soil tiles for a source domain.

Fast, CPU, no sbatch. Samples a few real images from a source prefix and reports,
per tile size, how many random crops are (a) free of any weed box and (b) also
bare soil (mean Excess-Green <= threshold). Tells us whether the blocker is the
no-box constraint (dense field) or the greenness gate.

  python scripts/probe_backgrounds.py --sources TOMATO_2__ --n-images 15
"""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.common import paths
from src.generators.composite_gen import _boxes_px, _mean_exg, _matches_sources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--sources", nargs="*", default=["TOMATO_2__"])
    ap.add_argument("--n-images", type=int, default=15)
    ap.add_argument("--attempts", type=int, default=60)
    ap.add_argument("--sizes", type=int, nargs="*", default=[1024, 768, 512])
    ap.add_argument("--greens", type=float, nargs="*", default=[0.05, 0.10, 0.15, 0.20])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    img_dir = paths.REAL / "images" / a.split
    lbl_dir = paths.REAL / "labels" / a.split
    cand = [p for p in sorted(img_dir.iterdir())
            if p.is_file() and _matches_sources(p.name, a.sources)]
    rng = random.Random(a.seed)
    rng.shuffle(cand)
    cand = cand[:a.n_images]
    print(f"probing {len(cand)} images matching {a.sources} in {img_dir}")

    for size in a.sizes:
        boxfree = 0
        exg_vals = []
        imgs_with_boxfree = 0
        for ip in cand:
            im = Image.open(ip).convert("RGB")
            W, H = im.size
            if W < size or H < size:
                continue
            boxes = _boxes_px(lbl_dir / f"{ip.stem}.txt", W, H)
            found_here = 0
            for _ in range(a.attempts):
                x = rng.randint(0, W - size); y = rng.randint(0, H - size)
                if any(not (bx2 <= x or bx1 >= x + size or by2 <= y or by1 >= y + size)
                       for bx1, by1, bx2, by2 in boxes):
                    continue
                found_here += 1; boxfree += 1
                exg_vals.append(_mean_exg(im.crop((x, y, x + size, y + size))))
            if found_here:
                imgs_with_boxfree += 1
        line = (f"size={size:4d} | box-free crops={boxfree:4d} "
                f"({imgs_with_boxfree}/{len(cand)} imgs have >=1) | ")
        if exg_vals:
            ev = np.array(exg_vals)
            passes = " ".join(f"g<={g}:{int((ev <= g).sum())}" for g in a.greens)
            line += f"ExG min/med/max={ev.min():.3f}/{np.median(ev):.3f}/{ev.max():.3f} | soil {passes}"
        else:
            line += "NO box-free crops -> dense field, not a greenness problem"
        print(line)


if __name__ == "__main__":
    main()
