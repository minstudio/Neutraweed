"""Is the composited soil as textured as real soil, and as uniform across a frame?

`fit_appearance_prior.py` measures soil BRIGHTNESS and how blocky a frame's
brightness is. Neither says anything about grain. Two ways the soil can be real
photography and still not read as a real photograph:

  1. The bank is smoother than real soil. `extract_backgrounds` keeps tiles with
     no weed box AND low greenness. In a dense field the regions passing both
     are disproportionately smooth, featureless patches — the same selection
     effect that makes the bank darker than average (see PHOTOREAL_STATE 4).

  2. The grain is uneven WITHIN a frame. A mosaic draws its quadrants from four
     different photographs, which may differ in focus, distance or exposure. A
     real photograph does not change sharpness across its own width, so a frame
     whose quadrants differ in grain reads as assembled even when every quadrant
     is genuine.

Measured as high-frequency energy: mean |I - blur(I)| over soil pixels, and the
spread of that quantity across the frame's four quadrants.

  python scripts/soil_texture_check.py --pool sd35cut_v8
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import paths  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}


def _hf(gray: np.ndarray) -> np.ndarray:
    import cv2
    return np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.2))


def _soil(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float32)
    exg = (2 * a[..., 1] - a[..., 0] - a[..., 2]) / (a.sum(2) + 1e-6)
    return exg < 0.03


def measure(rgb: np.ndarray):
    """-> (mean grain over soil, spread across quadrants as max/min)."""
    g = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    hf, soil = _hf(g), _soil(rgb)
    if soil.sum() < 2000:
        return None
    h, w = g.shape
    q = []
    for ys, xs in ((slice(0, h // 2), slice(0, w // 2)),
                   (slice(0, h // 2), slice(w // 2, w)),
                   (slice(h // 2, h), slice(0, w // 2)),
                   (slice(h // 2, h), slice(w // 2, w))):
        m = soil[ys, xs]
        if m.sum() < 400:
            continue
        q.append(float(hf[ys, xs][m].mean()))
    if len(q) < 4:
        return None
    return float(hf[soil].mean()), max(q) / max(min(q), 1e-6)


def scan(files, label, n, tile=None, out=None):
    from PIL import Image
    import random

    random.Random(0).shuffle(files)
    grain, spread = [], []
    for f in files[:n]:
        try:
            im = Image.open(f).convert("RGB")
        except Exception:
            continue
        if tile and out:
            W, H = im.size
            if min(W, H) < tile:
                continue
            rr = random.Random(hash(Path(f).name) % 9973)
            x, y = rr.randint(0, W - tile), rr.randint(0, H - tile)
            im = im.crop((x, y, x + tile, y + tile)).resize((out, out), Image.LANCZOS)
        r = measure(np.asarray(im))
        if r:
            grain.append(r[0])
            spread.append(r[1])
    if not grain:
        print(f"  {label:24s} no usable frames")
        return
    g, s = np.asarray(grain), np.asarray(spread)
    print(f"  {label:24s} n={g.size:4d}   grain {g.mean():6.2f} "
          f"[{np.percentile(g, 10):5.2f},{np.percentile(g, 90):5.2f}]"
          f"   quadrant spread {np.median(s):5.2f}x "
          f"[{np.percentile(s, 10):.2f},{np.percentile(s, 90):.2f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tile", type=int, default=1536)
    ap.add_argument("--imgsz", type=int, default=1024)
    args = ap.parse_args()

    print("\ngrain = mean |I - blur(I)| over soil pixels; higher is more textured")
    print("quadrant spread = max/min of that across the frame's four quadrants;")
    print("a real photograph should be near 1.0\n")

    real = [p for p in Path(paths.REAL / "images" / args.split).iterdir()
            if p.suffix in IMG_EXT]
    scan([str(p) for p in real], "REAL tiles", args.n, args.tile, args.imgsz)

    bank = sorted(glob.glob(str(paths.SYNTHETIC / args.pool / "backgrounds" / "*.png")))
    scan(bank, "background bank", args.n)

    pool = sorted(glob.glob(str(paths.SYNTHETIC / args.pool / "images" / "*.*")))
    scan(pool, f"{args.pool} frames", args.n)

    print("\nIf the bank's grain is well below real, the harvest is selecting smooth")
    print("soil and the fix is at harvest time. If the bank matches real but the")
    print("composited frames have a high quadrant spread, the mosaic is combining")
    print("crops of unlike sharpness and the fix is to select quadrants that agree.")


if __name__ == "__main__":
    main()
