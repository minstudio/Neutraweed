"""Measure the real images' APPEARANCE statistics, and score a pool against them.

fit_scale_prior.py answers "how big are real weeds". This answers "what do real
frames look like" — foliage colour, how dark and how large real cast shadows
are, and how blocky the background is. The compositor's photoreal parameters are
set from these numbers rather than by eye.

  python scripts/fit_appearance_prior.py --split train
  python scripts/fit_appearance_prior.py --split train --score sd35cut_v7

The --score form composites nothing: it reads an existing pool's images and
prints them beside the real targets, so "is the pool realistic" becomes a table
instead of an argument. Login-safe: ~45 real frames and ~60 pool images.

WHAT THE NUMBERS MEAN

  foliage S / V         HSV of pixels the ExG matte calls plant, inside GT boxes.
                        A pool whose plants are more saturated than real is
                        over-corrected, which is what happens if you harden a
                        matte that was previously bleeding soil through.
  shadow depth          1 - V(p5)/V(median) in a ring of soil around each plant.
                        How much darker the darkest nearby soil is.
  shadow area           fraction of that ring below 0.80x the local median.
  blockiness R2         share of a frame's smoothed brightness variation
                        explained by a piecewise-constant 2x2 step. Photographs
                        vary smoothly; a 2x2 mosaic of unrelated crops does not.

IMPORTANT. The soil crops used as composite backgrounds are real photographs and
already carry real shadow. Measured on this dataset, backgrounds alone score
0.260 depth / 0.090 area against a real target of 0.273 / 0.096 — i.e. the
shadow budget is nearly spent before a single synthetic shadow is drawn. Any
cast shadow added on top must be small or the pool ends up darker than reality.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import paths  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
EXG_T = 0.06


def _exg(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float32)
    return (2 * a[..., 1] - a[..., 0] - a[..., 2]) / (a.sum(2) + 1e-6)


def _boxes(lp: Path, W: int, H: int):
    out = []
    if not lp.exists():
        return out
    for line in lp.read_text(encoding="utf-8").splitlines():
        q = line.split()
        if len(q) == 5:
            _, cx, cy, bw, bh = (float(v) for v in q)
            out.append((int((cx - bw / 2) * W), int((cy - bh / 2) * H),
                        int((cx + bw / 2) * W), int((cy + bh / 2) * H)))
    return out


def blockiness(rgb: np.ndarray) -> float:
    import cv2

    V = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2].astype(np.float32)
    V = cv2.GaussianBlur(V, (0, 0), max(2.0, min(V.shape) / 85.0))
    h, w = V.shape
    fit = np.zeros_like(V)
    for ys, xs in ((slice(0, h // 2), slice(0, w // 2)),
                   (slice(0, h // 2), slice(w // 2, w)),
                   (slice(h // 2, h), slice(0, w // 2)),
                   (slice(h // 2, h), slice(w // 2, w))):
        fit[ys, xs] = V[ys, xs].mean()
    tot = ((V - V.mean()) ** 2).sum()
    return float(1 - ((V - fit) ** 2).sum() / max(tot, 1e-6))


def measure(rgb: np.ndarray, boxes, acc) -> None:
    import cv2

    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    plant = _exg(rgb) > EXG_T
    for (x1, y1, x2, y2) in boxes[:12]:
        if min(x2 - x1, y2 - y1) < 40:
            continue
        sub = (slice(max(0, y1), min(H, y2)), slice(max(0, x1), min(W, x2)))
        pm = plant[sub]
        if pm.sum() < 200:
            continue
        acc["fol_s"].append(float(hsv[sub][..., 1][pm].mean()))
        acc["fol_v"].append(float(hsv[sub][..., 2][pm].mean()))
        acc["fol_exg"].append(float(_exg(rgb)[sub][pm].mean()))

        # Relighting statistics. The three above are GLOBAL means over a pool,
        # and a pool can match all of them while every individual plant is still
        # lit wrongly for the ground it stands on. These are per-plant and local,
        # which is what the eye actually judges.
        pv = hsv[sub][..., 2][pm].astype(np.float32)
        if pv.size >= 200:
            lo = float(np.percentile(pv, 10))
            # How much darker the plant's own shaded parts are than its lit
            # parts. A plant under hard sun self-shades; a diffusion render lit
            # softly and frontally does not, and no global gain can add it.
            acc["fol_contrast"].append(float(np.percentile(pv, 90)) / max(lo, 1.0))

        # Plant brightness against the soil immediately around it, as a ratio.
        # Foliage has lower albedo than dry pale soil, so this is well below 1
        # and the compositor must not drive it to 1 by matching means.
        pad = int(0.35 * max(x2 - x1, y2 - y1))
        ry0, ry3 = max(0, y1 - pad), min(H, y2 + pad)
        rx0, rx3 = max(0, x1 - pad), min(W, x2 + pad)
        loc = ~plant[ry0:ry3, rx0:rx3]
        lv = hsv[ry0:ry3, rx0:rx3][..., 2][loc].astype(np.float32)
        ls = hsv[ry0:ry3, rx0:rx3][..., 1][loc].astype(np.float32)
        if lv.size >= 300:
            m = float(np.median(lv))
            if m > 1.0:
                acc["rel_v"].append(float(pv.mean()) / m)
            ms = float(np.median(ls))
            if ms > 1.0:
                acc["rel_s"].append(float(hsv[sub][..., 1][pm].mean()) / ms)

        r = int(max(x2 - x1, y2 - y1) * 0.8)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        y0, y3 = max(0, cy - r), min(H, cy + r)
        x0, x3 = max(0, cx - r), min(W, cx + r)
        if y3 - y0 < 40 or x3 - x0 < 40:
            continue
        ring = ~plant[y0:y3, x0:x3]
        V = hsv[y0:y3, x0:x3][..., 2][ring].astype(np.float32)
        if V.size < 500:
            continue
        med = float(np.median(V))
        acc["sh_depth"].append(1.0 - float(np.percentile(V, 5)) / max(med, 1.0))
        acc["sh_area"].append(float((V < med * 0.80).mean()))
        acc["soil_v"].append(med)


def _new_acc():
    return {k: [] for k in ("fol_s", "fol_v", "fol_exg", "fol_contrast",
                            "rel_v", "rel_s",
                            "sh_depth", "sh_area", "soil_v", "block")}


def _summarise(acc) -> dict:
    out = {}
    for k, v in acc.items():
        if not v:
            continue
        a = np.asarray(v, dtype=np.float64)
        out[k] = {"n": int(a.size), "p10": float(np.percentile(a, 10)),
                  "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90))}
    return out


ROWS = [("fol_s", "foliage HSV saturation", "{:.0f}"),
        ("fol_v", "foliage HSV value", "{:.0f}"),
        ("fol_exg", "foliage ExG", "{:.3f}"),
        ("fol_contrast", "plant self-shading p90/p10", "{:.2f}"),
        ("rel_v", "plant V / local soil V", "{:.3f}"),
        ("rel_s", "plant S / local soil S", "{:.3f}"),
        ("soil_v", "soil V around plants", "{:.0f}"),
        ("sh_depth", "shadow depth", "{:.3f}"),
        ("sh_area", "shadow area", "{:.3f}"),
        ("block", "2x2 blockiness R2", "{:.3f}")]


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-real", type=int, default=45)
    ap.add_argument("--n-pool", type=int, default=60)
    ap.add_argument("--score", default=None,
                    help="pool name under data/synthetic/ to score against real")
    ap.add_argument("--tile", type=int, default=1536,
                    help="native window cut from each real frame (= tiling.size)")
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="size that window is shown at (= detector imgsz, and the "
                         "pool's canvas size). Real and pool must match here.")
    ap.add_argument("--tiles-per-frame", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", type=Path, default=paths.REAL / "appearance_prior.json")
    args = ap.parse_args()

    img_dir = paths.REAL / "images" / args.split
    lbl_dir = paths.REAL / "labels" / args.split
    files = [p for p in sorted(img_dir.iterdir()) if p.suffix in IMG_EXT]
    random.Random(args.seed).shuffle(files)

    # Measure real on TILES, not on full frames.
    #
    # A pool image is a 1024 px canvas. A real frame is ~3400x3700. Measuring the
    # two directly against each other compares different spatial scales: the
    # shadow ring around a plant covers a very different amount of soil, and a
    # full frame has far fewer neighbouring plants inside that ring than a
    # 1024 px composite holding 3-12 of them. That mismatch alone produced an
    # apparent "+57% shadow depth" for a pool whose cutouts measure clean.
    # Cutting the frame the way the tiler does (tiling.size window -> imgsz)
    # makes the comparison like for like.
    tile = int(args.tile)
    out = int(args.imgsz)
    real = _new_acc()
    for p in files[:args.n_real]:
        frame = np.asarray(Image.open(p).convert("RGB"))
        H, W = frame.shape[:2]
        fb = _boxes(lbl_dir / f"{p.stem}.txt", W, H)
        if not fb or min(H, W) < tile:
            continue
        rr = random.Random(hash(p.name) % 9973)
        got = 0
        for _ in range(30):
            if got >= args.tiles_per_frame:
                break
            x, y = rr.randint(0, W - tile), rr.randint(0, H - tile)
            k = out / tile
            tb = []
            for (x1, y1, x2, y2) in fb:
                cx1, cy1 = max(x1, x), max(y1, y)
                cx2, cy2 = min(x2, x + tile), min(y2, y + tile)
                if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
                    continue
                # same keep_frac rule the tiler uses
                if (cx2 - cx1) * (cy2 - cy1) < 0.4 * (x2 - x1) * (y2 - y1):
                    continue
                tb.append((int((cx1 - x) * k), int((cy1 - y) * k),
                           int((cx2 - x) * k), int((cy2 - y) * k)))
            if not tb:
                continue
            t = np.asarray(Image.fromarray(frame[y:y + tile, x:x + tile])
                           .resize((out, out), Image.LANCZOS))
            measure(t, tb, real)
            real["block"].append(blockiness(t))
            got += 1

    cols = [("REAL", _summarise(real))]

    if args.score:
        pdir = paths.SYNTHETIC / args.score
        pimg = sorted((pdir / "images").glob("*.*"))
        if not pimg:
            raise SystemExit(f"no images in {pdir / 'images'}")
        random.Random(args.seed).shuffle(pimg)
        pool = _new_acc()
        for p in pimg[:args.n_pool]:
            rgb = np.asarray(Image.open(p).convert("RGB"))
            H, W = rgb.shape[:2]
            measure(rgb, _boxes(pdir / "labels" / f"{p.stem}.txt", W, H), pool)
            pool["block"].append(blockiness(rgb))
        cols.append((args.score.upper()[:16], _summarise(pool)))

    w = 22
    print(f"{'':26s}" + "".join(f"{n:>{w}s}" for n, _ in cols))
    for key, label, fmt in ROWS:
        line = f"{label:26s}"
        for _, s in cols:
            if key not in s:
                line += f"{'-':>{w}s}"
                continue
            e = s[key]
            line += f"{fmt.format(e['p50']) + '  [' + fmt.format(e['p10']) + ',' + fmt.format(e['p90']) + ']':>{w}s}"
        print(line)

    if len(cols) == 2:
        print("\ngap vs real (p50), as a fraction of the real value:")
        for key, label, _ in ROWS:
            a, b = cols[0][1].get(key), cols[1][1].get(key)
            if not a or not b:
                continue
            g = (b["p50"] - a["p50"]) / max(abs(a["p50"]), 1e-6)
            flag = "  <-- off" if abs(g) > 0.20 else ""
            print(f"  {label:26s}{g:+7.1%}{flag}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"split": args.split, "exg_threshold": EXG_T, "real": _summarise(real)},
        indent=1), encoding="utf-8")
    print(f"\n[appearance] -> {args.out}")


if __name__ == "__main__":
    main()
