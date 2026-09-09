"""Locate the near-black pixels in a composited pool.

v8 shipped a fix for one mechanism that produces black rectangles — a hole fill
that sealed the background of a border-touching cutout — and the pool's
near-black fraction did not move (v7 0.867%, v8 0.934%). So either that
mechanism was never the main source, or something upstream of the compositor
carries the black in already. This separates the candidates instead of guessing:

  A. inside the cutouts themselves, under the matte -> no compositor fix helps
  B. inside pasted instances in the frame -> matte or blend
  C. outside every label box -> backgrounds, shadows, or the mosaic

  python scripts/diagnose_black.py --pool sd35cut_v8
  python scripts/diagnose_black.py --pool sd35cut_v8 --dump 12
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import paths  # noqa: E402

DARK = 60          # sum over RGB; a pixel this dark is not soil and not foliage


def _dark(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.int32).sum(2) < DARK


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


def check_cutouts(pool: str, limit: int = 400) -> None:
    """A. Do the cutouts carry black INSIDE the matte?"""
    from PIL import Image

    files = sorted(glob.glob(str(paths.SYNTHETIC / pool / "cutouts" / "*" / "*.png")))
    if not files:
        print(f"  no cutouts under {paths.SYNTHETIC / pool / 'cutouts'}")
        return
    step = max(1, len(files) // limit)
    per_cls: dict[str, list[float]] = {}
    for f in files[::step]:
        a = np.asarray(Image.open(f).convert("RGBA"))
        al = a[:, :, 3].astype(np.float32) / 255.0
        if al.max() <= 0:
            continue
        inside = al >= 0.5 * al.max()
        if inside.sum() < 50:
            continue
        frac = float(_dark(a[:, :, :3])[inside].mean())
        per_cls.setdefault(Path(f).parent.name, []).append(frac)

    print("  fraction of each cutout's OWN matte interior that is near-black:")
    worst = 0.0
    for c, v in sorted(per_cls.items()):
        arr = np.asarray(v)
        worst = max(worst, float(arr.mean()))
        print(f"    {c:8s} n={arr.size:4d}  mean {arr.mean() * 100:6.3f}%  "
              f"p90 {np.percentile(arr, 90) * 100:6.3f}%  "
              f"max {arr.max() * 100:6.3f}%")
    if worst > 0.005:
        print("    -> the CUTOUTS carry the black. No compositor change fixes this;")
        print("       it is the matte that produced the bank, upstream of pasting.")
    else:
        print("    -> cutout interiors are clean. The black is added downstream.")


def check_opaque(pool: str, limit: int = 300) -> None:
    """A2. Run the pool's own cutouts through _opaque and see what it makes opaque.

    Section A measures the cutout as delivered: black under the matte, where the
    matte is the cutout's own alpha. But the compositor does not paste that
    alpha. `_opaque` thresholds, closes and hole-fills it, and every pixel the
    result covers is pasted at alpha 1 — including any pixel the ORIGINAL matte
    called background, where the RGB is the cutout's discarded backing and is
    frequently pure black.

    `_extend_edge_colour` repaints the untrusted RGB, but it is driven by the NEW
    mask, so anywhere `_opaque` grew into is treated as trustworthy plant and is
    not repainted. That is the gap this measures: black RGB sitting under alpha
    1 at the moment of blending.
    """
    from PIL import Image

    from src.annotate.composite import Photoreal, _opaque, _extend_edge_colour

    pr = Photoreal()
    files = sorted(glob.glob(str(paths.SYNTHETIC / pool / "cutouts" / "*" / "*.png")))
    if not files:
        print("  no cutouts")
        return
    step = max(1, len(files) // limit)

    grow, blk_raw, blk_fix = [], [], []
    for f in files[::step]:
        a = np.asarray(Image.open(f).convert("RGBA"))
        a0 = a[:, :, 3].astype(np.float32) / 255.0
        if a0.max() <= 0:
            continue
        rgb = a[:, :, :3].astype(np.float32)
        src = a0 >= pr.matte_thresh * a0.max()
        new = _opaque(a0, pr) > 0.5
        if new.sum() < 50:
            continue
        grow.append(float(new.sum() - src.sum()) / float(new.size))
        dark = _dark(rgb)
        blk_raw.append(float(dark[new].mean()))
        fixed = _extend_edge_colour(rgb.copy(), new.astype(np.uint8))
        blk_fix.append(float(_dark(fixed)[new].mean()))

    g, r, x = (np.asarray(v) for v in (grow, blk_raw, blk_fix))
    print(f"  n={g.size} cutouts")
    print(f"  _opaque mask growth over the thresholded matte: "
          f"mean {g.mean() * 100:6.3f}%  p90 {np.percentile(g, 90) * 100:6.3f}%  "
          f"max {g.max() * 100:6.3f}%  (cap is {pr.fill_max_growth * 100:.0f}%)")
    print(f"  near-black RGB under the pasted alpha, BEFORE edge_extend: "
          f"mean {r.mean() * 100:6.3f}%  p90 {np.percentile(r, 90) * 100:6.3f}%")
    print(f"  near-black RGB under the pasted alpha, AFTER  edge_extend: "
          f"mean {x.mean() * 100:6.3f}%  p90 {np.percentile(x, 90) * 100:6.3f}%")
    if x.mean() > 0.002:
        print("    -> _opaque is making black pixels opaque and edge_extend is")
        print("       not repairing them, because it is driven by the grown mask.")
    else:
        print("    -> this stage is clean; the black is added later in the blend.")


def check_backgrounds(pool: str, limit: int = 120) -> None:
    """C1. Does the soil bank carry black before anything is pasted?"""
    from PIL import Image

    files = sorted(glob.glob(str(paths.SYNTHETIC / pool / "backgrounds" / "*.png")))
    if not files:
        print(f"  no backgrounds under {paths.SYNTHETIC / pool / 'backgrounds'}")
        return
    fr = [float(_dark(np.asarray(Image.open(f).convert("RGB"))).mean())
          for f in files[:limit]]
    a = np.asarray(fr)
    print(f"  soil tiles n={a.size}  mean near-black {a.mean() * 100:6.3f}%  "
          f"max {a.max() * 100:6.3f}%")


def check_frames(pool: str, n: int, dump: int) -> None:
    """B vs C. Inside pasted instances, or outside every label box?"""
    import cv2
    from PIL import Image

    pdir = paths.SYNTHETIC / pool
    imgs = sorted(glob.glob(str(pdir / "images" / "*.*")))[:n]
    if not imgs:
        print(f"  no images under {pdir / 'images'}")
        return

    tot = in_box = out_box = 0
    shapes = []
    ranked = []
    for f in imgs:
        rgb = np.asarray(Image.open(f).convert("RGB"))
        H, W = rgb.shape[:2]
        d = _dark(rgb)
        if not d.any():
            continue
        boxes = _boxes(pdir / "labels" / f"{Path(f).stem}.txt", W, H)
        bm = np.zeros((H, W), bool)
        for (x1, y1, x2, y2) in boxes:
            bm[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
        tot += int(d.sum())
        in_box += int((d & bm).sum())
        out_box += int((d & ~bm).sum())
        ranked.append((float(d.mean()), f))

        k, lab = cv2.connectedComponents(
            cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((9, 9), np.uint8)), 8)
        for i in range(1, k):
            ys, xs = np.where(lab == i)
            if ys.size < 300:
                continue
            h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
            shapes.append((ys.size, ys.size / (h * w), w, h))

    if tot == 0:
        print("  no near-black pixels found")
        return

    print(f"  near-black pixels: {in_box / tot * 100:5.1f}% inside a label box, "
          f"{out_box / tot * 100:5.1f}% outside every box")
    if shapes:
        s = np.asarray(shapes, dtype=np.float64)
        print(f"  blobs >=300 px: n={len(shapes)}  median area {np.median(s[:, 0]):.0f} px"
              f"  median rect-fill {np.median(s[:, 1]):.2f}"
              f"  median bbox {np.median(s[:, 2]):.0f}x{np.median(s[:, 3]):.0f}")
        print("  rect-fill near 1.0 = filled rectangles (the matte bug);"
              " 0.4-0.7 = plant-shaped")

    if dump:
        out = paths.RESULTS / "figures" / f"black_{pool}"
        out.mkdir(parents=True, exist_ok=True)
        ranked.sort(reverse=True)
        for frac, f in ranked[:dump]:
            Image.open(f).save(out / f"{frac * 100:07.3f}pct_{Path(f).name}")
        print(f"  worst {dump} frames -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n", type=int, default=200, help="frames to scan")
    ap.add_argument("--dump", type=int, default=0,
                    help="save this many worst frames for eyeballing")
    args = ap.parse_args()

    print(f"\n=== {args.pool} ===")
    print("A. cutouts (upstream of the compositor)")
    check_cutouts(args.pool)
    print("A2. after _opaque, at the moment of blending")
    check_opaque(args.pool)
    print("C1. background bank")
    check_backgrounds(args.pool)
    print("B/C2. composited frames")
    check_frames(args.pool, args.n, args.dump)
    print()


if __name__ == "__main__":
    main()
