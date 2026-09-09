"""Render the SAME scenes under two blend modes, side by side, plus real tiles.

The point is to decide whether the photoreal compositor is worth a training run
by looking at it, not by reading a metric. Same seed both sides, so any visible
difference is the blend and nothing else.

  python scripts/preview_blend.py --pool sd35cut_lora_v2 --n 8 \
      --a feather --b photoreal --out results/figures/blend_ab.png

Add real tiles for reference (this is the comparison that actually matters —
the question is not "is B nicer than A" but "is B closer to the right column"):

  python scripts/preview_blend.py --pool sd35cut_lora_v2 --n 6 --with-real

Zoom in on single instances, where the matte and shadow differences live:

  python scripts/preview_blend.py --pool sd35cut_lora_v2 --n 8 --crop 320

Cheap: 8 scenes at 1024 is well under a minute on a login node.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotate.composite import Photoreal  # noqa: E402
from src.common import paths  # noqa: E402
from src.generators import composite_gen  # noqa: E402


def _gen(pool: str, blend: str, n: int, seed: int, imgsz: int, tmp: Path,
         photoreal: Photoreal | None, prefer_downscale: bool, bg_qc: bool):
    """Generate into a scratch pool so nothing real is touched."""
    out = tmp / blend
    (out / "images").mkdir(parents=True, exist_ok=True)

    g = composite_gen.CompositeGenerator(
        image_size=imgsz, blend=blend, name=pool,
        scale_mode="prior", bg_mode="tilematch",
        prefer_downscale=prefer_downscale, bg_qc=bg_qc, photoreal=photoreal)
    # Redirect output without disturbing the pool on disk.
    g.pool_dir = out
    g.generate(n, seed=seed)
    return sorted((out / "images").glob("*.png"))


def _real_tiles(n: int, seed: int, size: int) -> list[np.ndarray]:
    from PIL import Image

    d = paths.REAL / "images" / "train"
    if not d.exists():
        return []
    lbl = paths.REAL / "labels" / "train"
    cand = [p for p in sorted(d.iterdir()) if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
    # prefer tiles that actually contain weeds, otherwise the comparison is soil
    with_gt = [p for p in cand if (lbl / f"{p.stem}.txt").exists()
               and (lbl / f"{p.stem}.txt").stat().st_size > 0]
    cand = with_gt or cand
    random.Random(seed).shuffle(cand)
    out = []
    for p in cand[:n]:
        with Image.open(p) as im:
            out.append(np.asarray(im.convert("RGB").resize((size, size))))
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 12, 26], fill=(0, 0, 0))
    d.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(im)


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="sd35cut_lora_v2")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--a", default="feather")
    ap.add_argument("--b", default="photoreal")
    ap.add_argument("--with-real", action="store_true",
                    help="add a third column of real training tiles")
    ap.add_argument("--crop", type=int, default=0,
                    help="centre-crop each panel to this many pixels before "
                         "downscaling, to inspect matte and shadow up close")
    ap.add_argument("--cell", type=int, default=420, help="output px per panel")
    ap.add_argument("--no-shadow", action="store_true")
    ap.add_argument("--no-noise-match", action="store_true")
    ap.add_argument("--shadow-strength", type=float, default=0.45)
    ap.add_argument("--feather-px", type=float, default=1.2)
    ap.add_argument("--max-overlap", type=float, default=0.15)
    ap.add_argument("--prefer-downscale", action="store_true", default=True)
    ap.add_argument("--no-prefer-downscale", dest="prefer_downscale",
                    action="store_false")
    ap.add_argument("--bg-qc", action="store_true", default=True)
    ap.add_argument("--no-bg-qc", dest="bg_qc", action="store_false")
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/blend_ab.png"))
    args = ap.parse_args()

    pr = Photoreal(feather_px=args.feather_px, shadow=not args.no_shadow,
                   shadow_strength=args.shadow_strength,
                   noise_match=not args.no_noise_match,
                   max_overlap=args.max_overlap)

    tmp = Path(".preview_blend")
    cols = []
    for blend in (args.a, args.b):
        # Column A is the legacy path: it must be generated with the legacy
        # settings too, or the comparison silently includes the selection and
        # background changes as well as the blend.
        legacy = blend != "photoreal"
        files = _gen(args.pool, blend, args.n, args.seed, args.imgsz, tmp,
                     None if legacy else pr,
                     False if legacy else args.prefer_downscale,
                     False if legacy else args.bg_qc)
        cols.append((blend, [np.asarray(Image.open(f).convert("RGB")) for f in files]))

    if args.with_real:
        real = _real_tiles(args.n, args.seed, args.imgsz)
        if real:
            cols.append(("REAL", real))

    rows = min(len(v) for _, v in cols)
    cell = args.cell
    W = cell * len(cols)
    H = cell * rows
    sheet = Image.new("RGB", (W, H), (18, 18, 18))

    for ci, (title, imgs) in enumerate(cols):
        for ri in range(rows):
            a = imgs[ri]
            if args.crop:
                s = min(args.crop, a.shape[0], a.shape[1])
                y = (a.shape[0] - s) // 2
                x = (a.shape[1] - s) // 2
                a = a[y:y + s, x:x + s]
            im = Image.fromarray(a).resize((cell, cell), Image.LANCZOS)
            if ri == 0:
                im = Image.fromarray(_label(np.asarray(im), title))
            sheet.paste(im, (ci * cell, ri * cell))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"[preview] {rows} scenes x {len(cols)} columns -> {args.out}")
    print("[preview] look for: translucent grass blades (column A), missing "
          "shadows (A), plants sharper or softer than the soil, bright bands "
          "from blown background tiles.")


if __name__ == "__main__":
    main()
