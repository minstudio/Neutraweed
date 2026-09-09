"""Tile a few cutouts per species into one labelled contact sheet (QC + sharing).

Reads data/synthetic/<pool>/cutouts/<CLASS>/*.png (RGBA), composites each onto a
neutral tile, and lays them out one species per row with a label gutter. Light
(opens only per-class*cols images), so it won't trip the login-node killer — but
sbatch it on the standard partition if you want to be safe.

  python scripts/cutout_contactsheet.py --pool sd35cut_lora --per-class 6
  -> data/synthetic/sd35cut_lora/contact_sheet.png
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import classes, paths

BG = (232, 230, 224)
INK = (20, 24, 20)
def _ttf(bold: bool) -> str:
    """Resolve a DejaVu Sans ttf path portably (cluster has no Ubuntu font dir).
    Tries common locations, then matplotlib's bundled copy (always in this env)."""
    import os
    names = ["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]
    dirs = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu",
            "/usr/share/fonts/TTF", "/usr/share/fonts"]
    for dd in dirs:
        for nm in names:
            p = os.path.join(dd, nm)
            if os.path.exists(p):
                return p
    try:
        from matplotlib import font_manager as fm
        return fm.findfont("DejaVu Sans:bold" if bold else "DejaVu Sans")
    except Exception:
        return names[0]  # last resort; PIL will raise a clear error


FONT = _ttf(False)
FONTB = _ttf(True)


def build(pool: str, per_class: int, tile: int, seed: int) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    cut_dir = paths.SYNTHETIC / pool / "cutouts"
    if not cut_dir.exists():
        raise SystemExit(f"No cutouts at {cut_dir} — render them first (gen_sd35_cutout).")
    rng = random.Random(seed)
    gutter = 238
    pad = 14
    rows = len(classes.CLASS_NAMES)
    W = gutter + per_class * (tile + pad) + pad
    H = rows * (tile + pad) + pad + 60
    sheet = Image.new("RGB", (W, H), (18, 24, 20))
    d = ImageDraw.Draw(sheet)
    title = ImageFont.truetype(FONTB, 30)
    lab = ImageFont.truetype(FONTB, 24)
    sub = ImageFont.truetype(FONTB, 17)
    d.text((pad + 4, 16), f"SD3.5 + species-LoRA cutouts · {pool}", fill=(120, 230, 160), font=title)

    y0 = 62
    for r, c in enumerate(classes.CLASS_NAMES):
        files = list((cut_dir / c).glob("*.png"))
        rng.shuffle(files)
        y = y0 + r * (tile + pad) + pad
        disp = classes.CLASS_DISPLAY.get(c, c)
        d.text((pad + 4, y + tile // 2 - 26), c, fill=(235, 240, 235), font=lab)
        d.text((pad + 4, y + tile // 2 + 4), disp.split("(")[0].strip(), fill=(150, 170, 155), font=sub)
        for i in range(per_class):
            x = gutter + i * (tile + pad) + pad
            cell = Image.new("RGBA", (tile, tile), BG + (255,))
            if i < len(files):
                cut = Image.open(files[i]).convert("RGBA")
                cut.thumbnail((tile - 16, tile - 16), Image.LANCZOS)
                ox = (tile - cut.width) // 2
                oy = (tile - cut.height) // 2
                cell.alpha_composite(cut, (ox, oy))
            sheet.paste(cell.convert("RGB"), (x, y))
    out = paths.SYNTHETIC / pool / "contact_sheet.png"
    sheet.save(out)
    print(f"wrote {out} ({W}x{H})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="sd35cut_lora")
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--tile", type=int, default=220)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.pool, args.per_class, args.tile, args.seed)


if __name__ == "__main__":
    main()
