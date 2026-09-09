"""Polished annotated full-scene overlays for any pool (QC + sharing).

Unlike draw_overlays.py (plain thin boxes, first N images), this ranks images by
box count + class diversity and draws LinkedIn-grade colored boxes with class
chips and a species legend — the same style as the real ground-truth samples, so
real and synthetic scenes look directly comparable.

  # a synthetic pool (composited scenes with perfect labels):
  python scripts/draw_scene_overlays.py --pool sd35cut_lora --n 4
  # or any images+labels dir:
  python scripts/draw_scene_overlays.py --images data/real/images/train \
      --labels data/real/labels/train --n 4 --out /tmp/ov
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import classes, paths

COLORS = {
    "SOLNI": (255, 59, 48), "POROL": (0, 210, 255), "SETVE": (255, 149, 0),
    "CYPRO": (255, 45, 149), "ECHCG": (255, 230, 0),
}
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
MAXSIDE = 1500
LEGEND_H = 132


def _boxes(lp: Path):
    out = []
    if not lp.exists():
        return out
    for ln in lp.read_text().splitlines():
        p = ln.split()
        if len(p) == 5:
            out.append((int(p[0]), *map(float, p[1:])))
    return out


def _rank(images: Path, labels: Path, n: int):
    scored = []
    for ip in images.iterdir():
        if ip.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        bx = _boxes(labels / f"{ip.stem}.txt")
        if not bx:
            continue
        scored.append((len({b[0] for b in bx}), len(bx), ip))
    scored.sort(key=lambda t: (t[0], -abs(t[1] - 8)), reverse=True)
    return [ip for *_, ip in scored[:n]]


def render(ip: Path, labels: Path, out_dir: Path, title: str):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(ip).convert("RGB")
    W, H = im.size
    bx = _boxes(labels / f"{ip.stem}.txt")
    s = MAXSIDE / max(W, H)
    if s < 1:
        im = im.resize((int(W * s), int(H * s)), Image.LANCZOS)
        W, H = im.size
    canvas = Image.new("RGB", (W, H + LEGEND_H), (14, 20, 16))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas, "RGBA")
    lw = max(3, int(W / 380))
    fsz = max(15, int(W / 62))
    f = ImageFont.truetype(FONTB, fsz)
    for cid, cx, cy, bw, bh in bx:
        c = classes.CLASS_NAMES[cid]
        col = COLORS[c]
        x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
        x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
        d.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        tw = d.textlength(c, font=f); th = fsz + 6
        ly = y1 - th if y1 - th > 0 else y1
        d.rectangle([x1, ly, x1 + tw + 12, ly + th], fill=col + (235,))
        d.text((x1 + 6, ly + 2), c, fill=(10, 12, 10), font=f)

    d.text((26, H + 16), title, fill=(120, 230, 160), font=ImageFont.truetype(FONTB, 24))

    def common(c):
        disp = classes.CLASS_DISPLAY.get(c, c)
        return disp[disp.find("(") + 1:disp.find(")")] if "(" in disp else disp

    items = [(c, f"{c}  {common(c)}") for c in classes.CLASS_NAMES]
    sw, gap = 26, 12
    # auto-fit: shrink the legend font/pad until the row fits the image width
    for fs in range(22, 12, -1):
        sf = ImageFont.truetype(FONTB, fs)
        pad = max(16, fs + 8)
        widths = [sw + gap + int(d.textlength(lab, font=sf)) for _, lab in items]
        total = sum(widths) + pad * (len(items) - 1)
        if total <= W - 40:
            break
    x = max(20, (W - total) // 2); y = H + 66
    for (c, lab), wd in zip(items, widths):
        d.rounded_rectangle([x, y + 4, x + sw, y + 30], radius=5, fill=COLORS[c])
        d.text((x + sw + gap, y + 4), lab, fill=(230, 240, 232), font=sf)
        x += wd + pad
    out_dir.mkdir(parents=True, exist_ok=True)
    op = out_dir / f"scene_{ip.stem}.png"
    canvas.save(op)
    print(f"wrote {op} ({canvas.size[0]}x{canvas.size[1]}, {len(bx)} boxes)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool")
    ap.add_argument("--images", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    if args.pool:
        base = paths.SYNTHETIC / args.pool
        images, labels = base / "images", base / "labels"
        out = args.out or base / "overlays_pretty"
        title = f"SYNTHETIC · {args.pool}"
    else:
        if not (args.images and args.labels):
            ap.error("give --pool, or both --images and --labels")
        images, labels = args.images, args.labels
        out = args.out or images.parent / "overlays_pretty"
        title = "GROUND TRUTH · 5 weed species"
    picks = _rank(images, labels, args.n)
    if not picks:
        raise SystemExit(f"No labelled images found in {images}")
    for ip in picks:
        render(ip, labels, out, title)


if __name__ == "__main__":
    main()
