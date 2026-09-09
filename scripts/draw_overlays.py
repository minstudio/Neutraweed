"""Draw YOLO label boxes onto images for quick visual QC of any pool.

  python scripts/draw_overlays.py --pool composite --n 24
  python scripts/draw_overlays.py --pool sd35 --n 12
  python scripts/draw_overlays.py --images data/real/images/train \
      --labels data/real/labels/train --out /tmp/ov --n 12

Writes <pool>/overlays/ (or --out). scp those down to eyeball that boxes hug weeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import classes, paths  # noqa: E402

COLORS = [(220, 40, 40), (40, 200, 60), (40, 120, 220), (230, 200, 40), (180, 60, 220)]


def draw(images: Path, labels: Path, out: Path, n: int) -> int:
    from PIL import Image, ImageDraw

    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})[:n]
    for ip in imgs:
        lp = labels / f"{ip.stem}.txt"
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for row in (lp.read_text().splitlines() if lp.exists() else []):
            p = row.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            col = COLORS[c % len(COLORS)]
            d.rectangle([(cx - bw / 2) * W, (cy - bh / 2) * H,
                         (cx + bw / 2) * W, (cy + bh / 2) * H], outline=col, width=3)
            d.text(((cx - bw / 2) * W + 2, (cy - bh / 2) * H + 2),
                   classes.CLASS_NAMES[c], fill=col)
        im.save(out / ip.name)
    print(f"wrote {len(imgs)} overlays -> {out}")
    return len(imgs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw YOLO label overlays for QC.")
    ap.add_argument("--pool", help="shortcut: composite | sd35 | <gen> (uses data/synthetic/<gen>)")
    ap.add_argument("--images", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()

    if args.pool:
        base = paths.SYNTHETIC / args.pool
        images = args.images or base / "images"
        labels = args.labels or base / "labels"
        out = args.out or base / "overlays"
    else:
        if not (args.images and args.labels):
            ap.error("give --pool, or both --images and --labels")
        images, labels = args.images, args.labels
        out = args.out or images.parent / "overlays"

    draw(images, labels, out, args.n)


if __name__ == "__main__":
    main()
