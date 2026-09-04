"""Re-annotate generated images from their actual pixels (CLAUDE.md §4 Stage C).

The SD3.5 layout labels were wrong because the model didn't obey the layout
(boxes on bare soil). Fix: for each layout box, look at what the model ACTUALLY
rendered there — run greenness (ExG) inside the box; if there's a weed, tighten
the box to it and keep the layout's intended class; if there's only soil, drop
the box. This turns realistic-but-mislabelled SD3.5 images into a usable
generator arm (CPU-only).

Caveat: the class is the *intended* class (what we conditioned), not verified
species — SD3.5's rendering may not be species-accurate. Report accordingly.

Run:  python -m src.annotate.reannotate_greenness --gen sd35
"""

from __future__ import annotations

import argparse

import numpy as np

from ..common import paths
from .greenness import green_mask_in_box
from .masks import mask_to_bbox
from ..data_prep.tile_dataset import _read_yolo_px, _yolo_line

IMG_EXT = {".png", ".jpg", ".jpeg"}


def reannotate(gen: str = "sd35", padding: float = 0.1, min_area: int = 150,
               close_ksize: int = 5) -> dict:
    from PIL import Image

    pool = paths.SYNTHETIC / gen
    img_dir, lbl_dir = pool / "images", pool / "labels"
    if not img_dir.exists():
        raise SystemExit(f"No images at {img_dir} — generate the {gen} pool first.")

    kept = dropped = imgs = 0
    for ip in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        img = np.asarray(Image.open(ip).convert("RGB"))
        H, W = img.shape[:2]
        layout = _read_yolo_px(lbl_dir / f"{ip.stem}.txt", W, H)
        lines = []
        for cid, x1, y1, x2, y2 in layout:
            mask, _ = green_mask_in_box(img, (x1, y1, x2, y2), padding, close_ksize, 0.0)
            if int(mask.sum()) < min_area:      # model rendered no weed here -> drop
                dropped += 1
                continue
            bx = mask_to_bbox(mask)
            lines.append(_yolo_line(cid, *bx, W, H))
            kept += 1
        (lbl_dir / f"{ip.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        imgs += 1

    print(f"[reannotate {gen}] {imgs} imgs | kept {kept} boxes | dropped {dropped} "
          f"(soil-only). Labels now match generated content.")
    return {"images": imgs, "kept": kept, "dropped": dropped}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Re-annotate a generated pool from its pixels (greenness).")
    ap.add_argument("--gen", default="sd35")
    ap.add_argument("--padding", type=float, default=0.1)
    ap.add_argument("--min-area", type=int, default=150)
    args = ap.parse_args()
    reannotate(args.gen, args.padding, args.min_area)
