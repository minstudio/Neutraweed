"""Tile an already-materialised YOLO dataset in place (Stage A, cluster path).

prepare_data tiles from the raw VOC sources — but on the cluster only the
converted data/real/ exists (full images + YOLO labels), not the TOMATO_*/ VOC
folders. This tiles data/real/{images,labels}/<split> directly: each full frame
becomes overlapping tiles, boxes are remapped (edge-clipped ones dropped), empty
tiles dropped, and the original full frame is removed. Split membership is
preserved (tiles stay in their image's split → no leakage).

WARNING: in-place and one-shot — running it twice would tile the tiles. Your
local data/real is the backup if needed.

Run:  python -m src.data_prep.tile_dataset
"""

from __future__ import annotations

from collections import Counter

from ..common import classes, paths
from .tile import tile_boxes


def _read_yolo_px(lbl_path, W, H):
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                boxes.append((c, (cx - bw / 2) * W, (cy - bh / 2) * H,
                              (cx + bw / 2) * W, (cy + bh / 2) * H))
    return boxes


def _yolo_line(cid, x1, y1, x2, y2, w, h) -> str:
    return (f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")


def tile_split(split: str, size: int, stride: int, keep_frac: float,
               drop_empty: bool, out_img_dir=None, out_lbl_dir=None, remove=True):
    from PIL import Image

    img_dir = paths.REAL / "images" / split
    lbl_dir = paths.REAL / "labels" / split
    out_img_dir = out_img_dir or img_dir
    out_lbl_dir = out_lbl_dir or lbl_dir
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    originals = sorted(p for p in img_dir.iterdir() if p.is_file())
    n_tiles = 0
    cls: Counter = Counter()
    for ip in originals:
        uid = ip.stem
        lp = lbl_dir / f"{uid}.txt"
        img = Image.open(ip).convert("RGB")
        W, H = img.size
        pboxes = _read_yolo_px(lp, W, H)
        for ti, (tx, ty, tw, th, kept) in enumerate(
            tile_boxes(W, H, pboxes, size, stride, keep_frac)
        ):
            if drop_empty and not kept:
                continue
            stem = f"{uid}_t{ti}"
            img.crop((tx, ty, tx + tw, ty + th)).save(
                out_img_dir / f"{stem}.jpg", quality=92)
            lines = [_yolo_line(c, x1, y1, x2, y2, tw, th) for c, x1, y1, x2, y2 in kept]
            (out_lbl_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            for c, *_ in kept:
                cls[classes.CLASS_NAMES[c]] += 1
            n_tiles += 1
        if remove:
            ip.unlink()
            if lp.exists():
                lp.unlink()
    return n_tiles, cls


def main() -> None:
    from ..common.config import load_config

    t = load_config("base.yaml")["tiling"]
    print(f"Tiling data/real in place: size={t['size']} stride={t['stride']} "
          f"keep_frac={t['keep_frac']} drop_empty={t['drop_empty']}")
    print(f"{'split':6s}{'tiles':>8s}   " + "  ".join(f"{c:>6s}" for c in classes.CLASS_NAMES))
    for sp in paths.SPLITS:
        n, cls = tile_split(sp, t["size"], t["stride"], t["keep_frac"], t["drop_empty"])
        print(f"{sp:6s}{n:8d}   " + "  ".join(f"{cls[c]:6d}" for c in classes.CLASS_NAMES))
    print("Done. data/real is now tiled; real.yaml unchanged (still points to images/<split>).")


if __name__ == "__main__":
    main()
