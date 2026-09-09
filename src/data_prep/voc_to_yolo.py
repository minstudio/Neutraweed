"""Materialise the frozen split as a YOLO dataset (Stage A).

Reads data/real/split.json, then for each image:
  * writes a YOLO label file (normalised cx cy w h, frozen class ids),
  * links the image into data/real/images/<split>/ (hardlink by default — the
    source JPGs are ~8 GB, so we avoid duplicating them).

Also emits data/real/real.yaml, the Ultralytics dataset descriptor used by the
real-only baseline and as the base for every hybrid dataset.

Run:  python -m src.data_prep.voc_to_yolo            (after make_splits)
      python -m src.data_prep.voc_to_yolo --copy     (copy instead of hardlink)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from ..common import classes, paths
from .voc import VocAnnotation, iter_annotations


def _load_split() -> dict:
    if not paths.SPLIT_FILE.exists():
        raise SystemExit("No split.json — run `python -m src.data_prep.make_splits` first.")
    with open(paths.SPLIT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _all_annotations() -> dict[str, VocAnnotation]:
    anns: dict[str, VocAnnotation] = {}
    for source, src_dir in paths.RAW_SOURCES.items():
        if src_dir.exists():
            for ann in iter_annotations(source, src_dir):
                anns[ann.uid] = ann
    return anns


def _to_yolo_lines(ann: VocAnnotation) -> tuple[list[str], Counter]:
    """Convert one annotation's boxes to normalised YOLO rows. Skips bad boxes."""
    lines: list[str] = []
    skipped: Counter = Counter()
    w, h = ann.width, ann.height
    if w <= 0 or h <= 0:
        skipped["no_size"] += 1
        return lines, skipped
    for box in ann.objects:
        if not classes.is_known(box.name):
            # crop (LYPES) / NR are excluded by design
            skipped[f"excluded:{box.name}"] += 1
            continue
        box = box.clamp(w, h)
        if not box.is_valid:
            skipped["degenerate_box"] += 1
            continue
        cid = classes.class_id(box.name)
        cx = (box.xmin + box.xmax) / 2.0 / w
        cy = (box.ymin + box.ymax) / 2.0 / h
        bw = (box.xmax - box.xmin) / w
        bh = (box.ymax - box.ymin) / h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, skipped


def _link(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        if mode == "symlink":
            os.symlink(src, dst)
        else:  # hardlink
            os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)  # cross-volume / permission fallback


def _clean_split_dirs() -> None:
    """Remove any previously materialised images/labels so a re-split can't leave
    orphans behind (a stale file in val/test from an earlier split would be
    silent leakage)."""
    for kind in ("images", "labels"):
        for sp in paths.SPLITS:
            d = paths.REAL / kind / sp
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()


def _pixel_boxes(ann):
    """(cls_id, x1, y1, x2, y2) pixel boxes for the 5 weed classes only."""
    out, skipped = [], Counter()
    w, h = ann.width, ann.height
    if w <= 0 or h <= 0:
        return out, Counter({"no_size": 1})
    for box in ann.objects:
        if not classes.is_known(box.name):
            skipped[f"excluded:{box.name}"] += 1
            continue
        b = box.clamp(w, h)
        if not b.is_valid:
            skipped["degenerate_box"] += 1
            continue
        out.append((classes.class_id(box.name), b.xmin, b.ymin, b.xmax, b.ymax))
    return out, skipped


def _yolo_line(cid, x1, y1, x2, y2, w, h) -> str:
    return (f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")


def convert(link_mode: str = "hardlink") -> None:
    from ..common.config import load_config

    split = _load_split()
    anns = _all_annotations()
    paths.ensure_dirs()
    _clean_split_dirs()
    tcfg = load_config("base.yaml").get("tiling", {"enabled": False})

    uid_to_split = split["uid_to_split"]
    counts: dict[str, int] = defaultdict(int)
    class_per_split: dict[str, Counter] = {s: Counter() for s in paths.SPLITS}
    skipped_total: Counter = Counter()
    missing_images = 0

    if tcfg.get("enabled"):
        _convert_tiled(uid_to_split, anns, tcfg, counts, class_per_split,
                       skipped_total, lambda: None)
        missing_images = sum(1 for u, s in uid_to_split.items()
                             if anns.get(u) is None or anns[u].image_path is None)
    else:
        for uid, sp in uid_to_split.items():
            ann = anns.get(uid)
            if ann is None or ann.image_path is None:
                missing_images += 1
                continue
            lines, skipped = _to_yolo_lines(ann)
            skipped_total.update(skipped)
            for ln in lines:
                class_per_split[sp][classes.CLASS_NAMES[int(ln.split()[0])]] += 1
            img_dst = paths.REAL / "images" / sp / f"{uid}{ann.image_path.suffix}"
            lbl_dst = paths.REAL / "labels" / sp / f"{uid}.txt"
            _link(ann.image_path, img_dst, link_mode)
            lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts[sp] += 1

    _write_dataset_yaml()
    mode = f"tiled {tcfg['size']}/{tcfg['stride']}" if tcfg.get("enabled") else link_mode
    _report(counts, class_per_split, skipped_total, missing_images, mode)


def _convert_tiled(uid_to_split, anns, tcfg, counts, class_per_split,
                   skipped_total, _unused) -> None:
    from PIL import Image

    from .tile import tile_boxes

    size, stride, keep = tcfg["size"], tcfg["stride"], tcfg["keep_frac"]
    drop_empty = tcfg.get("drop_empty", True)
    for uid, sp in uid_to_split.items():
        ann = anns.get(uid)
        if ann is None or ann.image_path is None:
            continue
        pboxes, skipped = _pixel_boxes(ann)
        skipped_total.update(skipped)
        img = Image.open(ann.image_path).convert("RGB")
        W, H = img.size
        for ti, (tx, ty, tw, th, kept) in enumerate(
            tile_boxes(W, H, pboxes, size, stride, keep)
        ):
            if drop_empty and not kept:
                continue
            stem = f"{uid}_t{ti}"
            img.crop((tx, ty, tx + tw, ty + th)).save(
                paths.REAL / "images" / sp / f"{stem}.jpg", quality=92
            )
            lines = [_yolo_line(c, x1, y1, x2, y2, tw, th) for c, x1, y1, x2, y2 in kept]
            (paths.REAL / "labels" / sp / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            for c, *_ in kept:
                class_per_split[sp][classes.CLASS_NAMES[c]] += 1
            counts[sp] += 1


def _write_dataset_yaml() -> None:
    descriptor = {
        "path": str(paths.REAL.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(classes.CLASS_NAMES)},
    }
    out = paths.REAL / "real.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(descriptor, f, sort_keys=False)
    print(f"Dataset descriptor -> {out}")


def _report(counts, class_per_split, skipped, missing, mode) -> None:
    print(f"\nVOC -> YOLO conversion ({mode}):")
    for s in paths.SPLITS:
        print(f"  {s:5s}: {counts.get(s,0):4d} images")
    print("\n  Objects per class per split:")
    header = "    {:8s}".format("class") + "".join(f"{s:>8s}" for s in paths.SPLITS)
    print(header)
    for c in classes.CLASS_NAMES:
        row = "    {:8s}".format(c) + "".join(f"{class_per_split[s][c]:8d}" for s in paths.SPLITS)
        print(row)
    if missing:
        print(f"\n  WARNING: {missing} annotations had no matching image (skipped).")
    if skipped:
        print(f"  Skipped boxes: {dict(skipped)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true", help="copy images instead of hardlinking")
    ap.add_argument("--symlink", action="store_true", help="symlink images (needs privilege on Windows)")
    args = ap.parse_args()
    mode = "copy" if args.copy else "symlink" if args.symlink else "hardlink"
    convert(mode)
