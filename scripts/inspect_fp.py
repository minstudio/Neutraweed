"""Render a class's false positives as a contact sheet, to eyeball the labels.

A false positive is only a false positive if the ground truth is right. When one
class shows a background-FP count comparable to its GT count, the two candidate
explanations — the model hallucinates, or the plants are there and unlabelled —
look identical in every metric and completely different to the eye.

Crops each unmatched detection with context, draws the prediction in red and any
ground truth in the crop in green (labelled), and tiles them into one PNG.

  python scripts/inspect_fp.py --runs 'hybrid_yolo26_sd35cut_lora_v2_r125_seed0' --cls POROL
  python scripts/inspect_fp.py --runs 'realonly_yolo26_seed0' --cls POROL --split val --n 48
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common import classes, paths  # noqa: E402

PREDS = paths.TABLES / "preds"


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)).astype(np.float32)


_TILE_RE = re.compile(r"^(.*)_t\d+$")


def _source_photo(tile_name: str) -> str:
    """Tiles are '<photo>_t<idx>'. Overlapping tiles of one photo show the same
    plant, so duplicates only ever arise within a source photo."""
    m = _TILE_RE.match(Path(tile_name).stem)
    return m.group(1) if m else Path(tile_name).stem


def _dhash(img, size: int = 8) -> int:
    """64-bit difference hash of a crop; near-identical crops collide."""
    from PIL import Image

    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | int(row[c] < row[c + 1])
    return bits


def _find_image(name: str, split: str) -> Path | None:
    d = paths.REAL / "images" / split
    p = d / name
    if p.exists():
        return p
    stem = Path(name).stem
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".PNG"):
        q = d / f"{stem}{ext}"
        if q.exists():
            return q
    return None


def main() -> None:
    from PIL import Image, ImageDraw

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="one cached run name (or glob; first match used)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cls", required=True, help="class code, e.g. POROL")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--sample", default="top", choices=["top", "random"],
                    help="'top' shows the most confident FPs, which flatters the "
                         "model. Use 'random' for an unbiased sample you can "
                         "actually count and quote.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dedup", dest="dedup", action="store_false",
                    help="keep near-duplicate crops. Tiles overlap by 512 px "
                         "(size 1536, stride 1024), so the same plant is detected "
                         "in up to 4 tiles; by default those are collapsed to one.")
    ap.add_argument("--dedup-dist", type=int, default=8,
                    help="max Hamming distance between 64-bit crop hashes to count "
                         "as the same plant (default 8; raise to dedup harder)")
    ap.add_argument("--valid-regions", type=Path, default=None,
                    help="regions JSON (from auto_region_from_gt or "
                         "draw_valid_region, mapped to tiles). Only detections "
                         "INSIDE the valid region are shown — the ones the region "
                         "correction cannot explain away.")
    ap.add_argument("--outside", action="store_true",
                    help="invert: show only detections OUTSIDE the valid region")
    ap.add_argument("--pad", type=float, default=1.4, help="crop size / box size")
    ap.add_argument("--cell", type=int, default=320)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.cls not in classes.CLASS_NAMES:
        raise SystemExit(f"--cls must be one of {classes.CLASS_NAMES}")
    cid = classes.CLASS_NAMES.index(args.cls)

    hits = [p for p in sorted(PREDS.glob(f"*_{args.split}.npz"))
            if fnmatch.fnmatch(p.name[: -len(f'_{args.split}.npz')], args.runs)]
    if not hits:
        raise SystemExit(f"no cache matches {args.runs!r} (split={args.split})")
    z = np.load(hits[0], allow_pickle=False)
    print(f"[fp] {hits[0].name}, class={args.cls}, conf>={args.conf}")

    vr = None
    if args.valid_regions:
        from scripts.ap_analysis import load_valid_regions, _valid_mask
        vr = load_valid_regions(args.valid_regions)
        where = "OUTSIDE" if args.outside else "INSIDE"
        print(f"[fp] restricted to detections {where} the valid region "
              f"({len(vr)} images have one)")

    found = []
    names = z["image_names"]
    n_region_dropped = 0
    for i in range(len(names)):
        k = f"{i:06d}"
        pred, gt = z[f"p{k}"], z[f"g{k}"]
        if pred.size == 0:
            continue
        p = pred[(pred[:, 5] == cid) & (pred[:, 4] >= args.conf)]
        if p.size == 0:
            continue
        if vr is not None:
            m = _valid_mask(p[:, :4], vr.get(str(names[i])))
            if args.outside:
                m = ~m
            n_region_dropped += int((~m).sum())
            p = p[m]
            if p.size == 0:
                continue
        g_same = gt[gt[:, 4] == cid][:, :4] if gt.size else np.zeros((0, 4), np.float32)
        ious = _iou(p[:, :4], g_same)
        best = ious.max(axis=1) if g_same.shape[0] else np.zeros(p.shape[0], np.float32)
        for j in np.flatnonzero(best < args.iou):
            found.append((float(p[j, 4]), str(names[i]), p[j, :4].copy(),
                          gt.copy() if gt.size else np.zeros((0, 5), np.float32)))

    if vr is not None:
        side = "inside" if args.outside else "outside"
        print(f"[fp] {n_region_dropped} detections dropped for being {side} the region")
    if not found:
        raise SystemExit(f"no unmatched {args.cls} detections above conf {args.conf}")
    total = len(found)
    if args.sample == "random":
        order = np.random.default_rng(args.seed).permutation(total)
        found = [found[int(i)] for i in order]
        how = f"at RANDOM (seed {args.seed}) — countable, unbiased"
    else:
        found.sort(key=lambda t: -t[0])
        how = "most confident first (biased toward the model's best guesses)"

    cell, cols = args.cell, args.cols
    kept, seen, dropped = [], {}, 0

    for conf, name, box, gt in found:
        if len(kept) >= args.n:
            break
        ip = _find_image(name, args.split)
        if ip is None:
            continue
        with Image.open(ip) as im:
            im = im.convert("RGB")
            W, H = im.size
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            side = max(max(box[2] - box[0], box[3] - box[1]) * args.pad, 64.0)
            x0 = int(np.clip(cx - side / 2, 0, max(W - 1, 0)))
            y0 = int(np.clip(cy - side / 2, 0, max(H - 1, 0)))
            x1 = int(np.clip(x0 + side, 0, W))
            y1 = int(np.clip(y0 + side, 0, H))
            crop = im.crop((x0, y0, x1, y1)).resize((cell, cell), Image.LANCZOS)

        if args.dedup:
            photo = _source_photo(name)
            h = _dhash(crop)
            if any(bin(h ^ prev).count("1") <= args.dedup_dist
                   for prev in seen.get(photo, ())):
                dropped += 1
                continue
            seen.setdefault(photo, []).append(h)

        kept.append((conf, name, box, gt, crop, (x0, y0, x1, y1)))

    print(f"[fp] {total} unmatched detections; showing {len(kept)} {how}")
    if args.dedup:
        print(f"[fp] dropped {dropped} near-duplicate crops (tiling overlaps 512 px, "
              f"so one plant appears in up to 4 tiles)")
    if not kept:
        raise SystemExit("nothing left to render")

    rows = (len(kept) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (18, 18, 18))
    draw_sheet = ImageDraw.Draw(sheet)

    for idx, (conf, name, box, gt, crop, (x0, y0, x1, y1)) in enumerate(kept):
        sx, sy = cell / max(x1 - x0, 1), cell / max(y1 - y0, 1)
        d = ImageDraw.Draw(crop)
        for gb in gt:
            gx0, gy0 = (gb[0] - x0) * sx, (gb[1] - y0) * sy
            gx1, gy1 = (gb[2] - x0) * sx, (gb[3] - y0) * sy
            if gx1 < 0 or gy1 < 0 or gx0 > cell or gy0 > cell:
                continue
            d.rectangle([gx0, gy0, gx1, gy1], outline=(60, 230, 90), width=3)
            d.text((gx0 + 3, gy0 + 3), classes.CLASS_NAMES[int(gb[4])], fill=(60, 230, 90))
        d.rectangle([(box[0] - x0) * sx, (box[1] - y0) * sy,
                     (box[2] - x0) * sx, (box[3] - y0) * sy],
                    outline=(240, 60, 60), width=3)
        d.text((4, cell - 14), f"{args.cls} {conf:.2f}", fill=(240, 60, 60))
        d.rectangle([0, 0, 26, 16], fill=(0, 0, 0))
        d.text((5, 3), f"{idx + 1}", fill=(255, 255, 0))

        r, c = divmod(idx, cols)
        sheet.paste(crop, (c * cell, r * cell))
        draw_sheet.rectangle([c * cell, r * cell, (c + 1) * cell - 1, (r + 1) * cell - 1],
                             outline=(60, 60, 60))

    out = args.out or (paths.FIGURES / f"fp_{args.cls}_{args.split}_{hits[0].stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"[fp] red = prediction with no matching {args.cls} label; green = ground truth")
    print(f"[fp] -> {out}")


if __name__ == "__main__":
    main()
