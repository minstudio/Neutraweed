"""Export per-class real weed crops for the SD3.5 domain LoRA (Stage B).

Base SD3.5 has no grounding for these rare EPPO weeds, so text-only prompts draw
generic flowers/leaves and the species label is wrong. Fix: fine-tune a LoRA on
real crops of each species so SD learns their actual appearance. This script
assembles that training set from the real TRAIN split (CPU / login-node safe).

Crops each ground-truth weed box (padded, squared), resizes, and writes a single
diffusers image-folder with metadata.jsonl captioning every crop by species via
the SAME caption the generator uses (sd35_cutout.caption_for) — so the LoRA fires
at render time. Val/test are never touched (no leakage).

  python scripts/export_weed_crops.py --max-per-class 400 --size 768
  -> data/synthetic/lora_crops/{<crops>.png, metadata.jsonl}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import classes, paths
from src.generators.sd35_cutout import caption_for, stages_for


def _yolo_boxes(label_path: Path, W: int, H: int):
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cid, cx, cy, bw, bh = int(p[0]), *(float(v) for v in p[1:])
        out.append((cid, (cx - bw / 2) * W, (cy - bh / 2) * H,
                    (cx + bw / 2) * W, (cy + bh / 2) * H))
    return out


def _square_pad(x1, y1, x2, y2, W, H, pad_frac):
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + 2 * pad_frac)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nx1, ny1 = cx - side / 2, cy - side / 2
    nx2, ny2 = cx + side / 2, cy + side / 2
    # clamp inside the image while keeping it square where possible
    nx1, ny1 = max(0, nx1), max(0, ny1)
    nx2, ny2 = min(W, nx2), min(H, ny2)
    return int(nx1), int(ny1), int(nx2), int(ny2)


def export(split: str, out_name: str, size: int, max_per_class: int,
           pad_frac: float, min_box_px: int, seed: int,
           only: list[str] | None = None, strata: int = 0,
           size_captions: bool = False) -> dict:
    """Export LoRA training crops.

    `strata > 0` samples evenly across the class's own box-size distribution
    instead of taking whatever appears first. That matters when a species spans
    a wide morphological range: POROL runs from two-leaf cotyledons to sprawling
    mats, and first-come sampling fills the set with seedlings, so the LoRA never
    learns the mature form and cannot render it at any requested size.

    `size_captions` then labels each crop with the growth-stage phrase matching
    its size stratum, so the phrase and the appearance are associated during
    fine-tuning and `--stages` gives real morphological control at generation.
    """
    from PIL import Image

    img_dir = paths.REAL / "images" / split
    lbl_dir = paths.REAL / "labels" / split
    out_dir = paths.SYNTHETIC / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if not img_dir.exists():
        raise SystemExit(f"No real images at {img_dir} — run Stage A (prepare_data) first.")

    wanted = set(only) if only else set(classes.CLASS_NAMES)
    bad = wanted - set(classes.CLASS_NAMES)
    if bad:
        raise SystemExit(f"unknown class(es) {sorted(bad)}")

    imgs = [p for p in sorted(img_dir.iterdir()) if p.is_file()]
    rng = random.Random(seed)
    rng.shuffle(imgs)

    cand: dict[str, list] = {c: [] for c in wanted}
    for ip in imgs:
        with Image.open(ip) as probe:
            W, H = probe.size
        for cid, x1, y1, x2, y2 in _yolo_boxes(lbl_dir / f"{ip.stem}.txt", W, H):
            if cid < 0 or cid >= len(classes.CLASS_NAMES):
                continue
            cname = classes.CLASS_NAMES[cid]
            if cname not in wanted or min(x2 - x1, y2 - y1) < min_box_px:
                continue
            cand[cname].append((max(x2 - x1, y2 - y1), ip, W, H, x1, y1, x2, y2))

    picked: dict[str, list] = {}
    for cname, items in cand.items():
        if not items:
            picked[cname] = []
            continue
        if strata > 0 and len(items) > strata:
            items = sorted(items, key=lambda t: t[0])
            per = max(1, max_per_class // strata)
            out = []
            for s in range(strata):
                lo = s * len(items) // strata
                hi = (s + 1) * len(items) // strata
                chunk = items[lo:hi]
                rng.shuffle(chunk)
                out += chunk[:per]
            rng.shuffle(out)
            picked[cname] = out[:max_per_class]
        else:
            rng.shuffle(items)
            picked[cname] = items[:max_per_class]

    counts = {c: 0 for c in classes.CLASS_NAMES}
    meta = []
    for cname, items in picked.items():
        phrases = stages_for(cname) if size_captions else None
        order = sorted(range(len(items)), key=lambda i: items[i][0])
        rank = {j: r for r, j in enumerate(order)}
        for j, (_, ip, W, H, x1, y1, x2, y2) in enumerate(items):
            sx1, sy1, sx2, sy2 = _square_pad(x1, y1, x2, y2, W, H, pad_frac)
            if sx2 - sx1 < min_box_px or sy2 - sy1 < min_box_px:
                continue
            with Image.open(ip) as im:
                crop = im.convert("RGB").crop((sx1, sy1, sx2, sy2)).resize(
                    (size, size), Image.LANCZOS)
            fname = f"{cname}__{ip.stem}_{counts[cname]:04d}.png"
            crop.save(out_dir / fname)
            stage = None
            if phrases:
                b = min(len(phrases) - 1, rank[j] * len(phrases) // max(len(items), 1))
                stage = phrases[b]
            meta.append({"file_name": fname, "text": caption_for(cname, stage)})
            counts[cname] += 1

    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    counts = {c: n for c, n in counts.items() if c in wanted}
    print(f"[lora_crops] per-class: {counts}")
    if strata:
        for cname, items in picked.items():
            if items:
                s = [t[0] for t in items]
                print(f"[lora_crops] {cname} box longest side: "
                      f"min {min(s):.0f} med {sorted(s)[len(s)//2]:.0f} max {max(s):.0f}")
    print(f"[lora_crops] {len(meta)} crops + metadata.jsonl -> {out_dir}")
    if counts and min(counts.values()) < 30:
        thin = [c for c, n in counts.items() if n < 30]
        print(f"[lora_crops] WARNING thin classes {thin} (<30 crops) — LoRA may "
              f"underfit these species; lower --min-box-px or raise --max-per-class.")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="lora_crops")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--max-per-class", type=int, default=400)
    ap.add_argument("--pad-frac", type=float, default=0.30)
    ap.add_argument("--min-box-px", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", nargs="*", default=None,
                    help="export only these species (e.g. POROL) for a "
                         "single-species LoRA. Default: all five.")
    ap.add_argument("--size-strata", type=int, default=0, metavar="N",
                    help="sample crops evenly across N buckets of the class's own "
                         "box-size distribution instead of first-come. Use 4 for "
                         "species with a wide morphological range, so the LoRA "
                         "sees mature plants and not only seedlings.")
    ap.add_argument("--size-captions", action="store_true",
                    help="caption each crop with the growth-stage phrase matching "
                         "its size bucket, so --stages controls morphology at "
                         "generation time. Requires --size-strata to be useful.")
    args = ap.parse_args()
    export(args.split, args.out, args.size, args.max_per_class,
           args.pad_frac, args.min_box_px, args.seed,
           args.classes, args.size_strata, args.size_captions)


if __name__ == "__main__":
    main()
