"""Cache raw detections for a set of checkpoints (GPU, one pass each).

The test field is 27 photos / ~530 tiles and cannot be made bigger without
breaking the frozen split. What we CAN do is stop throwing away the per-image
detections: dump them once, then re-analyse them for free — per-class AP,
AP by object size, and bootstrap confidence intervals over tiles, which uses
every tile as a resampling unit instead of leaning on 3 seeds.

Writes results/tables/preds/<run>_<split>.npz with, per image: predicted boxes
(xyxy, conf, cls) and ground-truth boxes (xyxy, cls), both in ORIGINAL image
pixels. Analyse with scripts/ap_analysis.py (CPU, login-safe).

  sbatch --export=ALL,PATTERN='realonly_yolo26_seed*',SPLIT=test scripts/slurm/cache_preds.slurm
  python scripts/cache_preds.py --pattern 'hybrid_yolo26_sd35cut_lora_v2_r125_seed*'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common import paths  # noqa: E402
from scripts.eval_perclass import dataset_for, discover  # noqa: E402

PREDS = paths.TABLES / "preds"


def _list_images(data_yaml: Path, split: str) -> list[Path]:
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    entry = cfg.get(split)
    if entry is None:
        raise SystemExit(f"{data_yaml} has no '{split}' entry")
    root = data_yaml.parent
    listing = (root / entry) if not Path(entry).is_absolute() else Path(entry)
    if listing.suffix == ".txt" and listing.exists():
        out = []
        for line in listing.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            out.append(p if p.is_absolute() else (root / p))
        return out
    d = listing if listing.is_dir() else (root / entry)
    return sorted(p for p in d.iterdir() if p.is_file())


def _gt_for(img: Path) -> np.ndarray:
    parts = list(img.parts)
    if "images" in parts:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        lp = Path(*parts).with_suffix(".txt")
    else:
        lp = img.with_suffix(".txt")
    if not lp.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in lp.read_text(encoding="utf-8").splitlines():
        q = line.split()
        if len(q) == 5:
            rows.append([float(x) for x in q])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def _loader(run_name: str):
    from ultralytics import RTDETR, YOLO

    return RTDETR if "rtdetr" in run_name else YOLO


def cache_one(run_name: str, ckpt: Path, data_yaml: Path, split: str,
              imgsz: int, conf: float, iou: float, batch: int, tag: str = "") -> Path:
    from PIL import Image

    Model = _loader(run_name)
    model = Model(str(ckpt))
    imgs = _list_images(data_yaml, split)
    if not imgs:
        raise SystemExit(f"no {split} images for {run_name}")

    store = {}
    names = []
    for i in range(0, len(imgs), batch):
        chunk = imgs[i:i + batch]
        res = model.predict([str(p) for p in chunk], imgsz=imgsz, conf=conf, iou=iou,
                            max_det=1000, verbose=False, stream=False)
        for p, r in zip(chunk, res):
            b = r.boxes
            if b is None or len(b) == 0:
                pred = np.zeros((0, 6), dtype=np.float32)
            else:
                pred = np.concatenate([
                    b.xyxy.cpu().numpy().astype(np.float32),
                    b.conf.cpu().numpy().astype(np.float32)[:, None],
                    b.cls.cpu().numpy().astype(np.float32)[:, None],
                ], axis=1)
            with Image.open(p) as im:
                W, H = im.size
            g = _gt_for(p)
            if g.size:
                cx, cy, bw, bh = g[:, 1] * W, g[:, 2] * H, g[:, 3] * W, g[:, 4] * H
                gt = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, g[:, 0]], 1)
            else:
                gt = np.zeros((0, 5), dtype=np.float32)
            key = f"{len(names):06d}"
            names.append(p.name)
            store[f"p{key}"] = pred
            store[f"g{key}"] = gt.astype(np.float32)
            store[f"s{key}"] = np.asarray([W, H], dtype=np.float32)

    PREDS.mkdir(parents=True, exist_ok=True)
    out = PREDS / f"{run_name}{tag}_{split}.npz"
    np.savez_compressed(out, image_names=np.asarray(names), imgsz=np.asarray([imgsz]),
                        conf=np.asarray([conf]), **store)
    n_p = sum(v.shape[0] for k, v in store.items() if k.startswith("p"))
    n_g = sum(v.shape[0] for k, v in store.items() if k.startswith("g"))
    print(f"  {run_name}: {len(names)} images, {n_g} gt, {n_p} preds -> {out.name}")
    return out


def main() -> None:
    from src.common.config import load_config

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="realonly_*")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default="",
                    help="suffix the cache filename, e.g. --tag hires when caching "
                         "at a non-default --imgsz, so the 1024 caches survive")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    tag = f"__{args.tag}" if args.tag else ""
    imgsz = args.imgsz or int(load_config("base.yaml")["detector"]["imgsz"])
    runs = discover(args.pattern)
    if not runs:
        raise SystemExit(f"no checkpoints with best.pt match {args.pattern}")
    print(f"[cache_preds] {len(runs)} checkpoints, split={args.split}, imgsz={imgsz}")
    for run_name, ckpt in runs:
        out = PREDS / f"{run_name}{tag}_{args.split}.npz"
        if args.skip_existing and out.exists():
            print(f"  skip {run_name}")
            continue
        dy = args.data or dataset_for(run_name)
        if not dy.exists():
            print(f"  WARN {run_name}: dataset.yaml not found at {dy}; skipping")
            continue
        cache_one(run_name, ckpt, dy, args.split, imgsz, args.conf, args.iou,
                  args.batch, tag)


if __name__ == "__main__":
    main()
