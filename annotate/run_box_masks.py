"""Phase 0 driver — turn GT boxes into instance masks (SAM2 or BiRefNet).

For each image in a split of the frozen real dataset, prompt the chosen
segmenter with every ground-truth box and save instance masks. Computes a
mask->GT-box IoU per instance (a good box-prompted mask fills its prompt box),
so you can compare methods head-to-head on the same images.

Methods (CLAUDE.md §4 Stage C):
  * sam2     — box-prompted SAM2. Strong on compact dicots; tends to drop the
               thin strands of grass-like monocots (CYPRO/SETVE/ECHCG).
  * birefnet — per-box BiRefNet matte. Crisp on thin stems/strands; the intended
               fix for the monocots.

Outputs (under data/real/masks/<method>/<split>/):
  <uid>.png    uint16 instance label-map (0=bg, i=instance i)
  <uid>.json   {width,height, instances:[{id,class,gt_box,mask_box,iou,area}]}
  _overlays/<uid>.jpg   colour overlay WITH class labels drawn on each mask
And results/tables/mask_qc_<method>_<split>.json (aggregate IoU per class).

  python -m src.annotate.run_box_masks --method sam2     --split test --limit 20
  python -m src.annotate.run_box_masks --method birefnet --split test --limit 20
  python -m src.annotate.run_box_masks --method sam2 --split train --resume  # continue a crashed run
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm isn't installed
    def tqdm(it, **_kw):
        return it

from ..common import paths
from ..common.config import load_config
from . import masks as mask_utils
from ..qc.mask_iou import _iou  # pixel-space box IoU


def _read_yolo_boxes(label_path: Path, w: int, h: int):
    """Return list of (class_name, xyxy_pixels) from a YOLO label file."""
    from ..common import classes

    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cid, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        out.append((classes.CLASS_NAMES[cid], np.array([x1, y1, x2, y2], dtype=np.float32)))
    return out


def _save_label_map(instance_masks: list[np.ndarray], shape, out_png: Path) -> None:
    from PIL import Image

    label_map = np.zeros(shape, dtype=np.uint16)
    for i, m in enumerate(instance_masks, start=1):
        label_map[m > 0] = i
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # uint16 array -> PIL infers mode 'I;16'; passing mode= is deprecated.
    Image.fromarray(label_map).save(out_png)


def _save_overlay(image, instance_masks, labels, out_jpg: Path) -> None:
    """Tint each mask and draw its class label, so non-botanists can read it."""
    try:
        import cv2
    except ImportError:
        return
    h, w = image.shape[:2]
    rng = np.random.default_rng(0)
    canvas = image.copy()
    font_scale = max(0.8, w / 1800.0)
    thick = max(2, int(w / 700))

    for m, label in zip(instance_masks, labels):
        colour = tuple(int(c) for c in rng.integers(60, 255, size=3))
        canvas[m > 0] = (0.5 * canvas[m > 0] + 0.5 * np.array(colour)).astype(canvas.dtype)
        box = mask_utils.mask_to_bbox(m)
        if box is None:
            continue
        x1, y1, _x2, _y2 = box
        # readable caption: filled background box + text at the mask's top-left
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ty = max(th + 4, int(y1))
        cv2.rectangle(canvas, (int(x1), ty - th - 6), (int(x1) + tw + 6, ty + 4), colour, -1)
        cv2.putText(canvas, label, (int(x1) + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thick, cv2.LINE_AA)

    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_jpg), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _save_debug(segmenter, instance_masks, shape, out_png: Path) -> None:
    """Viewable grayscale of the raw mask. For BiRefNet this is the SOFT matte
    (pre-threshold) so you can see strands that the 0.5 cutoff may drop;
    otherwise it's the union of the binary instance masks."""
    from PIL import Image

    soft = getattr(segmenter, "last_soft", None)
    if soft is not None:
        img = (np.clip(soft, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        h, w = shape
        img = np.zeros((h, w), dtype=np.uint8)
        for m in instance_masks:
            img[m > 0] = 255
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_png)


def _report_device(method: str, device) -> None:
    """Print the device up front so a silent CPU fallback (the usual 'why is it
    so slow') is obvious."""
    line = f"[run_box_masks] method={method}  device={device}"
    try:
        import torch

        avail = torch.cuda.is_available()
        line += f"  cuda_available={avail}"
        wants_cuda = device is None or "cuda" in str(device)
        if avail and wants_cuda:
            line += f"  gpu={torch.cuda.get_device_name(0)}"
        elif wants_cuda and not avail:
            line += "  !! WARNING: cuda requested but unavailable -> CPU (slow)"
    except Exception as exc:  # torch missing / odd build — don't crash the run
        line += f"  (torch check skipped: {exc})"
    print(line, flush=True)


class _MockSegmenter:
    """Returns a filled rectangle for each prompt box — for plumbing tests only."""

    def masks_from_boxes(self, image, boxes_xyxy):
        h, w = image.shape[:2]
        out = []
        for x1, y1, x2, y2 in np.asarray(boxes_xyxy):
            m = np.zeros((h, w), dtype=np.uint8)
            m[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))] = 1
            out.append(m)
        return out


def _build_segmenter(method: str, cfg: dict):
    if method == "sam2":
        from .sam2 import SAM2Annotator
        c = cfg["annotate"]["sam2"]
        return SAM2Annotator(model_size=c["model_size"], device=c["device"],
                             batch_size=c.get("batch_size", 16))
    if method == "birefnet":
        from .birefnet import BiRefNetMatte
        c = cfg["annotate"]["birefnet"]
        return BiRefNetMatte(weights=c["hf_repo"], device=c["device"],
                             padding=c["padding"], threshold=c.get("threshold", 0.5))
    if method == "exg":
        from .greenness import ExGAnnotator
        c = cfg["annotate"]["exg"]
        return ExGAnnotator(padding=c["padding"], close_ksize=c["close_ksize"],
                            green_floor=c["green_floor"])
    if method == "hybrid":
        from .greenness import ExGAnnotator, HybridAnnotator
        from .sam2 import SAM2Annotator
        sc, ec = cfg["annotate"]["sam2"], cfg["annotate"]["exg"]
        sam2 = SAM2Annotator(model_size=sc["model_size"], device=sc["device"],
                             batch_size=sc.get("batch_size", 16))
        exg = ExGAnnotator(padding=ec["padding"], close_ksize=ec["close_ksize"],
                           green_floor=ec["green_floor"])
        return HybridAnnotator(sam2, exg)
    raise SystemExit(f"Unknown method {method!r} (use sam2 | birefnet | exg | hybrid | mock)")


def _completed_records(out_dir: Path, uid: str, save_overlays: bool, debug: bool):
    """If every output for this image already exists (and the sidecar is intact),
    return its instance records so a resumed run can fold them into the QC table;
    otherwise None (reprocess). Lets a crashed run pick up where it stopped."""
    js = out_dir / f"{uid}.json"
    if not js.exists() or not (out_dir / f"{uid}.png").exists():
        return None
    if save_overlays and not (out_dir / "_overlays" / f"{uid}.jpg").exists():
        return None
    if debug and not (out_dir / "_debug" / f"{uid}.png").exists():
        return None
    try:
        return json.loads(js.read_text(encoding="utf-8")).get("instances", [])
    except (json.JSONDecodeError, OSError):
        return None  # partial/corrupt write -> redo this one


def run(method: str, split: str, limit: int | None, segmenter=None,
        save_overlays: bool | None = None, debug: bool = False,
        resume: bool = False) -> dict:
    cfg = load_config("base.yaml")
    if save_overlays is None:
        save_overlays = cfg["annotate"].get("save_overlays", True)
    if segmenter is None:
        segmenter = _build_segmenter(method, cfg)

    _report_device(method, getattr(segmenter, "device", None))

    img_dir = paths.REAL / "images" / split
    lbl_dir = paths.REAL / "labels" / split
    out_dir = paths.MASKS / method / split
    if not img_dir.exists():
        raise SystemExit(f"No images at {img_dir} — run Stage A (prepare_data) first.")

    images = sorted(p for p in img_dir.iterdir() if p.is_file())
    if limit:
        images = images[:limit]

    iou_by_class: dict[str, list[float]] = defaultdict(list)
    n_images = n_instances = n_skipped = 0

    for img_path in tqdm(images, desc=f"{method}:{split}", unit="img"):
        uid = img_path.stem

        if resume:
            done = _completed_records(out_dir, uid, save_overlays, debug)
            if done is not None:
                for r in done:                       # keep the QC table complete
                    iou_by_class[r["class"]].append(float(r["iou"]))
                n_images += 1
                n_instances += len(done)
                n_skipped += 1
                continue

        image = _load_image(img_path)
        h, w = image.shape[:2]
        boxes = _read_yolo_boxes(lbl_dir / f"{uid}.txt", w, h)
        if not boxes:
            continue

        class_names = [c for c, _ in boxes]
        gt_xyxy = np.stack([b for _, b in boxes])
        instance_masks = segmenter.masks_from_boxes(image, gt_xyxy)

        records = []
        for i, (cls_name, gt_box, m) in enumerate(zip(class_names, gt_xyxy, instance_masks), start=1):
            mbox = mask_utils.mask_to_bbox(m)
            iou = _iou(gt_box, np.array(mbox, dtype=np.float32)) if mbox else 0.0
            iou_by_class[cls_name].append(iou)
            records.append({
                "id": i, "class": cls_name,
                "gt_box": [round(float(x), 1) for x in gt_box],
                "mask_box": list(mbox) if mbox else None,
                "iou": round(float(iou), 4), "area": int((m > 0).sum()),
            })

        _save_label_map(instance_masks, (h, w), out_dir / f"{uid}.png")
        (out_dir / f"{uid}.json").write_text(
            json.dumps({"width": w, "height": h, "instances": records}, indent=2),
            encoding="utf-8",
        )
        if save_overlays:
            _save_overlay(image, instance_masks, class_names, out_dir / "_overlays" / f"{uid}.jpg")
        if debug:
            _save_debug(segmenter, instance_masks, (h, w), out_dir / "_debug" / f"{uid}.png")

        n_images += 1
        n_instances += len(records)

    return _summarise(method, split, iou_by_class, n_images, n_instances, n_skipped)


def _summarise(method, split, iou_by_class, n_images, n_instances, n_skipped=0) -> dict:
    per_class = {
        c: {"n": len(v), "mean_iou": round(float(np.mean(v)), 4)}
        for c, v in sorted(iou_by_class.items())
    }
    all_iou = [x for v in iou_by_class.values() for x in v]
    summary = {
        "method": method, "split": split, "images": n_images,
        "skipped_resumed": n_skipped, "instances": n_instances,
        "mean_iou": round(float(np.mean(all_iou)), 4) if all_iou else 0.0,
        "per_class": per_class,
    }
    paths.ensure_dirs()
    out = paths.TABLES / f"mask_qc_{method}_{split}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nMasks -> {paths.MASKS / method / split}\nQC -> {out}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 0: box-prompted masks + IoU QC.")
    ap.add_argument("--method", default="sam2",
                    choices=["sam2", "birefnet", "exg", "hybrid", "mock"])
    ap.add_argument("--split", default="train", choices=list(paths.SPLITS))
    ap.add_argument("--limit", type=int, default=None, help="process only the first N images")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="save raw mask (BiRefNet: soft pre-threshold matte) to _debug/")
    ap.add_argument("--resume", action="store_true",
                    help="skip images whose outputs already exist (resume a crashed run)")
    args = ap.parse_args()
    seg = _MockSegmenter() if args.method == "mock" else None
    run(args.method, args.split, args.limit, segmenter=seg,
        save_overlays=not args.no_overlays, debug=args.debug, resume=args.resume)
