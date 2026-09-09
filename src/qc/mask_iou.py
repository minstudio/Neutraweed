"""Validate auto-annotations against a hand-labelled subset.

"Report mask IoU for auto-annotations; don't trust segmenters blindly." This
computes box IoU between predicted YOLO labels and ground-truth YOLO labels over
a small hand-checked subset, per class and overall. Pure geometry — fully
implemented and testable.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_yolo(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            c, cx, cy, w, h = parts
            rows.append((int(c), float(cx), float(cy), float(w), float(h)))
    return rows


def _to_xyxy(box):
    _, cx, cy, w, h = box
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _iou(a, b) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def match_iou(pred_label: Path, gt_label: Path, iou_thresh: float = 0.5):
    """Greedy class-aware matching; returns mean IoU of matched pairs + counts."""
    preds = _read_yolo(pred_label)
    gts = _read_yolo(gt_label)
    matched_ious: list[float] = []
    used = set()
    for g in gts:
        best_iou, best_j = 0.0, -1
        for j, p in enumerate(preds):
            if j in used or p[0] != g[0]:
                continue
            i = _iou(_to_xyxy(g), _to_xyxy(p))
            if i > best_iou:
                best_iou, best_j = i, j
        if best_j >= 0 and best_iou >= iou_thresh:
            used.add(best_j)
            matched_ious.append(best_iou)
    return {
        "n_gt": len(gts),
        "n_pred": len(preds),
        "n_matched": len(matched_ious),
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def evaluate_subset(pred_dir: Path, gt_dir: Path, iou_thresh: float = 0.5) -> dict:
    """Aggregate match_iou over every GT label file in a hand-labelled subset."""
    agg = defaultdict(float)
    ious: list[float] = []
    n_files = 0
    for gt in sorted(gt_dir.glob("*.txt")):
        r = match_iou(pred_dir / gt.name, gt, iou_thresh)
        agg["n_gt"] += r["n_gt"]
        agg["n_pred"] += r["n_pred"]
        agg["n_matched"] += r["n_matched"]
        if r["n_matched"]:
            ious.append(r["mean_iou"])
        n_files += 1
    recall = agg["n_matched"] / agg["n_gt"] if agg["n_gt"] else 0.0
    return {
        "files": n_files,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "box_recall@iou": recall,
        **{k: int(v) for k, v in agg.items()},
    }
