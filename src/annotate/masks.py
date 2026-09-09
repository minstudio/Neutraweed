"""Mask -> YOLO box conversion, shared by every annotation route.

This is the one piece of Stage C that is the same regardless of which segmenter
produced the mask, so it is implemented for real (numpy/OpenCV) and unit-testable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import classes


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight (xmin, ymin, xmax, ymax) around the nonzero region, or None if empty."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_to_yolo_line(cls_name: str, box: tuple[int, int, int, int], w: int, h: int) -> str:
    xmin, ymin, xmax, ymax = box
    cid = classes.class_id(cls_name)
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def masks_to_yolo(
    instances: list[tuple[str, np.ndarray]],
    image_size: tuple[int, int],
    out_label: Path,
    min_area: int = 16,
) -> int:
    """Write a YOLO label file from (class_name, binary_mask) instances.

    Returns the number of boxes written. `image_size` is (width, height).
    """
    w, h = image_size
    lines: list[str] = []
    for cls_name, mask in instances:
        if int((mask > 0).sum()) < min_area:
            continue
        box = mask_to_bbox(mask)
        if box is None:
            continue
        lines.append(bbox_to_yolo_line(cls_name, box, w, h))
    out_label.parent.mkdir(parents=True, exist_ok=True)
    out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)
