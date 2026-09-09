"""ControlNet route — label known by construction (Stage C).

When SD3.5 generated an image from a conditioning mask, that mask already tells
us where each object is and which class it is. No segmentation needed: we just
threshold the (possibly multi-class) conditioning mask and emit boxes. This is
the cheapest, most reliable route and the reason SD3.5+ControlNet is primary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import masks


def annotate_from_control_mask(
    control_mask: np.ndarray,
    class_for_value: dict[int, str],
    image_size: tuple[int, int],
    out_label: Path,
) -> int:
    """Convert a label-map conditioning mask into YOLO boxes.

    `control_mask` is an integer HxW map where each pixel value identifies a
    class instance region; `class_for_value` maps value -> EPPO code. One box per
    connected component is recovered via the shared mask util.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenCV required for connected-component splitting.") from exc

    instances: list[tuple[str, np.ndarray]] = []
    for value, cls_name in class_for_value.items():
        binary = (control_mask == value).astype(np.uint8)
        n, comp = cv2.connectedComponents(binary)
        for label in range(1, n):
            instances.append((cls_name, (comp == label).astype(np.uint8)))
    return masks.masks_to_yolo(instances, image_size, out_label)
