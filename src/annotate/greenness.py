"""Greenness (ExGR) annotation route — for thin green weeds.

The deep matters (SAM2, BiRefNet) lock onto the dense centre of grass-like
monocots (CYPRO/SETVE/ECHCG) and drop the thin green strands. A classical
vegetation index recovers them: within a GT box, green = plant, brown = soil, so
an Excess-Green threshold captures the strands almost for free. This is the same
family of method the CSIC baseline used (ExGR + OpenCV), reached for exactly
where the neural models underperform.

Provides two segmenters with the standard `masks_from_boxes(image, boxes)`
interface used by run_box_masks.py:
  * ExGAnnotator     — greenness only.
  * HybridAnnotator  — SAM2 (solid centre) UNION ExG (strands), box-clipped.

Pure NumPy/OpenCV → CPU, fast, no model download.
"""

from __future__ import annotations

import numpy as np


def _excess_green(crop_rgb: np.ndarray) -> np.ndarray:
    """ExG on normalised chromatic coordinates (lighting-robust). Green > 0."""
    c = crop_rgb.astype(np.float32)
    s = c.sum(axis=2) + 1e-6
    rn, gn, bn = c[..., 0] / s, c[..., 1] / s, c[..., 2] / s
    return 2.0 * gn - rn - bn          # ExG; ExGR adds -(1.4*rn - gn), see below


def _excess_green_red(crop_rgb: np.ndarray) -> np.ndarray:
    exg = _excess_green(crop_rgb)
    c = crop_rgb.astype(np.float32)
    s = c.sum(axis=2) + 1e-6
    rn, gn = c[..., 0] / s, c[..., 1] / s
    exr = 1.4 * rn - gn
    return exg - exr                   # ExGR (Camargo-Neto)


def green_mask_in_box(
    image: np.ndarray,
    box_xyxy,
    padding: float = 0.15,
    close_ksize: int = 5,
    green_floor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary full-image mask, soft full-image greenness) for one box.

    ExGR + Otsu inside the padded box, AND-ed with a positive-greenness floor so
    an all-soil or all-plant crop can't make Otsu pick the wrong mode. Then a
    morphological close to bridge strand gaps, keeping only components that touch
    the inner GT box (so a neighbour's leaf in the padding margin is dropped).
    """
    import cv2

    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * padding, bh * padding
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))

    full = np.zeros((h, w), dtype=np.uint8)
    soft_full = np.zeros((h, w), dtype=np.float32)
    if cx2 <= cx1 or cy2 <= cy1:
        return full, soft_full

    crop = image[cy1:cy2, cx1:cx2]
    exgr = _excess_green_red(crop)
    exg = _excess_green(crop)

    u = cv2.normalize(exgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, otsu = cv2.threshold(u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    green = (otsu > 0) & (exg > green_floor)

    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0

    green = _keep_components_touching_inner_box(
        green.astype(np.uint8), crop.shape[:2], (x1, y1, x2, y2), (cx1, cy1)
    )

    full[cy1:cy2, cx1:cx2] = green
    soft_full[cy1:cy2, cx1:cx2] = np.clip(exg, 0.0, 1.0)
    return full, soft_full


def _keep_components_touching_inner_box(green, crop_shape, box, crop_origin):
    import cv2

    ch, cw = crop_shape
    x1, y1, x2, y2 = box
    ox, oy = crop_origin
    ix1, iy1 = max(0, int(x1) - ox), max(0, int(y1) - oy)
    ix2, iy2 = min(cw, int(x2) - ox), min(ch, int(y2) - oy)
    if ix2 <= ix1 or iy2 <= iy1:
        return green
    n, labels = cv2.connectedComponents(green)
    if n <= 1:
        return green
    inner = labels[iy1:iy2, ix1:ix2]
    keep_ids = set(int(i) for i in np.unique(inner) if i != 0)
    if not keep_ids:
        return green
    return np.isin(labels, list(keep_ids)).astype(np.uint8)


class ExGAnnotator:
    """Greenness-only segmenter (CPU)."""

    def __init__(self, padding: float = 0.15, close_ksize: int = 5, green_floor: float = 0.0):
        self.padding = padding
        self.close_ksize = close_ksize
        self.green_floor = green_floor
        self.device = "cpu"                 # classical — for the device banner
        self.last_soft: np.ndarray | None = None

    def masks_from_boxes(self, image: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray]:
        h, w = image.shape[:2]
        out: list[np.ndarray] = []
        soft_full = np.zeros((h, w), dtype=np.float32)
        for box in np.asarray(boxes_xyxy, dtype=np.float32):
            m, soft = green_mask_in_box(image, box, self.padding, self.close_ksize, self.green_floor)
            np.maximum(soft_full, soft, out=soft_full)
            out.append(m)
        self.last_soft = soft_full
        return out


class HybridAnnotator:
    """SAM2 (solid centre) UNION ExG (thin strands), box-clipped."""

    def __init__(self, sam2, exg: ExGAnnotator):
        self.sam2 = sam2
        self.exg = exg
        self.device = getattr(sam2, "device", "cuda")
        self.last_soft: np.ndarray | None = None

    def masks_from_boxes(self, image: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray]:
        sam_masks = self.sam2.masks_from_boxes(image, boxes_xyxy)
        green_masks = self.exg.masks_from_boxes(image, boxes_xyxy)
        self.last_soft = self.exg.last_soft          # show greenness field in --debug
        return [np.maximum(a, b).astype(np.uint8) for a, b in zip(sam_masks, green_masks)]
