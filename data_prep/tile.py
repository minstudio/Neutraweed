"""Image tiling for Stage A (CLAUDE.md §2.1 resolution note; follows Gómez 2025).

Full ~3400x3700 frames downscaled to the detector's imgsz make weeds tens of px
— too small, and the held-out test set is only ~27 images. Cutting each frame
into overlapping tiles makes weeds ~2x larger and turns the test set into
hundreds of tiles (statistical power). Overlap ensures a weed clipped at one
tile's edge appears whole in a neighbour; edge-clipped boxes are dropped.

Pure geometry — no I/O — so it's unit-testable.
"""

from __future__ import annotations


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Sliding-window start positions covering [0, length); last clamped to edge."""
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


def tile_boxes(
    width: int,
    height: int,
    boxes: list[tuple[int, float, float, float, float]],
    tile: int,
    stride: int,
    keep_frac: float,
):
    """Split a frame's boxes into per-tile boxes.

    `boxes` are (cls_id, x1, y1, x2, y2) in pixels on the full frame. Yields
    (tx, ty, tile_w, tile_h, kept) where `kept` are (cls_id, x1, y1, x2, y2) in
    TILE pixel coords. A box is kept for a tile only if >= keep_frac of its area
    lies inside (else it's an edge sliver — kept whole by an overlapping tile).
    """
    for ty in tile_origins(height, tile, stride):
        th = min(tile, height - ty)
        for tx in tile_origins(width, tile, stride):
            tw = min(tile, width - tx)
            kept = []
            for cid, bx1, by1, bx2, by2 in boxes:
                ix1, iy1 = max(bx1, tx), max(by1, ty)
                ix2, iy2 = min(bx2, tx + tw), min(by2, ty + th)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                area = (bx2 - bx1) * (by2 - by1)
                if area <= 0 or inter / area < keep_frac:
                    continue
                kept.append((cid, ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty))
            yield tx, ty, tw, th, kept
