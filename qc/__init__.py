"""Stage C QC (CLAUDE.md §4): FID + CLIP-IQA gating, dedup, and mask-IoU
validation of auto-annotations before any synthetic image enters training."""

from . import clip_iqa, dedupe, fid, mask_iou

__all__ = ["fid", "clip_iqa", "mask_iou", "dedupe"]
