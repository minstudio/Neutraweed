"""Stage C — auto-annotation routes for synthetic images (CLAUDE.md §4).

Route selection by layout:
  * controlnet_mask : label known by construction (ControlNet conditioning mask).
  * birefnet        : isolated single weed -> matte -> box (crisp on thin stems).
  * sam2            : multi-weed scene -> instance masks.
  * composite       : cut REAL weeds (segment real, not fake) -> paste on varied
                      backgrounds -> perfect labels.

All routes ultimately emit YOLO label files via `masks.masks_to_yolo`.
"""

from . import birefnet, composite, controlnet_mask, masks, sam2

__all__ = ["birefnet", "composite", "controlnet_mask", "masks", "sam2"]
