"""Stage D — assemble real / synthetic / hybrid training datasets.

Datasets are expressed as Ultralytics image-list files (train.txt/val.txt/test.txt)
so no image bytes are duplicated: each line is an absolute image path, and YOLO
finds its label by swapping `images/`->`labels/`. This works for both the real
tree (data/real/...) and any synthetic pool (data/synthetic/<gen>/...).

Hard invariant: val.txt and test.txt ALWAYS point at the real split only.
"""

from . import build_hybrid, matched_size

__all__ = ["build_hybrid", "matched_size"]
