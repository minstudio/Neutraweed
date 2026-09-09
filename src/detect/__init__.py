"""Stage E — detector training under a FROZEN protocol.

Every run uses identical hyperparameters (epochs, optimizer, lr, img-size, batch,
COCO-pretrained init) and the same real val/test; ONLY the training-data
composition (the dataset.yaml from Stage D) and the seed change. The frozen
settings live in configs/base.yaml `detector:` and are passed through verbatim.
"""

from . import train_rtdetr, train_yolo

__all__ = ["train_yolo", "train_rtdetr"]
