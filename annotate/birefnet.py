"""BiRefNet route — crisp mattes for thin-stemmed weeds (CLAUDE.md §4 Stage C).

SAM2's box prompt locks onto the dense centre of grass-like monocots (CYPRO,
SETVE, ECHCG) and drops the thin strands. BiRefNet produces high-resolution
mattes that trace those strands. We have GT *boxes*, so we run BiRefNet
per box: crop the box (with context padding), matte the single plant inside,
then place that matte back into full-image coordinates.

Same `masks_from_boxes(image, boxes)` interface as SAM2Annotator, so the Phase 0
driver (run_box_masks.py) can swap methods with no other change.

Install: BiRefNet ships as a Hugging Face model with custom code.
  pip install transformers timm kornia einops
  # weights auto-download from ZhengPeng7/BiRefNet (trust_remote_code=True)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import masks


# ImageNet normalisation, BiRefNet's expected 1024x1024 input.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_INPUT = 1024


class BiRefNetMatte:
    def __init__(
        self,
        weights: str = "ZhengPeng7/BiRefNet",
        device: str = "cuda",
        padding: float = 0.15,
        threshold: float = 0.5,
    ):
        self.weights = weights
        self.device = device
        self.padding = padding         # fraction of box size added as context
        self.threshold = threshold
        self._model = None
        self._tf = None
        # soft (pre-threshold) full-image matte from the most recent call, for --debug
        self.last_soft: np.ndarray | None = None

    def _load(self):
        if self._model is None:
            try:
                import torch  # noqa: F401
                from torchvision import transforms
                from transformers import AutoModelForImageSegmentation
            except ImportError as exc:
                raise ImportError(
                    "BiRefNet needs: pip install transformers torchvision timm kornia einops"
                ) from exc
            self._model = AutoModelForImageSegmentation.from_pretrained(
                self.weights, trust_remote_code=True
            ).to(self.device).eval()
            self._tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((_INPUT, _INPUT), antialias=True),
                transforms.Normalize(_MEAN, _STD),
            ])
        return self._model

    def _matte_crop(self, crop: np.ndarray) -> np.ndarray:
        """Return a [0,1] matte at the crop's native resolution."""
        import torch
        from PIL import Image
        import torch.nn.functional as F

        model = self._load()
        ch, cw = crop.shape[:2]
        x = self._tf(Image.fromarray(crop)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = model(x)[-1].sigmoid().cpu()          # (1,1,1024,1024)
        matte = F.interpolate(pred, size=(ch, cw), mode="bilinear", align_corners=False)
        return matte[0, 0].numpy()

    def masks_from_boxes(self, image: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray]:
        h, w = image.shape[:2]
        out: list[np.ndarray] = []
        soft_full = np.zeros((h, w), dtype=np.float32)   # for --debug
        for x1, y1, x2, y2 in np.asarray(boxes_xyxy, dtype=np.float32):
            bw, bh = x2 - x1, y2 - y1
            px, py = bw * self.padding, bh * self.padding
            cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
            cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
            full = np.zeros((h, w), dtype=np.uint8)
            if cx2 <= cx1 or cy2 <= cy1:
                out.append(full)
                continue
            matte = self._matte_crop(image[cy1:cy2, cx1:cx2])
            full[cy1:cy2, cx1:cx2] = (matte >= self.threshold).astype(np.uint8)
            soft_full[cy1:cy2, cx1:cx2] = np.maximum(soft_full[cy1:cy2, cx1:cx2], matte)
            out.append(full)
        self.last_soft = soft_full
        return out

    # convenience single-image API (isolated weed, class known from prompt)
    def annotate(self, image: np.ndarray, cls_name: str, out_label: Path) -> int:
        matte = self._matte_crop(image)
        mask = (matte >= self.threshold).astype(np.uint8)
        h, w = mask.shape[:2]
        return masks.masks_to_yolo([(cls_name, mask)], (w, h), out_label)
