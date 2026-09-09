"""SAM2 segmentation (Stage C).

Two modes:
  * box-prompted  -> Phase 0 keystone: our data has GT *boxes* but no masks.
    Prompting SAM2 with each GT box turns the real train set into instance masks
    cheaply and reliably (box-prompted is SAM2's strongest mode). Those masks
    feed ControlNet conditioning and cut-and-composite, and are validated by
    mask->box IoU (src/annotate/run_box_masks.py).
  * automatic     -> Phase 2+: dense synthetic scenes with no prompts.

Heavy deps (`sam2`, torch CUDA, the checkpoint) load lazily. On the 6 GB laptop
GPU use model_size='small'/'tiny'; use 'large' on the cluster.
"""

from __future__ import annotations

import numpy as np

from . import masks


class SAM2Annotator:
    """Box-prompted (and automatic) SAM2 wrapper. Lazy, device-configurable."""

    HF_REPO = {
        "tiny": "facebook/sam2-hiera-tiny",
        "small": "facebook/sam2-hiera-small",
        "base_plus": "facebook/sam2-hiera-base-plus",
        "large": "facebook/sam2-hiera-large",
    }

    def __init__(self, model_size: str = "small", device: str = "cuda", batch_size: int = 16):
        self.model_size = model_size
        self.device = device
        self.batch_size = batch_size      # boxes per predict() call — bounds VRAM
        self._predictor = None
        self._auto = None

    # ---- loading -------------------------------------------------------------
    def _load_predictor(self):
        if self._predictor is None:
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as exc:
                raise ImportError(
                    "SAM2 not installed. Install with:\n"
                    "  pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
                    "Checkpoints download automatically from the Hugging Face repo "
                    f"({self.HF_REPO[self.model_size]})."
                ) from exc
            self._predictor = SAM2ImagePredictor.from_pretrained(
                self.HF_REPO[self.model_size], device=self.device
            )
        return self._predictor

    # ---- box-prompted (Phase 0) ---------------------------------------------
    def masks_from_boxes(
        self, image: np.ndarray, boxes_xyxy: np.ndarray
    ) -> list[np.ndarray]:
        """Return one binary mask per input box (pixel coords, Nx4 xyxy).

        Boxes are predicted in chunks of `batch_size`: dense images can carry
        hundreds of weeds, and SAM2 upsamples every mask to the full (multi-MP)
        resolution at once, so a single all-boxes call OOMs small GPUs. Chunking
        bounds peak VRAM regardless of weed count. multimask_output=False -> one
        best mask per box. The image embedding is computed once via set_image and
        reused across chunks.
        """
        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        if len(boxes) == 0:
            return []
        predictor = self._load_predictor()
        import torch

        predictor.set_image(image)
        out: list[np.ndarray] = []
        bs = max(1, self.batch_size)
        with torch.inference_mode():
            for s in range(0, len(boxes), bs):
                m, _scores, _ = predictor.predict(
                    box=boxes[s:s + bs], multimask_output=False
                )
                m = np.asarray(m)
                if m.ndim == 4:        # (k,1,H,W) -> (k,H,W)
                    m = m[:, 0]
                elif m.ndim == 2:      # (H,W) single -> (1,H,W)
                    m = m[None]
                out.extend((mi > 0).astype(np.uint8) for mi in m)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()   # release the chunk buffers between images
        return out

    # ---- automatic (Phase 2+) -----------------------------------------------
    def _load_auto(self):
        if self._auto is None:
            try:
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                from sam2.build_sam import build_sam2_hf
            except ImportError as exc:
                raise ImportError("SAM2 not installed; see masks_from_boxes() for the install line.") from exc
            model = build_sam2_hf(self.HF_REPO[self.model_size], device=self.device)
            self._auto = SAM2AutomaticMaskGenerator(model)
        return self._auto

    def segment(self, image: np.ndarray) -> list[np.ndarray]:
        gen = self._load_auto()
        return [m["segmentation"].astype(np.uint8) for m in gen.generate(image)]

    def annotate(self, image: np.ndarray, classify, out_label) -> int:
        """Automatic scene annotation: segment, classify each instance, write YOLO."""
        h, w = image.shape[:2]
        instances = [(classify(m, image), m) for m in self.segment(image)]
        instances = [(c, m) for c, m in instances if c is not None]
        return masks.masks_to_yolo(instances, (w, h), out_label)
