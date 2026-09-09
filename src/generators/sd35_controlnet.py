"""Primary generator: Stable Diffusion 3.5 + ControlNet (Stage B/C).

The label is known by construction: we author the conditioning layout
(src/generators/conditioning.py), so each synthetic image ships with exact YOLO
boxes — no segmentation of generated pixels.

Two phases:
  * finetune() — prepares the on-disk data needed for the heavy training and
    documents the exact diffusers commands (LoRA domain fine-tune + optional seg
    ControlNet). Those trainers are not reimplemented here; they are handed
    prepared data.
  * generate() — fully implemented: load SD3.5 (+ ControlNet, + optional LoRA),
    sample layouts, and write image + YOLO label + conditioning to
    data/synthetic/sd35/. This is what feeds Stage D (build_hybrid).

Heavy deps (diffusers, torch CUDA) import lazily. Needs a big GPU (A100/cluster);
will not fit the 6 GB laptop — run it on the cluster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import paths
from .base import Generator, GenSample
from . import conditioning


class SD35ControlNet(Generator):
    name = "sd35"

    DEFAULT_BASE = "stabilityai/stable-diffusion-3.5-medium"
    # No official SD3.5 *segmentation* ControlNet exists yet. Two supported routes
    # (see the guide): train a seg ControlNet on exported pairs (control='seg'),
    # or use Stability's pretrained Canny ControlNet (control='canny').
    DEFAULT_CONTROLNET = {
        "seg": "data/synthetic/sd35/weights/controlnet",   # trained locally
        "canny": "stabilityai/stable-diffusion-3.5-large-controlnet-canny",
    }

    def __init__(self, config: dict | None = None):
        super().__init__(config or {})
        g = self.config.get("generator", {}) if self.config else {}
        self.base = g.get("base", self.DEFAULT_BASE)
        # Layout-mask conditioning -> free labels. 'seg' is the
        # prescribed route; the trained ControlNet lives under weights/controlnet.
        self.control_kind = g.get("control_kind", "seg")
        self.controlnet = g.get("controlnet", self.DEFAULT_CONTROLNET[self.control_kind])
        self.lora = g.get("lora", str(self.weights_dir / "lora"))
        self.device = g.get("device", "cuda")
        self.steps = g.get("steps", 28)
        self.guidance = g.get("guidance", 4.5)
        # How hard SD3.5 must obey the layout. 1.0 = default (weak with an
        # under-trained ControlNet -> weeds drift off the boxes). Raise toward
        # 1.5-3.0 to force weeds onto the conditioned positions.
        self.cn_scale = g.get("controlnet_scale", 1.0)
        self.image_size = tuple(g.get("image_size", (1024, 1024)))
        self._pipe = None

    # ---- Stage B -------------------------------------------------------------
    def finetune(self, train_yaml: Path) -> None:
        """Prepare data for the heavy training and print the exact commands.

        The trainers are the diffusers examples and belong on the cluster. This
        exports what they need and prints how to launch them.
        """
        self._guard_train_only(train_yaml)
        self.prepare_dirs()

        # 1) ControlNet training pairs (only needed for control_kind='seg').
        if self.control_kind == "seg":
            pairs = conditioning.export_controlnet_pairs(split="train", control_kind="seg")
            print(f"[Stage B] ControlNet pairs ready: {pairs}")
            print("  Train a seg ControlNet with diffusers examples/controlnet/"
                  "train_controlnet_sd3.py")
        else:
            print("[Stage B] control_kind='canny' -> using a pretrained ControlNet; "
                  "no ControlNet training needed.")

        # 2) Instance bank for the layout sampler (needed by generate()).
        bank = conditioning.InstanceBank.build(paths.MASKS / "sam2" / "train")
        bank_path = self.out_dir / "instance_bank.npz"
        bank.save(bank_path)
        print(f"[Stage B] Instance bank: {bank.counts()} -> {bank_path}")
        print("  Domain LoRA: fine-tune SD3.5 with examples/dreambooth/"
              "train_dreambooth_lora_sd3.py on real train crops (see the guide).")

    # ---- Stage C -------------------------------------------------------------
    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import (
                SD3ControlNetModel,
                StableDiffusion3ControlNetPipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "Install diffusers stack: pip install 'diffusers>=0.31' transformers "
                "accelerate peft sentencepiece protobuf"
            ) from exc

        dtype = torch.float16
        controlnet = SD3ControlNetModel.from_pretrained(self.controlnet, torch_dtype=dtype)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            self.base, controlnet=controlnet, torch_dtype=dtype
        )
        lora_dir = Path(self.lora)
        if lora_dir.exists():
            pipe.load_lora_weights(str(lora_dir))
            print(f"[Stage C] loaded domain LoRA from {lora_dir}")
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def generate(self, n: int, classes: list[str] | None = None,
                 seed: int = 0, prompt: str | None = None) -> list[GenSample]:
        from PIL import Image

        self.prepare_dirs()
        bank_path = self.out_dir / "instance_bank.npz"
        if not bank_path.exists():
            raise SystemExit("No instance bank — run finetune() (Stage B) first.")
        bank = conditioning.InstanceBank.load(bank_path)
        sampler = conditioning.LayoutSampler(bank, image_size=self.image_size, seed=seed)

        pipe = self._load_pipe()
        import torch

        base_prompt = prompt or conditioning._prompt_for([c for c in bank.shapes if bank.shapes[c]])
        samples: list[GenSample] = []
        gen = torch.Generator(device=self.device)

        for i in range(n):
            layout = sampler.sample()
            cond = conditioning.control_image(layout.semantic, self.control_kind)
            gen.manual_seed(seed * 100000 + i)
            image = pipe(
                prompt=base_prompt,
                control_image=Image.fromarray(cond),
                controlnet_conditioning_scale=self.cn_scale,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                height=self.image_size[1],
                width=self.image_size[0],
                generator=gen,
            ).images[0]

            stem = f"sd35_{seed:02d}_{i:06d}"
            img_path = self.out_dir / "images" / f"{stem}.png"
            lbl_path = self.out_dir / "labels" / f"{stem}.txt"
            ctl_path = self.out_dir / "control" / f"{stem}.png"
            image.save(img_path)
            Image.fromarray(cond).save(ctl_path)
            w, h = self.image_size
            lines = conditioning.instances_to_yolo(layout.instances, w, h)
            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            samples.append(GenSample(
                image_path=img_path, prompt=base_prompt,
                control_image=ctl_path, meta={"n_instances": len(layout.instances)},
            ))
        print(f"[Stage C] generated {len(samples)} images -> {self.out_dir / 'images'}")
        return samples
