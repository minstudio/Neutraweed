"""Photorealism arm: FLUX.1-dev + LoRA (CLAUDE.md §3).

Tests whether a newer architecture yields *more useful* training images than
SD3.5. FLUX has no native ControlNet of the same maturity, so images from this
arm typically go through the SAM2 / BiRefNet annotation routes (Stage C) rather
than the free-label ControlNet route.

Full FLUX fine-tunes want an A100/H100 (CLAUDE.md §7); LoRA fits on 24 GB.
"""

from __future__ import annotations

from pathlib import Path

from .base import Generator, GenSample


class FluxLoRA(Generator):
    name = "flux"

    DEFAULT_MODEL = "black-forest-labs/FLUX.1-dev"

    def finetune(self, train_yaml: Path) -> None:
        self._guard_train_only(train_yaml)
        self.prepare_dirs()
        # TODO: peft LoRA fine-tune FLUX.1-dev on real train images.
        raise NotImplementedError(
            "FLUX.1-dev LoRA fine-tune not implemented. Use diffusers FluxPipeline + peft."
        )

    def generate(self, n: int, classes: list[str] | None = None) -> list[GenSample]:
        self.prepare_dirs()
        # TODO: sample images; annotate downstream via SAM2/BiRefNet (no free mask).
        raise NotImplementedError(
            "FLUX sampling not implemented. Samples carry no known_mask; route "
            "them through src/annotate (SAM2 for scenes, BiRefNet for isolated)."
        )
