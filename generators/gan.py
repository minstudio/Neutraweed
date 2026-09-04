"""Paradigm baseline: modern conditional GAN (CLAUDE.md §3).

Re-confirms "diffusion > GAN" with current models. Trained per class (unlike the
single conditioned diffusion model). Candidate backbones: StyleGAN2-ADA (strong
on small data) or a shape-style conditional GAN (ref 2407.14119), CA-GAN
(PMC10981930).

Per-class training -> `generate(classes=[c])` loads the matching checkpoint.
"""

from __future__ import annotations

from pathlib import Path

from .base import Generator, GenSample


class ConditionalGAN(Generator):
    name = "gan"

    def finetune(self, train_yaml: Path) -> None:
        self._guard_train_only(train_yaml)
        self.prepare_dirs()
        # TODO: train one conditional GAN per class on real train crops.
        raise NotImplementedError(
            "Conditional GAN training not implemented. Plug in StyleGAN2-ADA "
            "(per class) or a shape-style conditional GAN."
        )

    def generate(self, n: int, classes: list[str] | None = None) -> list[GenSample]:
        self.prepare_dirs()
        # TODO: sample per-class; annotate via BiRefNet (isolated single weed).
        raise NotImplementedError("Conditional GAN sampling not implemented.")
