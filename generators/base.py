"""Common generator interface.

Every generator (SD3.5/ControlNet, FLUX, GAN) implements this so downstream code
(QC, dataset assembly) never special-cases a backend. A generator is fine-tuned
on the REAL TRAIN split only (hard invariant — it may never see val/test), then
emits images plus whatever conditioning it used, so labels can be recovered in
Stage C.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..common import paths


@dataclass
class GenSample:
    """One synthesized image and everything needed to auto-annotate it later."""
    image_path: Path
    prompt: str
    class_name: str | None = None        # if single-class conditioned
    control_image: Path | None = None    # ControlNet input (mask/edge/layout)
    known_mask: Path | None = None       # set on ControlNet route -> free label
    meta: dict = field(default_factory=dict)


class Generator(ABC):
    """Fine-tune on real train, then synthesize. Backends fill in the two hooks."""

    name: str = "base"

    def __init__(self, config: dict):
        self.config = config
        self.out_dir = paths.SYNTHETIC / self.name
        self.weights_dir = self.out_dir / "weights"

    # ---- Stage B -------------------------------------------------------------
    @abstractmethod
    def finetune(self, train_yaml: Path) -> None:
        """LoRA/DreamBooth (diffusion) or full train (GAN) on the REAL train split.

        Implementations MUST read only train images. Passing val/test here is a
        protocol violation and should raise.
        """

    # ---- Stage C -------------------------------------------------------------
    @abstractmethod
    def generate(self, n: int, classes: list[str] | None = None) -> list[GenSample]:
        """Produce `n` samples, optionally conditioned on the given class list."""

    # ---- shared --------------------------------------------------------------
    def _guard_train_only(self, train_yaml: Path) -> None:
        if any(tok in str(train_yaml).lower() for tok in ("val", "test")):
            raise ValueError(
                "Generators may only train on the real TRAIN split "
                f"(got {train_yaml}). val/test must stay unseen."
            )

    def prepare_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "images").mkdir(exist_ok=True)
        (self.out_dir / "control").mkdir(exist_ok=True)
        (self.out_dir / "labels").mkdir(exist_ok=True)
