"""Stage B/C — synthetic image generators.

Three arms (CLAUDE.md §3): SD 3.5 + ControlNet (primary), FLUX.1-dev (photoreal),
and a modern conditional GAN (paradigm baseline). Each exposes the same
`finetune()` / `generate()` interface from `base.Generator` so the dataset and
QC stages stay generator-agnostic.

Heavy deps (diffusers, peft, the GAN repo) are imported lazily inside each
backend so this package imports cleanly on a machine without them.
"""

from .base import Generator, GenSample

__all__ = ["Generator", "GenSample"]
