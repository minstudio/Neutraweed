"""Tiny YAML config loader with shallow include support.

Experiment configs in configs/ describe a single run (generator, ratio,
detector, seed). A config may set `extends: <relative-path>` to inherit from a
base file (used so every experiment shares the frozen detector hyperparameters).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single-level `extends:` parent."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIGS / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        base = load_config(parent_path)
        cfg = _deep_merge(base, cfg)
    return cfg
