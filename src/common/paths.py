"""Centralised filesystem layout.

Everything resolves relative to the repo root so the pipeline behaves the same
on the dev box (RTX 4090) and the CSIC cluster (A100). Override the root with
the WEED_GENAI_ROOT environment variable if you relocate the data.
"""

from __future__ import annotations

import os
from pathlib import Path

# repo root = two levels up from this file (src/common/paths.py -> repo)
REPO_ROOT = Path(os.environ.get("WEED_GENAI_ROOT", Path(__file__).resolve().parents[2]))

# Raw source datasets (Pascal VOC), as delivered.
RAW_SOURCES = {
    "TOMATO_1": REPO_ROOT / "TOMATO_1",
    "TOMATO_2": REPO_ROOT / "TOMATO_2",
}

DATA = REPO_ROOT / "data"
REAL = DATA / "real"                 # frozen split: images + YOLO labels
MASKS = REAL / "masks"               # SAM2 box-prompted instance masks (Phase 0)
SYNTHETIC = DATA / "synthetic"       # per-generator pools + auto-labels
DATASETS = DATA / "datasets"         # assembled real/synthetic/hybrid manifests

CONFIGS = REPO_ROOT / "configs"
RESULTS = REPO_ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

# The frozen split lives here once make_splits has run.
SPLIT_FILE = REAL / "split.json"

SPLITS = ("train", "val", "test")


def ensure_dirs() -> None:
    """Create the standard output directories. Idempotent."""
    for d in (
        DATA, REAL, SYNTHETIC, DATASETS,
        RESULTS, CHECKPOINTS, FIGURES, TABLES,
    ):
        d.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (REAL / "images" / split).mkdir(parents=True, exist_ok=True)
        (REAL / "labels" / split).mkdir(parents=True, exist_ok=True)
