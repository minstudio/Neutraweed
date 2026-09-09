"""Near-duplicate removal for synthetic pools (Stage C).

Perceptual-hash (aHash) based: cheap, dependency-light (PIL + numpy), good enough
to catch the mode-collapse near-duplicates diffusion/GAN pools sometimes produce.
Threshold = max Hamming distance to treat two images as duplicates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def ahash(image_path: Path, hash_size: int = 8) -> np.ndarray:
    from PIL import Image

    img = Image.open(image_path).convert("L").resize((hash_size, hash_size))
    arr = np.asarray(img, dtype=np.float32)
    return (arr > arr.mean()).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def find_duplicates(image_paths: list[Path], threshold: int = 6) -> list[Path]:
    """Return the paths to drop (keeps the first of each near-duplicate cluster)."""
    hashes: list[tuple[Path, np.ndarray]] = [(p, ahash(p)) for p in image_paths]
    drop: list[Path] = []
    kept: list[np.ndarray] = []
    for p, h in hashes:
        if any(hamming(h, k) <= threshold for k in kept):
            drop.append(p)
        else:
            kept.append(h)
    return drop


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("pool_dir", type=Path)
    ap.add_argument("--threshold", type=int, default=6)
    args = ap.parse_args()
    imgs = sorted(
        p for p in args.pool_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    dups = find_duplicates(imgs, args.threshold)
    print(f"{len(dups)}/{len(imgs)} near-duplicates (threshold={args.threshold})")
