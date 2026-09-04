"""Per-image CLIP-IQA perceptual quality (CLAUDE.md §4 Stage C, ref 2411.18513).

Scores each synthetic image in [0,1]; images below `clip_iqa_min` (configurable
in base.yaml) are dropped before training. Uses torchmetrics' CLIPImageQualityAssessment.

    pip install torchmetrics
"""

from __future__ import annotations

from pathlib import Path


def score_image(image_path: Path, device: str = "cuda") -> float:
    scores = score_batch([image_path], device)
    return scores[0]


def score_batch(image_paths: list[Path], device: str = "cuda") -> list[float]:
    try:
        import torch
        from PIL import Image
        from torchmetrics.multimodal import CLIPImageQualityAssessment
        from torchvision.transforms.functional import to_tensor
    except ImportError as exc:
        raise ImportError("Install torchmetrics + torchvision for CLIP-IQA.") from exc

    metric = CLIPImageQualityAssessment(model_name_or_path="clip_iqa").to(device)
    out: list[float] = []
    for p in image_paths:
        img = to_tensor(Image.open(p).convert("RGB")).unsqueeze(0).to(device) * 255.0
        with torch.no_grad():
            out.append(float(metric(img).item()))
    return out


def filter_pool(image_paths: list[Path], min_score: float, device: str = "cuda"):
    """Return (kept, dropped) partition by CLIP-IQA threshold."""
    scores = score_batch(image_paths, device)
    kept = [p for p, s in zip(image_paths, scores) if s >= min_score]
    dropped = [p for p, s in zip(image_paths, scores) if s < min_score]
    return kept, dropped


def score_arrays(images, device: str = "cuda") -> list[float]:
    """CLIP-IQA over in-memory RGB uint8 arrays (loads the metric once). Used to
    curate SD/FLUX cutouts before compositing."""
    try:
        import torch
        from PIL import Image
        from torchmetrics.multimodal import CLIPImageQualityAssessment
        from torchvision.transforms.functional import to_tensor
    except ImportError as exc:
        raise ImportError("Install torchmetrics + torchvision for CLIP-IQA.") from exc
    metric = CLIPImageQualityAssessment(model_name_or_path="clip_iqa").to(device)
    out: list[float] = []
    for arr in images:
        t = to_tensor(Image.fromarray(arr).convert("RGB")).unsqueeze(0).to(device) * 255.0
        with torch.no_grad():
            out.append(float(metric(t).item()))
    return out
