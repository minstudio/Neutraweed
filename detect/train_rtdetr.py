"""Train RT-DETRv2 under the frozen protocol (CLAUDE.md §4 Stage E).

The transformer arm that answers the CNN-vs-transformer question. Ultralytics
ships an RTDETR class; for the official RT-DETRv2 checkpoints, swap the loader
but keep the SAME frozen hyperparameters so the only difference vs the YOLOs is
the architecture.

Run:
  python -m src.detect.train_rtdetr --config experiments/hybrid_sd35_rtdetr_r50.yaml \
      --data data/datasets/hybrid_sd35_r50/dataset.yaml --seed 0
"""

from __future__ import annotations

import argparse

from ..common import paths
from ..common.config import load_config


def _ckpt_ok(path) -> bool:
    """True only if a checkpoint loads cleanly (a walltime kill mid-write leaves a
    truncated last.pt that crashes resume). Lets us fall back to a fresh start."""
    try:
        import torch
        torch.load(str(path), map_location="cpu", weights_only=False)
        return True
    except Exception as exc:
        print(f"[train] checkpoint {path} is unreadable ({exc}); ignoring it")
        return False


def train(config_path: str, data_yaml: str, seed: int | None = None,
          name: str | None = None) -> dict:
    cfg = load_config(config_path)
    d = cfg["detector"]
    seed = d.get("seed", 0) if seed is None else seed

    try:
        from ultralytics import RTDETR
    except ImportError as exc:
        raise ImportError("pip install ultralytics (provides RTDETR)") from exc

    weights = d.get("weights", "rtdetr-l.pt")
    run_name = f"{name or cfg.get('name', 'run')}_seed{seed}"

    last = paths.CHECKPOINTS / run_name / "weights" / "last.pt"
    if last.exists() and _ckpt_ok(last):   # resume after a cancellation / walltime kill
        print(f"[train] resuming {run_name} from {last}")
        results = RTDETR(str(last)).train(resume=True)
    else:
        if last.exists():
            last.unlink()              # drop corrupt checkpoint; start clean
        results = RTDETR(weights).train(
            data=str(data_yaml),
            imgsz=d["imgsz"],
            epochs=d["epochs"],
            patience=d["patience"],
            batch=d["batch"],
            optimizer=d["optimizer"],
            lr0=d["lr0"],
            seed=seed,
            deterministic=d.get("deterministic", True),
            project=str(paths.CHECKPOINTS),
            name=run_name,
            exist_ok=True,
            verbose=True,
        )
    print(f"Trained RT-DETRv2 -> {paths.CHECKPOINTS / run_name}")
    return {"run_name": run_name, "save_dir": str(getattr(results, 'save_dir', ''))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None, help="run name (distinguishes datasets)")
    args = ap.parse_args()
    train(args.config, args.data, args.seed, name=args.name)
