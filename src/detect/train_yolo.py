"""Train YOLO11 / YOLO26 under the frozen protocol (Stage E).

Both are Ultralytics one-stage detectors, so they share this entry point; the
`detector.name`/`detector.weights` in the config select the variant. Pin the
Ultralytics version for YOLO26 (young tooling).

Run:
  python -m src.detect.train_yolo --config experiments/real_only_yolo11.yaml \
      --data data/datasets/real_only/dataset.yaml --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..common import paths
from ..common.config import load_config


def _ckpt_ok(path) -> bool:
    """True only if a checkpoint loads cleanly. A walltime kill mid-write leaves a
    truncated last.pt that raises 'PytorchStreamReader ... failed finding central
    directory' on resume — this lets us fall back to a fresh start instead of
    crashing the whole array cell."""
    try:
        import torch
        torch.load(str(path), map_location="cpu", weights_only=False)
        return True
    except Exception as exc:  # corrupt / truncated / unreadable
        print(f"[train] checkpoint {path} is unreadable ({exc}); ignoring it")
        return False


def train(config_path: str, data_yaml: str, seed: int | None = None,
          name: str | None = None, imgsz: int | None = None,
          batch: int | None = None, epochs: int | None = None) -> dict:
    cfg = load_config(config_path)
    d = cfg["detector"]
    seed = d.get("seed", 0) if seed is None else seed
    the_imgsz = imgsz if imgsz is not None else d["imgsz"]   # resolution override
    the_batch = batch if batch is not None else d["batch"]   # batch override (OOM at hi-res)
    the_epochs = epochs if epochs is not None else d["epochs"]

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("pip install ultralytics") from exc

    # Optional augmentation overrides. Absent from every existing config, so the
    # frozen protocol is unchanged and prior runs stay comparable; an experiment
    # opts in by adding an `augment:` block.
    #
    # Worth knowing what the defaults are, because two obvious ones are off:
    # Ultralytics uses degrees=0.0 and flipud=0.0. This imagery is nadir — shot
    # straight down at soil — so it has no canonical orientation, and rotation
    # and vertical flip are label-preserving here in a way they are not for most
    # detection data. They are free training signal that is currently unused.
    aug = dict(cfg.get("augment") or {})
    if aug:
        print(f"[train] augmentation overrides: {aug}")

    # `name` (the dataset) distinguishes real-only / hybrid / matched runs; the
    # config's name is shared (frozen detector), so it can't identify the run.
    run_name = f"{name or cfg.get('name', 'run')}_seed{seed}"

    # Resume from last.pt if a prior (interrupted) run exists, so a cancellation /
    # walltime kill doesn't lose hours — the long tiled runs get interrupted.
    last = paths.CHECKPOINTS / run_name / "weights" / "last.pt"
    if last.exists() and _ckpt_ok(last):
        print(f"[train] resuming {run_name} from {last}")
        model = YOLO(str(last))
        results = model.train(resume=True)
    else:
        if last.exists():
            last.unlink()  # drop the corrupt file so Ultralytics starts clean
        model = YOLO(d["weights"])  # COCO-pretrained init (frozen)
        # FROZEN hyperparameters — identical across every experiment.
        results = model.train(
            data=str(data_yaml),
            imgsz=the_imgsz,
            epochs=the_epochs,
            patience=d["patience"],
            batch=the_batch,
            optimizer=d["optimizer"],
            lr0=d["lr0"],
            seed=seed,
            deterministic=d.get("deterministic", True),
            project=str(paths.CHECKPOINTS),
            name=run_name,
            exist_ok=True,
            verbose=True,
            **aug,
        )
    print(f"Trained {d['name']} -> {paths.CHECKPOINTS / run_name}")
    return {"run_name": run_name, "save_dir": str(getattr(results, 'save_dir', ''))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="experiment YAML (under configs/)")
    ap.add_argument("--data", required=True, help="dataset.yaml from Stage D")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None, help="run name (distinguishes datasets)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="override the config's training/inference resolution")
    ap.add_argument("--batch", type=int, default=None,
                    help="override the config's batch size (lower at high imgsz to avoid OOM)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the config's epoch budget. Set this explicitly "
                         "when comparing configs: a walltime kill stops bigger "
                         "datasets at fewer epochs, which silently confounds the "
                         "ratio sweep with training length.")
    args = ap.parse_args()
    train(args.config, args.data, args.seed, name=args.name,
          imgsz=args.imgsz, batch=args.batch, epochs=args.epochs)
