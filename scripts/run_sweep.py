"""Orchestrate the hybrid ratio sweep end to end.

For a given generator + detector, builds datasets across the ratio grid, trains
each over N seeds under the frozen protocol, evaluates on the real test set, and
(optionally) the matched-size control at each ratio. Designed to be run on the
GPU box once Stages B/C have produced a synthetic pool.

This is a thin driver over the src modules — it imports them so a dry run can be
validated without a GPU.

Examples:
  # dry run: just print the plan (no training)
  python scripts/run_sweep.py --generator sd35 --detector yolo11 --dry-run

  # real run: ratios 0.1..2.0 step 0.1, seeds 0..2
  python scripts/run_sweep.py --generator sd35 --detector yolo11 \
      --ratios 0.1:2.0:0.1 --seeds 0 1 2 --matched-control
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running as a loose script: add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import build_hybrid, matched_size  # noqa: E402


def parse_ratios(spec: str) -> list[float]:
    lo, hi, step = (float(x) for x in spec.split(":"))
    out, r = [], lo
    while r <= hi + 1e-9:
        out.append(round(r, 3))
        r += step
    return out


def _config_for(detector: str) -> str:
    # map detector -> a base experiment config providing frozen hyperparameters
    return {
        "yolo11": "experiments/real_only_yolo11.yaml",
        "yolo26": "experiments/real_only_yolo26.yaml",
        "rtdetrv2": "experiments/real_only_rtdetrv2.yaml",
    }[detector]


def run(args) -> None:
    detector = args.detector
    is_rtdetr = detector.startswith("rtdetr")

    plan = []
    # real-only baseline first (skippable: already trained, and it would be
    # rebuilt/retrained by every array cell otherwise)
    if not args.no_baseline:
        plan.append(("real-only", None, 0.0))
    for ratio in parse_ratios(args.ratios):
        plan.append(("hybrid", args.generator, ratio))

    print(f"Sweep: generator={args.generator} detector={detector} "
          f"seeds={args.seeds} | {len(plan)} compositions")

    for composition, gen, ratio in plan:
        tag = "real" if composition == "real-only" else f"{gen}_r{int(ratio*100)}"
        ds_name = f"{composition}_{detector}_{tag}".replace("real-only", "realonly")
        if args.max_overlap < 1.0:
            # a filtered dataset is a different dataset; never let it silently
            # reuse or overwrite the unfiltered one of the same ratio
            ds_name += f"_x{int(round(args.max_overlap * 100))}"
        print(f"\n=== {ds_name} ===")

        if args.dry_run:
            print(f"  build {composition} ratio={ratio}; train seeds {args.seeds}; eval test")
            continue

        data_yaml = build_hybrid.build(ds_name, composition, gen, ratio, seed=0,
                                       max_overlap=args.max_overlap,
                                       strict_unique=args.strict_unique)

        if not args.build_only:
            # train + eval each seed (frozen protocol)
            from src.detect import train_rtdetr, train_yolo
            from src.eval import metrics
            trainer = train_rtdetr if is_rtdetr else train_yolo
            for seed in args.seeds:
                info = trainer.train(_config_for(detector), str(data_yaml), seed, name=ds_name)
                weights = Path(info["save_dir"]) / "weights" / "best.pt"
                metrics.evaluate(weights, data_yaml, info["run_name"], split="test")

        # matched-size control at this ratio — build always; train unless build-only
        if args.matched_control and composition == "hybrid":
            import yaml
            meta = yaml.safe_load(Path(data_yaml).read_text())["meta"]
            ctrl_name = f"matched_{detector}_{tag}"
            ctrl_yaml = matched_size.build(ctrl_name, meta["n_train"], seed=0)
            if not args.build_only:
                from src.detect import train_rtdetr, train_yolo
                from src.eval import metrics
                trainer = train_rtdetr if is_rtdetr else train_yolo
                for seed in args.seeds:
                    info = trainer.train(_config_for(detector), str(ctrl_yaml), seed, name=ctrl_name)
                    weights = Path(info["save_dir"]) / "weights" / "best.pt"
                    metrics.evaluate(weights, ctrl_yaml, info["run_name"], split="test")

    print("\nSweep complete. Aggregate with: python -m src.eval.stats / src.eval.plots")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="composite",
                    help="synthetic pool name under data/synthetic/ (composite, sd35cut, ...)")
    ap.add_argument("--detector", choices=["yolo11", "yolo26", "rtdetrv2"], default="yolo11")
    ap.add_argument("--ratios", default="0.1:2.0:0.1", help="lo:hi:step")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--matched-control", action="store_true")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the real-only baseline (already trained; for array cells)")
    ap.add_argument("--build-only", action="store_true",
                    help="assemble datasets only, no training (prebuild step for the array)")
    ap.add_argument("--max-overlap", type=float, default=1.0,
                    help="drop TRAINING tiles where more than this fraction of "
                         "plants overlap another (IoU>=0.10), real and synthetic "
                         "alike. Use the same value with ap_analysis "
                         "--drop-overlap so training and scoring agree.")
    ap.add_argument("--strict-unique", action="store_true",
                    help="fail rather than reuse synthetic images to reach a "
                         "ratio the pool cannot cover uniquely.")
    ap.add_argument("--dry-run", action="store_true")
    run(ap.parse_args())
