"""Generate an SD3.5 + ControlNet synthetic pool (Stage C).

Run on the cluster (needs a big GPU). Two steps:

  # Stage B prep (once): export ControlNet training pairs + build instance bank.
  python scripts/gen_sd35.py prep

  # Stage C: synthesize N labelled images into data/synthetic/sd35/.
  python scripts/gen_sd35.py generate --n 2000 --control canny --seed 0

The output (images/ + labels/) is picked up directly by Stage D:
  python -m src.datasets.build_hybrid --name hybrid_sd35_r50 \
      --composition hybrid --generator sd35 --ratio 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import paths  # noqa: E402
from src.generators.sd35_controlnet import SD35ControlNet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="SD3.5 + ControlNet generation (Stage B/C).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prep", help="export ControlNet pairs + build instance bank")
    p_prep.add_argument("--control", default="seg", choices=["seg", "canny"])

    p_gen = sub.add_parser("generate", help="synthesize labelled images")
    p_gen.add_argument("--n", type=int, required=True)
    p_gen.add_argument("--control", default="seg", choices=["seg", "canny"])
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--steps", type=int, default=28)
    p_gen.add_argument("--imgsz", type=int, default=1024)
    p_gen.add_argument("--cn-scale", type=float, default=1.0,
                       help="controlnet_conditioning_scale; raise (1.5-3.0) for tighter layout adherence")

    args = ap.parse_args()
    gen_cfg = {"control_kind": args.control}
    if args.cmd == "generate":          # imgsz/steps only exist on the generate subcommand
        gen_cfg["image_size"] = (args.imgsz, args.imgsz)
        gen_cfg["steps"] = args.steps
        gen_cfg["controlnet_scale"] = args.cn_scale

    gen = SD35ControlNet({"generator": gen_cfg})

    if args.cmd == "prep":
        # train_yaml guard expects a 'train' path; pass the real train descriptor.
        gen.finetune(paths.REAL / "real.yaml")
    elif args.cmd == "generate":
        gen.generate(args.n, seed=args.seed)


if __name__ == "__main__":
    main()
