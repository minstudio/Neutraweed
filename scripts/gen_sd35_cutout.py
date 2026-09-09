"""Build the SD-cutout synthetic pool (Stage C).

The salvageable SD arm: SD3.5 renders single isolated weeds, we matte them to
RGBA cutouts, then paste them onto real soil backgrounds via the existing
composite pipeline. Labels are perfect by construction. Output is a standard
synthetic pool (data/synthetic/<pool>/{images,labels}) that Stage D consumes
like any other generator: build_hybrid/run_sweep --generator <pool>.

Order:
  1. cutouts   (GPU, sbatch) : SD3.5 single-weed render + matte -> cutouts/<CLS>/
  2. backgrounds (CPU)       : copy real soil tiles from a composite pool
  3. generate  (CPU)         : composite paste -> labelled pool

  python scripts/gen_sd35_cutout.py cutouts --pool sd35cut --per-class 300
  python scripts/gen_sd35_cutout.py backgrounds --pool sd35cut --from-pool composite2022
  python scripts/gen_sd35_cutout.py generate --pool sd35cut --n 2500 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generators import composite_gen, sd35_cutout
from src.generators.composite_gen import add_scene_args, scene_kwargs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cutouts", help="GPU: single-weed render (SD3.5 or FLUX) + matte")
    c.add_argument("--backend", default="sd35", choices=["sd35", "flux"])
    c.add_argument("--pool", default=None, help="default: sd35cut / fluxcut by backend")
    c.add_argument("--per-class", type=int, default=300)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--base", default=None)
    c.add_argument("--lora", default=None, help="domain LoRA dir (optional)")
    c.add_argument("--matte", default="exg", choices=["exg", "rembg"])
    c.add_argument("--render-size", type=int, default=768)
    c.add_argument("--steps", type=int, default=28)
    c.add_argument("--guidance", type=float, default=4.5)
    c.add_argument("--device", default="cuda")
    c.add_argument("--min-sharpness", type=float, default=0.0,
                   help="absolute variance-of-Laplacian floor; 0 = rank-only")
    c.add_argument("--buffer-frac", type=float, default=0.35,
                   help="over-render fraction; keep the sharpest per_class of these")
    c.add_argument("--min-clipiqa", type=float, default=0.0,
                   help="CLIP-IQA perceptual-quality gate on cutouts (0=off, e.g. 0.5)")
    c.add_argument("--classes", nargs="*", default=None,
                   help="render only these class codes (e.g. ECHCG); default all")
    c.add_argument("--cpu-offload", action="store_true",
                   help="FLUX only: model CPU offload for GPUs that can't hold it")
    c.add_argument("--keep-rejects", action="store_true",
                   help="save matte + soft (dropped) cutouts to rejects/ for QC")
    c.add_argument("--stages", action="store_true",
                   help="vary the growth-stage phrase per render (phenology coverage)")

    b = sub.add_parser("backgrounds", help="CPU: copy real soil tiles from a composite pool")
    b.add_argument("--pool", default="sd35cut")
    b.add_argument("--from-pool", default="composite",
                   help="composite pool whose backgrounds to reuse (domain match: composite2022)")

    g = sub.add_parser("generate", help="CPU: composite paste -> labelled pool")
    g.add_argument("--pool", default="sd35cut")
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--imgsz", type=int, default=1024)
    g.add_argument("--blend", default="feather",
                   choices=["poisson", "alpha", "feather", "photoreal"])
    add_scene_args(g)

    args = ap.parse_args()
    if args.cmd == "cutouts":
        if args.backend == "flux":
            pool = args.pool or "fluxcut"
            gen = sd35_cutout.FluxCutoutGenerator(
                pool=pool, base=args.base, lora=args.lora, device=args.device,
                steps=args.steps, guidance=args.guidance, render_size=args.render_size,
                matte=args.matte, min_sharpness=args.min_sharpness,
                buffer_frac=args.buffer_frac, min_clipiqa=args.min_clipiqa,
                cpu_offload=args.cpu_offload, stages=args.stages)
        else:
            pool = args.pool or "sd35cut"
            gen = sd35_cutout.SD35CutoutGenerator(
                pool=pool, base=args.base, lora=args.lora, device=args.device,
                steps=args.steps, guidance=args.guidance, render_size=args.render_size,
                matte=args.matte, min_sharpness=args.min_sharpness,
                buffer_frac=args.buffer_frac, min_clipiqa=args.min_clipiqa,
                stages=args.stages)
        gen.cutouts(args.per_class, seed=args.seed,
                    keep_rejects=args.keep_rejects, only=args.classes)
    elif args.cmd == "backgrounds":
        sd35_cutout.copy_backgrounds(args.pool, from_pool=args.from_pool)
    elif args.cmd == "generate":
        g = composite_gen.CompositeGenerator(
            image_size=args.imgsz, blend=args.blend, name=args.pool, **scene_kwargs(args))
        g.generate(args.n, seed=args.seed)


if __name__ == "__main__":
    main()
