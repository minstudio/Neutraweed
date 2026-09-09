"""Cut-and-composite synthetic generation (Stage C, Option B).

CPU-only — runs on a laptop or a CPU node; no GPU/diffusion needed.

  # one-time: extract real weed cutouts + soil backgrounds from the SAM2 masks
  python scripts/gen_composite.py prep

  # generate a labelled pool (labels are exact by construction)
  python scripts/gen_composite.py generate --n 2500 --seed 0

Output (data/synthetic/composite/{images,labels}) feeds Stage D as generator
'composite':
  python -m src.datasets.build_hybrid --name hybrid_comp_r50 \
      --composition hybrid --generator composite --ratio 0.5

Domain variants (annotation-reduction headline test) — build a
separate pool carrying the 2022 target domain at zero annotation cost: harvest
weed-free 2022-Finca soil backgrounds (no boxes) and REUSE the cutouts the
working 'composite' pool already has (--cutouts-from composite; no SAM2 masks
needed on the cluster):

  # 2022-background pool, existing cutouts
  python scripts/gen_composite.py prep --pool composite2022 \
      --bg-sources TOMATO_2__ --cutouts-from composite
  python scripts/gen_composite.py generate --pool composite2022 --n 2500 --seed 0

  # strict zero-cost: 2022 backgrounds + 2021-only cutouts (no 2022 boxes anywhere)
  python scripts/gen_composite.py prep --pool composite2022strict \
      --bg-sources TOMATO_2__ --cutouts-from composite --cut-sources TOMATO_1__
  python scripts/gen_composite.py generate --pool composite2022strict --n 2500 --seed 0

If a dense field yields too few soil tiles, loosen the greenness gate:
  ... prep --pool composite2022 --bg-sources TOMATO_2__ --max-green 0.10 --per-image 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generators import composite_gen  # noqa: E402
from src.generators.composite_gen import add_scene_args, scene_kwargs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Cut-and-composite generation (CPU).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep", help="extract cutouts + soil backgrounds from real data")
    p.add_argument("--pool", default="composite",
                   help="pool name -> data/synthetic/<pool> (e.g. composite2022)")
    p.add_argument("--bg-sources", nargs="*", default=None,
                   help="restrict background source images by filename prefix "
                        "(e.g. TOMATO_2__ for 2022-Finca soil). Default: all.")
    p.add_argument("--cut-sources", nargs="*", default=None,
                   help="restrict cutouts by filename prefix "
                        "(e.g. TOMATO_1__ for 2021-only cutouts). Default: all.")
    p.add_argument("--cutouts-from", default=None,
                   help="reuse cutouts from an existing pool (e.g. composite) instead "
                        "of re-extracting from SAM2 masks. Needs no mask files. "
                        "Combine with --cut-sources to subset by year.")
    p.add_argument("--max-green", type=float, default=0.05,
                   help="max mean Excess-Green for a tile to count as bare soil. "
                        "Raise (e.g. 0.10) if a dense field yields too few tiles.")
    p.add_argument("--per-image", type=int, default=2,
                   help="soil tiles to harvest per source image (default 2).")
    p.add_argument("--max-images", type=int, default=400,
                   help="cap on source images scanned for backgrounds (default 400).")
    p.add_argument("--bg-size", type=int, default=1024,
                   help="soil tile side in NATIVE pixels. Use 1536 (= tiling.size) "
                        "so generate --bg-mode crop can take one window and "
                        "downscale it exactly as a real tile is made, instead of "
                        "mosaicking four 1024 crops and leaving a 2x2 seam.")

    b = sub.add_parser("backgrounds",
                       help="(re)harvest soil tiles only, leaving cutouts alone")
    b.add_argument("--pool", required=True)
    b.add_argument("--split", default="train")
    b.add_argument("--bg-size", type=int, default=1536,
                   help="soil tile side in native px. 1536 (= tiling.size) is what "
                        "generate --bg-mode crop needs.")
    b.add_argument("--bg-sources", nargs="*", default=None)
    b.add_argument("--max-green", type=float, default=0.05)
    b.add_argument("--per-image", type=int, default=3)
    b.add_argument("--max-images", type=int, default=400)
    b.add_argument("--match-luminance", action="store_true",
                   help="subsample the harvested tiles so their brightness "
                        "follows that of unconstrained real tiles. The no-box + "
                        "low-ExG gate preferentially accepts shadowed soil, "
                        "leaving the bank 57%% darker in shadow depth than a "
                        "random real tile; this corrects it.")
    b.add_argument("--bg-oversample", type=int, default=4,
                   help="candidates gathered per tile finally kept under "
                        "--match-luminance. The correction can only discard.")

    g = sub.add_parser("generate", help="composite a labelled synthetic pool")
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--imgsz", type=int, default=1024)
    g.add_argument("--blend", default="feather",
                   choices=["poisson", "alpha", "feather", "photoreal"])
    g.add_argument("--pool", default="composite",
                   help="pool name -> data/synthetic/<pool> (must match prep --pool)")
    add_scene_args(g)

    args = ap.parse_args()
    if args.cmd == "prep":
        composite_gen.prep(name=args.pool, bg_sources=args.bg_sources,
                           cut_sources=args.cut_sources, cutouts_from=args.cutouts_from,
                           max_green=args.max_green, per_image=args.per_image,
                           max_images=args.max_images, bg_size=args.bg_size)
    elif args.cmd == "backgrounds":
        composite_gen.extract_backgrounds(
            args.split, name=args.pool, size=args.bg_size, sources=args.bg_sources,
            max_green=args.max_green, per_image=args.per_image,
            max_images=args.max_images, match_luminance=args.match_luminance,
            oversample=args.bg_oversample)
    elif args.cmd == "generate":
        gen = composite_gen.CompositeGenerator(image_size=args.imgsz, blend=args.blend,
                                               name=args.pool, **scene_kwargs(args))
        gen.generate(args.n, seed=args.seed)


if __name__ == "__main__":
    main()
