"""Merge per-species cutout banks into one pool the compositor can use.

Each per-species job writes data/synthetic/<prefix>_<cls>/cutouts/<CLS>/. This
links them into a single pool so Stage C composites scenes containing all five
species, with each species rendered by its own LoRA.

Backgrounds are taken from an existing pool (they are real soil crops and have
nothing to do with the generator).

  python scripts/merge_cutout_pools.py --out sd35cut_perspecies \
      --prefix sd35cut --backgrounds-from sd35cut_lora_v2

Then composite and sweep as usual:

  python scripts/fit_scale_prior.py --split train
  python scripts/gen_sd35_cutout.py generate --pool sd35cut_perspecies --n 2500 \
      --seed 0 --blend feather --imgsz 1024 --scale-mode prior --bg-mode tilematch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import classes, paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="name of the merged pool")
    ap.add_argument("--prefix", default="sd35cut",
                    help="per-species pools are <prefix>_<cls lowercase>")
    ap.add_argument("--backgrounds-from", default="sd35cut_lora_v2")
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking the class directories")
    args = ap.parse_args()

    dst = paths.SYNTHETIC / args.out
    (dst / "cutouts").mkdir(parents=True, exist_ok=True)

    total, missing = 0, []
    for cname in classes.CLASS_NAMES:
        src = paths.SYNTHETIC / f"{args.prefix}_{cname.lower()}" / "cutouts" / cname
        link = dst / "cutouts" / cname
        if not src.exists():
            missing.append(cname)
            continue
        n = len(list(src.glob("*.png")))
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            elif args.copy:
                import shutil
                shutil.rmtree(link)
        if args.copy:
            import shutil
            shutil.copytree(src, link)
        else:
            link.symlink_to(src.resolve(), target_is_directory=True)
        print(f"  {cname:8s} {n:5d} cutouts  <- {src}")
        total += n

    bg_src = paths.SYNTHETIC / args.backgrounds_from / "backgrounds"
    bg_dst = dst / "backgrounds"
    if not bg_src.exists():
        raise SystemExit(f"no backgrounds at {bg_src} — pass --backgrounds-from")
    if bg_dst.is_symlink() or bg_dst.exists():
        if bg_dst.is_symlink():
            bg_dst.unlink()
    if not bg_dst.exists():
        bg_dst.symlink_to(bg_src.resolve(), target_is_directory=True)
    print(f"  backgrounds <- {bg_src} ({len(list(bg_src.glob('*.png')))} tiles)")

    if missing:
        print(f"\n  MISSING: {', '.join(missing)} — those species have no cutouts, "
              f"so composited scenes will contain none of them.")
    print(f"\n[merge] {total} cutouts across {len(classes.CLASS_NAMES) - len(missing)} "
          f"species -> {dst}")
    print("[merge] next: python scripts/gen_sd35_cutout.py generate "
          f"--pool {args.out} --n 2500 --seed 0 --blend feather --imgsz 1024 "
          "--scale-mode prior --bg-mode tilematch")


if __name__ == "__main__":
    main()
