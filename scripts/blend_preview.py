from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotate.composite import Cutout, paste_cutouts
from src.common import classes, paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Side-by-side poisson vs feather composite preview, so the "
                    "soil-colour blending problem is visible before regenerating "
                    "a whole pool. CPU only, safe on the login node.")
    ap.add_argument("--pool", default="composite")
    ap.add_argument("--n", type=int, default=6, help="scenes (rows)")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--k", type=int, default=8, help="cutouts per scene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "blend_preview.png")
    args = ap.parse_args()

    from PIL import Image

    pool = paths.SYNTHETIC / args.pool
    cut_dir, bg_dir = pool / "cutouts", pool / "backgrounds"
    cutouts = {c: list((cut_dir / c).glob("*.png")) for c in classes.CLASS_NAMES}
    cutouts = {c: v for c, v in cutouts.items() if v}
    bgs = list(bg_dir.glob("*.png"))
    if not cutouts or not bgs:
        raise SystemExit(f"pool '{args.pool}' missing cutouts or backgrounds")

    rng = random.Random(args.seed)
    tmp = paths.FIGURES / "_blend_tmp.txt"
    rows = []
    for _ in range(args.n):
        bg = np.asarray(Image.open(rng.choice(bgs)).convert("RGB")
                        .resize((args.imgsz, args.imgsz)))
        cuts = []
        for _ in range(args.k):
            c = rng.choice(list(cutouts))
            rgba = np.asarray(Image.open(rng.choice(cutouts[c])).convert("RGBA"))
            s = rng.randint(80, 200) / max(rgba.shape[:2])
            nh, nw = max(8, int(rgba.shape[0] * s)), max(8, int(rgba.shape[1] * s))
            rgba = np.asarray(Image.fromarray(rgba).resize((nw, nh), Image.LANCZOS))
            cuts.append(Cutout(rgba=rgba, cls_name=c))
        sd = rng.randint(0, 10 ** 6)
        pos, _ = paste_cutouts(bg.copy(), cuts, tmp, seed=sd, blend="poisson")
        fea, _ = paste_cutouts(bg.copy(), cuts, tmp, seed=sd, blend="feather")
        rows.append(np.concatenate([pos, np.full((args.imgsz, 8, 3), 255, np.uint8), fea], 1))
    if tmp.exists():
        tmp.unlink()

    grid = np.concatenate(
        [np.concatenate([r, np.full((8, rows[0].shape[1], 3), 255, np.uint8)], 0)
         for r in rows], 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(args.out)
    print(f"wrote {args.out}  (left column = POISSON, right column = FEATHER)")


if __name__ == "__main__":
    main()
