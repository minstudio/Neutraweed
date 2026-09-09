"""Image-quality QC across synthetic pools (Stage C).

For each pool, computes FID vs the real images (pool-level, lower=better) and mean
CLIP-IQA over a sample (per-image perceptual quality, higher=better), and writes
results/tables/qc_pools.csv. Pairs with the downstream mAP so you can say
"sd35cut_lora is FID X / CLIP-IQA Y, and costs Z mAP". Needs a GPU (sbatch).

  python scripts/qc_pools.py --pools composite sd35cut sd35cut_lora --sample 500
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import paths


def _pool_images(pool: str):
    return sorted((paths.SYNTHETIC / pool / "images").glob("*.png"))


def run(pools, real_dir: Path, sample: int, device: str, out_csv: Path):
    from src.qc.fid import compute_fid
    from src.qc.clip_iqa import score_batch

    if not real_dir.exists():
        raise SystemExit(f"real reference dir not found: {real_dir}")
    rows = []
    for pool in pools:
        imgs = _pool_images(pool)
        if not imgs:
            print(f"[qc] skip {pool}: no images at {paths.SYNTHETIC/pool/'images'}")
            continue
        print(f"[qc] {pool}: {len(imgs)} images — computing FID vs {real_dir} ...")
        fid = compute_fid(real_dir, paths.SYNTHETIC / pool / "images", device)
        samp = random.Random(0).sample(imgs, min(sample, len(imgs)))
        scores = score_batch(samp, device)
        row = {
            "pool": pool, "n_images": len(imgs), "n_scored": len(samp),
            "fid": round(fid, 3),
            "clip_iqa_mean": round(st.mean(scores), 4),
            "clip_iqa_std": round(st.pstdev(scores) if len(scores) > 1 else 0.0, 4),
        }
        rows.append(row)
        print(f"[qc] {pool}: FID={row['fid']}  CLIP-IQA={row['clip_iqa_mean']}"
              f"+/-{row['clip_iqa_std']}")
    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[qc] wrote {out_csv}")
    else:
        print("[qc] no pools scored")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", nargs="+", default=["composite", "sd35cut", "sd35cut_lora"])
    ap.add_argument("--real-dir", type=Path, default=paths.REAL / "images" / "train")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=paths.TABLES / "qc_pools.csv")
    args = ap.parse_args()
    run(args.pools, args.real_dir, args.sample, args.device, args.out)


if __name__ == "__main__":
    main()
