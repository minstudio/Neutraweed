"""Detector efficiency: params, FLOPs, and inference latency (Stage F).

For each detector's checkpoint, reports #params, GFLOPs, per-image latency
(preprocess/inference/postprocess ms) and FPS on the real test set, into
results/tables/efficiency.csv. Latency is an architecture property, so one
checkpoint per detector suffices (e.g. the real-only run). Needs a GPU.

  python scripts/measure_efficiency.py \
    --runs yolo11=results/checkpoints/realonly_yolo11_real_seed0/weights/best.pt \
           yolo26=results/checkpoints/realonly_yolo26_real_seed0/weights/best.pt \
           rtdetrv2=results/checkpoints/realonly_rtdetrv2_real_seed0/weights/best.pt \
    --data data/datasets/realonly_yolo11_real/dataset.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import paths


def measure(det: str, weights: str, data: str, imgsz: int) -> dict:
    from ultralytics import RTDETR, YOLO
    Model = RTDETR if det.startswith("rtdetr") else YOLO
    m = Model(weights)
    # robust params/FLOPs (m.info() return shape varies across ultralytics versions)
    params = sum(p.numel() for p in m.model.parameters())
    gflops = None
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = get_flops(m.model, imgsz)
    except Exception as exc:
        print(f"[eff] {det}: FLOPs unavailable ({exc})")
    res = m.val(data=data, split="test", imgsz=imgsz, verbose=False)
    sp = res.speed                        # ms per image
    inf = float(sp.get("inference", 0.0))
    return {
        "detector": det,
        "params_M": round(params / 1e6, 2) if params else "",
        "gflops": round(gflops, 1) if gflops else "",
        "preprocess_ms": round(float(sp.get("preprocess", 0.0)), 2),
        "inference_ms": round(inf, 2),
        "postprocess_ms": round(float(sp.get("postprocess", 0.0)), 2),
        "fps": round(1000.0 / inf, 1) if inf else "",
        "imgsz": imgsz,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="det=weights.pt pairs, e.g. yolo11=.../best.pt")
    ap.add_argument("--data", required=True, help="a dataset.yaml (for the real test split)")
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--out", type=Path, default=paths.TABLES / "efficiency.csv")
    args = ap.parse_args()

    rows = []
    for spec in args.runs:
        det, w = spec.split("=", 1)
        if not Path(w).exists():
            print(f"[eff] skip {det}: {w} not found"); continue
        row = measure(det, w, args.data, args.imgsz)
        rows.append(row); print("[eff]", row)
    if rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        print(f"[eff] wrote {args.out}")


if __name__ == "__main__":
    main()
