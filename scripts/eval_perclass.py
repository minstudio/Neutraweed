from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import classes, paths

RUNS_PC = paths.TABLES / "runs_perclass"


def _loader(run_name: str):
    from ultralytics import RTDETR, YOLO

    return RTDETR if "rtdetrv2" in run_name or "rtdetr" in run_name else YOLO


def eval_one(ckpt: Path, data_yaml: Path, run_name: str, split: str) -> list[dict]:
    Model = _loader(run_name)
    res = Model(str(ckpt)).val(data=str(data_yaml), split=split, verbose=False)
    box = res.box

    maps = list(getattr(box, "maps", []))          # map50-95 per class, len == nc
    ap_idx = list(getattr(box, "ap_class_index", []))
    ap50 = list(getattr(box, "ap50", []))          # only classes present in test
    ap = list(getattr(box, "ap", []))
    ap50_by_cls = {int(c): float(ap50[i]) for i, c in enumerate(ap_idx)} if ap50 else {}
    ap_by_cls = {int(c): float(ap[i]) for i, c in enumerate(ap_idx)} if ap else {}

    rows = []
    for cid, name in enumerate(classes.CLASS_NAMES):
        m5095 = float(maps[cid]) if cid < len(maps) else ap_by_cls.get(cid, float("nan"))
        rows.append({
            "run_name": run_name,
            "split": split,
            "class_id": cid,
            "class": name,
            "present_in_test": int(cid in ap_idx),
            "map50": round(ap50_by_cls.get(cid, float("nan")), 4),
            "map50_95": round(m5095, 4),
        })
    return rows


def write_rows(rows: list[dict], run_name: str, split: str) -> Path:
    RUNS_PC.mkdir(parents=True, exist_ok=True)
    out = RUNS_PC / f"{run_name}_{split}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out


def dataset_for(run_name: str) -> Path:
    import re

    base = re.sub(r"_seed\d+$", "", run_name)          # drop the seed suffix
    # sweep convention: the dataset dir IS the run name minus the seed
    # (e.g. hybrid_yolo26_sd35cut_lora_v2_r125).
    cand = paths.DATASETS / base / "dataset.yaml"
    if cand.exists():
        return cand
    # annotred / real-only convention: dataset == the prefix before _<detector>
    # (e.g. annotred_A_rtdetrv2 -> annotred_A).
    stem = base
    for det in ("_yolo11", "_yolo26", "_rtdetrv2"):
        i = stem.find(det)
        if i != -1:
            stem = stem[:i]
            break
    return paths.DATASETS / stem / "dataset.yaml"


def discover(pattern: str) -> list[tuple[str, Path]]:
    out = []
    for d in sorted(paths.CHECKPOINTS.glob(pattern)):
        best = d / "weights" / "best.pt"
        if best.exists():
            out.append((d.name, best))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-class AP for annotation-reduction checkpoints. Re-runs "
                    "model.val on each best.pt (GPU — sbatch, not login) and "
                    "writes per-class rows to results/tables/runs_perclass/. "
                    "Aggregate with scripts/annotred_recovery.py --per-class.")
    ap.add_argument("--pattern", default="annotred_*",
                    help="checkpoint dir glob under results/checkpoints/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", type=Path, default=None,
                    help="override dataset.yaml (default: inferred per run)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    runs = discover(args.pattern)
    if not runs:
        raise SystemExit(f"No checkpoints with best.pt match {args.pattern}")
    print(f"[eval_perclass] {len(runs)} checkpoints, split={args.split}")
    for run_name, ckpt in runs:
        out = RUNS_PC / f"{run_name}_{args.split}.csv"
        if args.skip_existing and out.exists():
            print(f"  skip {run_name} (exists)")
            continue
        dy = args.data or dataset_for(run_name)
        if not dy.exists():
            print(f"  WARN {run_name}: dataset.yaml not found at {dy}; skipping")
            continue
        rows = eval_one(ckpt, dy, run_name, args.split)
        write_rows(rows, run_name, args.split)
        present = sum(r["present_in_test"] for r in rows)
        print(f"  {run_name}: wrote {len(rows)} classes ({present} present) -> {out.name}")


if __name__ == "__main__":
    main()
