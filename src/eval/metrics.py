"""Detection metrics on the real test set.

Wraps Ultralytics `model.val(split='test')` and extracts the headline numbers
into a flat dict, appending a row to results/tables/metrics.csv so every run is
comparable. Always evaluate on the REAL test split (the dataset.yaml test entry
is real by construction — Stage D enforces this).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..common import paths


def evaluate(weights: Path, data_yaml: Path, run_name: str, split: str = "test") -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("pip install ultralytics") from exc

    model = YOLO(str(weights))
    res = model.val(data=str(data_yaml), split=split, verbose=False)

    box = res.box
    precision = float(box.mp)        # mean precision
    recall = float(box.mr)           # mean recall
    map50 = float(box.map50)
    map5095 = float(box.map)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    row = {
        "run_name": run_name,
        "split": split,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "map50": round(map50, 4),
        "map50_95": round(map5095, 4),
    }
    _append_csv(row)
    print(row)
    return row


RUNS_DIR = paths.TABLES / "runs"


def _append_csv(row: dict) -> None:
    """Write a per-run result file (race-free for parallel SLURM-array jobs;
    each task writes its own file, so concurrent evals never corrupt a shared
    CSV). Aggregate with load_all_rows()."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{row['run_name']}_{row['split']}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def load_all_rows() -> list[dict]:
    """All result rows, from per-run files (preferred) or a legacy metrics.csv."""
    rows: list[dict] = []
    if RUNS_DIR.exists():
        for p in sorted(RUNS_DIR.glob("*.csv")):
            rows += list(csv.DictReader(open(p, encoding="utf-8")))
    if rows:
        return rows
    legacy = paths.TABLES / "metrics.csv"
    return list(csv.DictReader(open(legacy, encoding="utf-8"))) if legacy.exists() else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    evaluate(args.weights, args.data, args.name, args.split)
