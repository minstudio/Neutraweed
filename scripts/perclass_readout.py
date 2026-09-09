from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import classes, paths

RUNS_PC = paths.TABLES / "runs_perclass"


def _read_one(path: Path, metric: str) -> dict[str, float]:
    out = {}
    for r in csv.DictReader(open(path)):
        try:
            out[r["class"]] = float(r[metric])
        except (ValueError, KeyError):
            out[r["class"]] = float("nan")
    return out


def load(run: str, split: str, metric: str, avg: bool) -> tuple[dict[str, float], int] | None:
    """Return (per-class metric, n_seeds). If avg, average every <base>_seed*.csv,
    where base = run with any trailing _seedN removed."""
    if avg:
        base = re.sub(r"_seed\d+$", "", run)
        files = sorted(RUNS_PC.glob(f"{base}_seed*_{split}.csv"))
        if not files:
            f = RUNS_PC / f"{run}_{split}.csv"
            files = [f] if f.exists() else []
        if not files:
            return None
        dicts = [_read_one(f, metric) for f in files]
        out = {}
        for c in dicts[0]:
            vals = [d[c] for d in dicts if d.get(c) == d.get(c)]
            out[c] = st.mean(vals) if vals else float("nan")
        return out, len(files)
    f = RUNS_PC / f"{run}_{split}.csv"
    if not f.exists():
        return None
    return _read_one(f, metric), 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-class AP table from results/tables/runs_perclass/. "
                    "Pass run names to compare (baseline first).")
    ap.add_argument("runs", nargs="+", help="run names (with or without _seedN)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--metric", default="map50", choices=["map50", "map50_95"])
    ap.add_argument("--avg-seeds", action="store_true",
                    help="average every _seed* csv sharing each run's base name")
    args = ap.parse_args()

    loaded = {r: load(r, args.split, args.metric, args.avg_seeds) for r in args.runs}
    missing = [r for r, d in loaded.items() if d is None]
    if missing:
        print("MISSING csv for:", ", ".join(missing))
    runs = [r for r in args.runs if loaded[r] is not None]
    if not runs:
        raise SystemExit("no matching per-class CSVs found")
    data = {r: loaded[r][0] for r in runs}
    nseed = {r: loaded[r][1] for r in runs}

    w = 22
    label = f"{args.metric}" + (" (seed-avg)" if args.avg_seeds else "")
    print(f"\nPer-class {label} (split={args.split})")
    header = f"{'class':<34}" + "".join(f"{r[:w]:>{w+2}}" for r in runs)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    base = runs[0]
    for name in classes.CLASS_NAMES:
        disp = classes.CLASS_DISPLAY.get(name, name)
        row = f"{disp:<34}"
        b = data[base].get(name, float("nan"))
        for i, r in enumerate(runs):
            v = data[r].get(name, float("nan"))
            if i == 0:
                row += f"{v:>{w+2}.3f}"
            else:
                row += f"{v:>{w-6}.3f}{v-b:>+8.3f}"
        print(row)
    print("-" * len(header))
    for r in runs:
        vals = [data[r][n] for n in classes.CLASS_NAMES
                if data[r].get(n) == data[r].get(n)]
        m = sum(vals) / len(vals) if vals else float("nan")
        print(f"  mean {args.metric} ({r}): {m:.4f}  over {len(vals)} classes"
              f"  [{nseed[r]} seed(s)]")
    print()


if __name__ == "__main__":
    main()
