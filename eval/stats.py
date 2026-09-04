"""Paired significance tests across seeds (CLAUDE.md §6/§8).

"Report variance over seeds; 1-2 mAP points is often noise." Given two
configurations each run over the same set of seeds, this pairs them by seed and
runs a paired t-test and a Wilcoxon signed-rank test on the chosen metric.

Reads results/tables/metrics.csv produced by eval.metrics.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from ..common import paths


def _load_metric(metric: str) -> dict[str, dict[int, float]]:
    """config_name (seed stripped) -> {seed: metric_value}."""
    from .metrics import load_all_rows

    out: dict[str, dict[int, float]] = defaultdict(dict)
    rows = load_all_rows()
    if not rows:
        raise SystemExit("No metrics yet — run eval.metrics (or the sweep) first.")
    for r in rows:
        m = re.match(r"^(.*)_seed(\d+)$", r["run_name"])
        if not m:
            continue
        cfg, seed = m.group(1), int(m.group(2))
        out[cfg][seed] = float(r[metric])
    return out


def summary(metric: str = "map50_95") -> dict[str, dict]:
    import statistics as st

    data = _load_metric(metric)
    rows = {}
    for cfg, by_seed in sorted(data.items()):
        vals = list(by_seed.values())
        rows[cfg] = {
            "n_seeds": len(vals),
            "mean": round(st.mean(vals), 4),
            "std": round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        }
    return rows


def compare(cfg_a: str, cfg_b: str, metric: str = "map50_95") -> dict:
    try:
        from scipy import stats as sps
    except ImportError as exc:
        raise ImportError("pip install scipy") from exc

    data = _load_metric(metric)
    a, b = data.get(cfg_a, {}), data.get(cfg_b, {})
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        raise SystemExit(f"Need >=2 shared seeds; got {seeds}")
    xa = [a[s] for s in seeds]
    xb = [b[s] for s in seeds]

    t = sps.ttest_rel(xa, xb)
    try:
        w = sps.wilcoxon(xa, xb)
        w_p = float(w.pvalue)
    except ValueError:
        w_p = float("nan")  # all-zero differences

    diffs = [x - y for x, y in zip(xa, xb)]
    result = {
        "metric": metric,
        "seeds": seeds,
        "mean_a": round(sum(xa) / len(xa), 4),
        "mean_b": round(sum(xb) / len(xb), 4),
        "mean_diff": round(sum(diffs) / len(diffs), 4),
        "paired_t_p": float(t.pvalue),
        "wilcoxon_p": w_p,
    }
    print(result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="map50_95")
    ap.add_argument("--compare", nargs=2, metavar=("CFG_A", "CFG_B"))
    args = ap.parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1], args.metric)
    else:
        import json
        print(json.dumps(summary(args.metric), indent=2))
