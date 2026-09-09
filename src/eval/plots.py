"""Figures for the paper (research questions).

Main figure: the hybrid ratio curve — detection metric vs synthetic ratio, with
the real-only baseline as a horizontal reference, one line per detector/generator,
mean±std shaded over seeds. Reads results/tables/metrics.csv.

Run:  python -m src.eval.plots --metric map50_95
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from ..common import paths


def _parse_rows(metric: str):
    """Yield (series, detector, ratio, seed, value) parsed from run names.

    Run names look like 'realonly_yolo11_real_seed0', 'hybrid_yolo11_composite_r100_seed0',
    'matched_yolo11_composite_r100_seed0'. We pull out ratio (`_r<pct>`), detector,
    and a series label (generator / matched / real) robustly.
    """
    from .metrics import load_all_rows

    rows = load_all_rows()
    if not rows:
        raise SystemExit("No metrics yet — run the sweep first.")
    for r in rows:
        name = r["run_name"]
        m_seed = re.search(r"_seed(\d+)$", name)
        if not m_seed:
            continue
        base = name[: m_seed.start()]
        m_r = re.search(r"_r(\d+)", base)
        ratio = int(m_r.group(1)) / 100.0 if m_r else 0.0
        m_det = re.search(r"(yolo11|yolo26|rtdetrv2|rtdetr)", base)
        det = m_det.group(1) if m_det else "det"
        m_gen = re.search(r"(composite|sd35|flux|gan)", base)
        series = m_gen.group(1) if m_gen else ("matched" if base.startswith("matched") else "real")
        yield series, det, ratio, int(m_seed.group(1)), float(r[metric])


def ratio_curve(metric: str = "map50_95", out: Path | None = None) -> Path:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError("pip install matplotlib") from exc

    # (gen, det, ratio) -> list of values across seeds
    series: dict[tuple, list[float]] = defaultdict(list)
    for gen, det, ratio, _seed, val in _parse_rows(metric):
        series[(gen, det, ratio)].append(val)

    lines: dict[tuple, list[tuple]] = defaultdict(list)
    for (gen, det, ratio), vals in series.items():
        lines[(gen, det)].append((ratio, float(np.mean(vals)), float(np.std(vals))))

    fig, ax = plt.subplots(figsize=(7, 5))
    for (gen, det), pts in sorted(lines.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"{gen}/{det}")

    ax.set_xlabel("synthetic ratio (fraction of real train count)")
    ax.set_ylabel(metric)
    ax.set_title("Detection vs synthetic ratio (mean ± std over seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    paths.ensure_dirs()
    out = out or (paths.FIGURES / f"ratio_curve_{metric}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Figure -> {out}")
    return out


def detector_bars(metric: str = "map50_95", out: Path | None = None) -> Path:
    """Grouped bar chart for the professor: per detector, real-only vs hybrid vs
    matched (mean ± std over seeds). Reads run names like
    realonly_yolo11, hybrid_composite_r50_yolo11, matched_composite_r50_yolo11."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError("pip install matplotlib") from exc

    from .metrics import load_all_rows
    # (detector, group) -> [values]; group in {real, hybrid, matched}
    vals: dict[tuple, list[float]] = defaultdict(list)
    for r in load_all_rows():
        name = r["run_name"]
        m = re.search(r"_seed\d+$", name)
        if not m:
            continue
        base = name[: m.start()]
        det = (re.search(r"(yolo11|yolo26|rtdetrv2|rtdetr)", base) or [None, "?"])[1] \
            if re.search(r"(yolo11|yolo26|rtdetrv2|rtdetr)", base) else "?"
        if base.startswith("realonly"):
            grp = "real-only"
        elif base.startswith("hybrid"):
            grp = "hybrid"
        elif base.startswith("matched"):
            grp = "matched"
        else:
            continue
        vals[(det, grp)].append(float(r[metric]))

    dets = sorted({d for d, _ in vals})
    groups = ["real-only", "hybrid", "matched"]
    colors = {"real-only": "#ff7f0e", "hybrid": "#1f77b4", "matched": "#7f7f7f"}
    x = np.arange(len(dets))
    w = 0.26

    fig, ax = plt.subplots(figsize=(8, 5))
    for gi, grp in enumerate(groups):
        means = [float(np.mean(vals.get((d, grp), [np.nan]))) for d in dets]
        errs = [float(np.std(vals.get((d, grp), [0.0]))) for d in dets]
        ax.bar(x + (gi - 1) * w, means, w, yerr=errs, capsize=4, label=grp, color=colors[grp])

    ax.set_xticks(x)
    ax.set_xticklabels(dets)
    ax.set_ylabel(metric)
    ax.set_title("Real vs composite-hybrid vs matched-size control (mean ± std over seeds)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    paths.ensure_dirs()
    out = out or (paths.FIGURES / f"detector_bars_{metric}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Figure -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="map50_95")
    ap.add_argument("--kind", default="bars", choices=["bars", "curve"],
                    help="bars = per-detector grouped bars (professor); curve = ratio sweep")
    args = ap.parse_args()
    (detector_bars if args.kind == "bars" else ratio_curve)(args.metric)
