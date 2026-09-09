from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import classes, paths
from annotred_recovery import load, load_perclass, mean_std

ARM_ORDER = ["A", "M", "T", "T_composite2022", "T_composite2022strict", "B"]
ARM_SHORT = {
    "A": "A\n2021-only",
    "M": "M\nmatched",
    "T": "T\ncomp.\n2021bg",
    "T_composite2022": "T\ncomp.\n2022",
    "T_composite2022strict": "T\nstrict",
    "B": "B\n+real2022",
}
ARM_COLOR = {
    "A": "#9e9e9e", "M": "#bdbdbd", "T": "#c6a15b",
    "T_composite2022": "#2e7d32", "T_composite2022strict": "#81c784", "B": "#1565c0",
}


def fig_arms(data, metric: str, out: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    dets = sorted(data)
    fig, axes = plt.subplots(1, len(dets), figsize=(5.2 * len(dets), 4.6), squeeze=False)
    for ax, det in zip(axes[0], dets):
        arms = data[det]
        present = [a for a in ARM_ORDER if a in arms]
        means, stds, colors, labels = [], [], [], []
        for a in present:
            mu, sd = mean_std(list(arms[a].values()))
            means.append(mu); stds.append(sd)
            colors.append(ARM_COLOR[a]); labels.append(ARM_SHORT[a])
        x = np.arange(len(present))
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.4)
        if "A" in arms and "B" in arms:
            a0 = mean_std(list(arms["A"].values()))[0]
            b0 = mean_std(list(arms["B"].values()))[0]
            ax.axhline(a0, ls=":", c="#616161", lw=1)
            ax.axhline(b0, ls="--", c="#1565c0", lw=1)
            ax.axhspan(a0, b0, color="#1565c0", alpha=0.05)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(det, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", ls=":", alpha=0.4)
    fig.suptitle("Annotation-reduction arms — test mAP@50:95 (mean ± std, 3 seeds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def fig_recovery(data, metric: str, out: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    dets = sorted(data)
    t_variants = ["T", "T_composite2022", "T_composite2022strict"]
    fig, ax = plt.subplots(figsize=(1.6 + 2.0 * len(dets), 4.6))
    width = 0.24
    x = np.arange(len(dets))
    for i, tv in enumerate(t_variants):
        vals, errs = [], []
        for det in dets:
            arms = data[det]
            if tv not in arms or "A" not in arms or "B" not in arms:
                vals.append(np.nan); errs.append(0); continue
            mA = mean_std(list(arms["A"].values()))[0]
            mB = mean_std(list(arms["B"].values()))[0]
            gap = mB - mA
            per_seed = []
            for s in set(arms[tv]) & set(arms["A"]) & set(arms["B"]):
                d = arms["B"][s] - arms["A"][s]
                if abs(d) > 1e-9:
                    per_seed.append((arms[tv][s] - arms["A"][s]) / d * 100)
            vals.append(((mean_std(list(arms[tv].values()))[0] - mA) / gap * 100) if abs(gap) > 1e-9 else np.nan)
            errs.append(mean_std(per_seed)[1] if len(per_seed) > 1 else 0)
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=3,
               label=ARM_SHORT[tv].replace("\n", " "), color=ARM_COLOR[tv],
               edgecolor="black", linewidth=0.4)
    ax.axhline(0, c="#616161", lw=1, ls=":")
    ax.axhline(100, c="#1565c0", lw=1, ls="--")
    ax.text(ax.get_xlim()[1], 100, " B (oracle)", va="center", fontsize=8, c="#1565c0")
    ax.set_xticks(x); ax.set_xticklabels(dets)
    ax.set_ylabel("recovery  (T−A)/(B−A)  [%]")
    ax.set_title("Recovery of the oracle gap  (100% = real-2022 annotation)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def fig_perclass(pc, metric: str, out: Path, t_arm: str):
    import matplotlib.pyplot as plt
    import numpy as np

    dets = sorted(pc)
    order = classes.CLASS_NAMES
    fig, ax = plt.subplots(figsize=(1.6 + 1.9 * len(dets), 4.6))
    width = 0.8 / max(len(dets), 1)
    x = np.arange(len(order))
    for i, det in enumerate(dets):
        vals = []
        for cls in order:
            arms = pc[det].get(cls, {})
            if t_arm not in arms or "A" not in arms or "B" not in arms:
                vals.append(np.nan); continue
            mA = mean_std(list(arms["A"].values()))[0]
            mB = mean_std(list(arms["B"].values()))[0]
            mT = mean_std(list(arms[t_arm].values()))[0]
            gap = mB - mA
            vals.append((mT - mA) / gap * 100 if abs(gap) > 1e-9 else np.nan)
        ax.bar(x + (i - (len(dets) - 1) / 2) * width, vals, width,
               label=det, edgecolor="black", linewidth=0.4)
    ax.axhline(0, c="#616161", lw=1, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("per-class recovery [%]")
    ax.set_title(f"Per-class recovery — {t_arm}  (ECHCG is the 2022-heavy class)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Professor figures for the annotation-reduction result: arms "
                    "bar chart, recovery-fraction bars, and (if per-class CSVs "
                    "exist) the per-class recovery panel. Reads the per-run CSVs "
                    "used by annotred_recovery.py. Login-node safe (matplotlib/CPU).")
    ap.add_argument("--metric", default="map50_95")
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs-dir", type=Path, default=paths.TABLES / "runs")
    ap.add_argument("--perclass-dir", type=Path, default=paths.TABLES / "runs_perclass")
    ap.add_argument("--outdir", type=Path, default=paths.FIGURES / "annotred")
    ap.add_argument("--perclass-arm", default="T_composite2022")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        raise SystemExit("pip install matplotlib")

    data = load(args.runs_dir, args.metric, args.split)
    if not data:
        raise SystemExit(f"No annotred_* rows in {args.runs_dir}")
    fig_arms(data, args.metric, args.outdir / f"arms_{args.split}.png")
    fig_recovery(data, args.metric, args.outdir / f"recovery_{args.split}.png")

    pc = load_perclass(args.perclass_dir, args.metric, args.split)
    if pc:
        fig_perclass(pc, args.metric, args.outdir / f"perclass_{args.perclass_arm}_{args.split}.png",
                     args.perclass_arm)
    else:
        print(f"[skip] no per-class CSVs in {args.perclass_dir} "
              f"(run scripts/eval_perclass.py first for the per-class figure)")


if __name__ == "__main__":
    main()
