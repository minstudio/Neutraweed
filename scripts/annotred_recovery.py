from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import paths

_WARNED = False

RUN_RE = re.compile(r"^annotred_(?P<arm>A|B|M|T(?:_.+)?)_(?P<det>[^_]+)_seed(?P<seed>\d+)$")

ARM_LABELS = {
    "A": "A  2021-only (lower bound)",
    "B": "B  +real 2022 Finca (oracle)",
    "M": "M  matched-size 2021 control",
    "T": "T  composite (2021-bg, confounded)",
}


def arm_label(arm: str) -> str:
    if arm in ARM_LABELS:
        return ARM_LABELS[arm]
    return f"T  {arm[2:]}"


def load(runs_dir: Path, metric: str, split: str) -> dict[str, dict[str, dict[int, float]]]:
    data: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for p in sorted(runs_dir.glob("*.csv")):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("split") != split:
                    continue
                m = RUN_RE.match(r["run_name"])
                if not m:
                    continue
                data[m["det"]][m["arm"]][int(m["seed"])] = float(r[metric])
    return data


def mean_std(vals: list[float]) -> tuple[float, float]:
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def paired(a: dict[int, float], b: dict[int, float]) -> dict | None:
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    xa = [a[s] for s in seeds]
    xb = [b[s] for s in seeds]
    t_p = w_p = float("nan")
    try:
        from scipy import stats as sps

        t_p = float(sps.ttest_rel(xa, xb).pvalue)
        try:
            w_p = float(sps.wilcoxon(xa, xb).pvalue)
        except ValueError:
            pass
    except ImportError:
        global _WARNED
        if not _WARNED:
            print("[warn] scipy missing; p-values reported as nan", file=sys.stderr)
            _WARNED = True
    diffs = [x - y for x, y in zip(xa, xb)]
    return {"n": len(seeds), "mean_diff": statistics.mean(diffs), "t_p": t_p, "w_p": w_p}


def per_seed_recovery(t: dict[int, float], a: dict[int, float], b: dict[int, float]) -> list[float]:
    out = []
    for s in sorted(set(t) & set(a) & set(b)):
        den = b[s] - a[s]
        if abs(den) > 1e-9:
            out.append((t[s] - a[s]) / den)
    return out


def report(data: dict, metric: str, split: str, out_csv: Path) -> None:
    if not data:
        raise SystemExit("No annotred_* rows found — check --runs-dir and that eval.metrics ran.")
    csv_rows = []
    for det in sorted(data):
        arms = data[det]
        t_arms = sorted(k for k in arms if k.startswith("T"))
        print(f"\n=== {det}  ({metric}, split={split}) ===")
        print(f"{'arm':<42}{'seeds':>6}{'mean':>9}{'std':>9}")
        for arm in ["A", "B", "M"] + t_arms:
            if arm not in arms:
                print(f"{arm_label(arm):<42}{'-':>6}   MISSING")
                continue
            vals = list(arms[arm].values())
            mu, sd = mean_std(vals)
            print(f"{arm_label(arm):<42}{len(vals):>6}{mu:>9.4f}{sd:>9.4f}")

        if "A" not in arms or "B" not in arms:
            print("  [recovery skipped: need arms A and B]")
            continue
        mA, _ = mean_std(list(arms["A"].values()))
        mB, _ = mean_std(list(arms["B"].values()))
        gap = mB - mA
        print(f"\n  oracle gap B-A = {gap:+.4f}")
        if abs(gap) < 1e-9:
            print("  [recovery undefined: B == A]")
            continue

        print(f"\n  recovery (T-A)/(B-A):")
        for t in t_arms:
            mT, _ = mean_std(list(arms[t].values()))
            rec = (mT - mA) / gap
            ps = per_seed_recovery(arms[t], arms["A"], arms["B"])
            ps_str = ""
            if ps:
                pm, psd = mean_std(ps)
                ps_str = f"   per-seed {pm*100:6.1f}% +/- {psd*100:.1f}% (n={len(ps)})"
            print(f"    {arm_label(t):<40} mean-based {rec*100:6.1f}%{ps_str}")

            row = {
                "detector": det, "t_arm": t, "metric": metric, "split": split,
                "mean_A": round(mA, 4), "mean_B": round(mB, 4),
                "mean_M": round(mean_std(list(arms["M"].values()))[0], 4) if "M" in arms else "",
                "mean_T": round(mT, 4), "recovery": round(rec, 4),
                "recovery_per_seed_mean": round(mean_std(ps)[0], 4) if ps else "",
                "recovery_per_seed_std": round(mean_std(ps)[1], 4) if ps else "",
                "n_seeds_T": len(arms[t]),
            }
            for ref in ("A", "M", "B"):
                p = paired(arms[t], arms[ref]) if ref in arms else None
                row[f"diff_vs_{ref}"] = round(p["mean_diff"], 4) if p else ""
                row[f"t_p_vs_{ref}"] = round(p["t_p"], 4) if p else ""
                row[f"wilcoxon_p_vs_{ref}"] = round(p["w_p"], 4) if p else ""
            csv_rows.append(row)

        print(f"\n  paired significance (t-test / Wilcoxon over shared seeds):")
        for t in t_arms:
            for ref in ("A", "M", "B"):
                if ref not in arms:
                    continue
                p = paired(arms[t], arms[ref])
                if p is None:
                    print(f"    {t} vs {ref}: <2 shared seeds, skipped")
                    continue
                print(f"    {t:<28} vs {ref}:  diff={p['mean_diff']:+.4f}  "
                      f"t_p={p['t_p']:.4f}  wilcoxon_p={p['w_p']:.4f}  (n={p['n']})")

    if csv_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n[csv] {out_csv}")
    print("\nNote: with 3 seeds the Wilcoxon floor is p=0.25 and the t-test has 2 df; "
          "read effect sizes and seed-consistency, not p-values alone.")


def load_perclass(runs_dir: Path, metric: str, split: str):
    """det -> cls -> arm -> {seed: value}, from runs_perclass/ (present classes only)."""
    data: dict[str, dict[str, dict[str, dict[int, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    for p in sorted(runs_dir.glob("*.csv")):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("split") != split or r.get("present_in_test") == "0":
                    continue
                m = RUN_RE.match(r["run_name"])
                if not m:
                    continue
                v = r[metric]
                if v == "" or v != v:
                    continue
                data[m["det"]][r["class"]][m["arm"]][int(m["seed"])] = float(v)
    return data


def report_perclass(data, metric: str, split: str, out_csv: Path) -> None:
    if not data:
        raise SystemExit(
            "No per-class rows — run scripts/eval_perclass.py (GPU/sbatch) first.")
    csv_rows = []
    for det in sorted(data):
        print(f"\n=== {det}  per-class recovery ({metric}, split={split}) ===")
        for cls in sorted(data[det]):
            arms = data[det][cls]
            if "A" not in arms or "B" not in arms:
                continue
            mA = mean_std(list(arms["A"].values()))[0]
            mB = mean_std(list(arms["B"].values()))[0]
            gap = mB - mA
            t_arms = sorted(k for k in arms if k.startswith("T"))
            print(f"\n  [{cls}]  A={mA:.3f}  B={mB:.3f}  gap={gap:+.3f}")
            for t in t_arms:
                mT = mean_std(list(arms[t].values()))[0]
                rec = ((mT - mA) / gap) if abs(gap) > 1e-9 else float("nan")
                rec_str = f"{rec*100:6.1f}%" if rec == rec else "   n/a"
                print(f"    {t:<26} T={mT:.3f}  recovery {rec_str}")
                csv_rows.append({
                    "detector": det, "class": cls, "t_arm": t, "metric": metric,
                    "split": split, "mean_A": round(mA, 4), "mean_B": round(mB, 4),
                    "mean_T": round(mT, 4), "gap_B_A": round(gap, 4),
                    "recovery": round(rec, 4) if rec == rec else "",
                })
    if csv_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n[csv] {out_csv}")
    print("\nHypothesis check: if composite2022's aggregate gain is ECHCG-driven, "
          "ECHCG recovery should dominate and the other four stay ~0.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Annotation-reduction recovery readout: per detector, the "
                    "A/B/T(pools)/M table with (T-A)/(B-A) recovery and paired "
                    "significance of each T variant vs A, M, B. Reads the per-run "
                    "CSVs from src.eval.metrics. With --per-class, reads "
                    "runs_perclass/ and breaks recovery out per weed species.")
    ap.add_argument("--metric", default="map50_95")
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--per-class", action="store_true")
    args = ap.parse_args()
    if args.per_class:
        runs = args.runs_dir or paths.TABLES / "runs_perclass"
        out = args.out or paths.TABLES / "annotred_recovery_perclass.csv"
        report_perclass(load_perclass(runs, args.metric, args.split),
                        args.metric, args.split, out)
    else:
        runs = args.runs_dir or paths.TABLES / "runs"
        out = args.out or paths.TABLES / "annotred_recovery.csv"
        report(load(runs, args.metric, args.split), args.metric, args.split, out)


if __name__ == "__main__":
    main()
