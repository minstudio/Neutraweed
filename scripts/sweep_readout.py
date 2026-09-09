"""Ratio-sweep + generator-comparison readout.

Reads results/tables/runs/*.csv and, per detector, prints the real-only baseline
and each generator's hybrid curve across ratios: mean+/-std over seeds, delta vs
real-only, the matched-size control (is it diversity not volume?), and paired
significance. Flags the best ratio per generator and a plain verdict. Login-safe.

  python scripts/sweep_readout.py                 # all generators found
  python scripts/sweep_readout.py --metric map50
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import paths

DETS = ("yolo11", "yolo26", "rtdetrv2")
# A crowding-filtered run carries an _x<NN> suffix (build_hybrid --max-overlap).
# It is scored on a different test set, so it is reported as its own benchmark
# rather than mixed in with the unfiltered runs — including its own real-only
# baseline, since comparing a filtered hybrid against an unfiltered baseline
# would attribute the filter's effect to the synthetic data.
RE_REAL = re.compile(r"^realonly_(?P<det>yolo11|yolo26|rtdetrv2)"
                     r"(?:_real)?(?:_(?P<flt>x\d+))?$")
RE_HYB = re.compile(r"^hybrid_(?P<det>yolo11|yolo26|rtdetrv2)_(?P<gen>.+?)"
                    r"_r(?P<pct>\d+)(?:_(?P<flt>x\d+))?$")
RE_MAT = re.compile(r"^matched_(?P<det>yolo11|yolo26|rtdetrv2)_(?P<gen>.+?)"
                    r"_r(?P<pct>\d+)(?:_(?P<flt>x\d+))?$")


def _key(mm) -> str:
    """Detector key, widened by the crowding filter when one is in use."""
    f = mm.groupdict().get("flt")
    return mm["det"] + (f" [{f}]" if f else "")


def load(runs_dir: Path, metric: str, split: str):
    # real[det]={seed:v}; hyb[det][gen][pct]={seed:v}; mat[det][gen][pct]={seed:v}
    real = defaultdict(dict)
    hyb = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    mat = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for p in sorted(runs_dir.glob("*.csv")):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("split") != split:
                continue
            m = re.match(r"^(.*)_seed(\d+)$", r["run_name"])
            if not m:
                continue
            base, seed = m.group(1), int(m.group(2))
            v = r.get(metric)
            if v in (None, ""):
                continue
            v = float(v)
            mm = RE_REAL.match(base)
            if mm:
                real[_key(mm)][seed] = v; continue
            mm = RE_HYB.match(base)
            if mm:
                hyb[_key(mm)][mm["gen"]][int(mm["pct"])][seed] = v; continue
            mm = RE_MAT.match(base)
            if mm:
                mat[_key(mm)][mm["gen"]][int(mm["pct"])][seed] = v
    return real, hyb, mat


def ms(d):
    v = list(d.values())
    return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0, len(v))


def paired_p(a, b):
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return float("nan")
    try:
        from scipy import stats as s
        return float(s.ttest_rel([a[k] for k in seeds], [b[k] for k in seeds]).pvalue)
    except Exception:
        return float("nan")


def report(real, hyb, mat, metric, split):
    dets = [d for d in DETS if d in real or d in hyb]
    dets += sorted(k for k in set(real) | set(hyb) if k not in dets)
    if not dets:
        raise SystemExit("No sweep rows found — check results/tables/runs/.")
    for det in dets:
        rmean = ms(real[det]) if det in real and real[det] else None
        print(f"\n=== {det}  ({metric}, split={split}) ===")
        if rmean:
            print(f"real-only baseline: {rmean[0]:.4f} +/- {rmean[1]:.4f}  (n={rmean[2]})")
        else:
            print("real-only baseline: MISSING")
        for gen in sorted(hyb.get(det, {})):
            print(f"\n  generator: {gen}")
            print(f"  {'ratio':>6}{'hybrid':>18}{'d vs real':>11}{'matched':>16}"
                  f"{'hyb-matched':>13}{'p(real)':>9}")
            best = None
            for pct in sorted(hyb[det][gen]):
                hm = ms(hyb[det][gen][pct])
                dvr = (hm[0] - rmean[0]) if rmean else float("nan")
                mrow = mat.get(det, {}).get(gen, {}).get(pct)
                mstr = f"{ms(mrow)[0]:.4f}" if mrow else "  -"
                hmm = (hm[0] - ms(mrow)[0]) if mrow else float("nan")
                pv = paired_p(hyb[det][gen][pct], real[det]) if rmean else float("nan")
                print(f"  {pct/100:>6.2f}{hm[0]:>10.4f}+/-{hm[1]:.3f}{dvr:>+11.4f}"
                      f"{mstr:>16}{hmm:>+13.4f}{pv:>9.3f}")
                if rmean and (best is None or hm[0] > best[1]):
                    best = (pct, hm[0], dvr, hmm, pv)
            if best:
                verdict = ("HELPS" if best[2] > 0 else "no gain")
                vol = ("(diversity, beats matched)" if best[3] == best[3] and best[3] > 0
                       else "(<= matched: volume, not diversity)" if best[3] == best[3]
                       else "")
                print(f"  -> best r={best[0]/100:.2f}: {best[1]:.4f} "
                      f"({best[2]:+.4f} vs real, p={best[4]:.3f}) {verdict} {vol}")
    # generator head-to-head at each ratio
    print("\n=== generator head-to-head (hybrid mean, per detector/ratio) ===")
    for det in dets:
        gens = sorted(hyb.get(det, {}))
        if len(gens) < 2:
            continue
        pcts = sorted({p for g in gens for p in hyb[det][g]})
        print(f"\n  {det}: " + "  ".join(f"{g}" for g in gens))
        for pct in pcts:
            cells = [f"{g}={ms(hyb[det][g][pct])[0]:.4f}" if pct in hyb[det][g] else f"{g}=-" for g in gens]
            print(f"    r{pct/100:.2f}: " + "  ".join(cells))
    print("\nNote: 3 seeds -> paired t has 2 df; read effect sizes + seed consistency, "
          "not p alone. 'hyb-matched'>0 means the gain is synthetic diversity, not count.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="map50_95")
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs-dir", type=Path, default=paths.TABLES / "runs")
    args = ap.parse_args()
    real, hyb, mat = load(args.runs_dir, args.metric, args.split)
    report(real, hyb, mat, args.metric, args.split)


if __name__ == "__main__":
    main()
