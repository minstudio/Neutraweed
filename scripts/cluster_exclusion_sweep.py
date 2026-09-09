"""Cluster exclusion crossed with the two measurement adjustments.

Excluding densely clustered tiles is a request about the benchmark, not the
model: where plants overlap, "one plant" is ambiguous to annotate and therefore
ambiguous to score. The objection is that it discards evaluation data, and a
27-photo test field has little to spare.

But clustering is not independent of the other two known problems. Overlapping
plants are exactly where a box gets drawn around a clump instead of an
individual (the nutsedge problem), and dense patches are where annotation stops
before the crop row does (the scope problem). So the question is not only what
exclusion costs, but whether it still buys anything once those two are already
corrected. If it does not, exclusion is unnecessary rather than merely
expensive — which is a far stronger answer.

Rows are exclusion settings, columns are adjustments:

  none    AP against the annotations as they are.
  scope   background false positives credited as unannotated real plants,
          P = 0.95 verified, R = 0.75 assumed recall.
  extent  matched on intersection-over-GT-area, so a prediction containing the
          annotation counts however much larger it is.
  both    scope + extent.

  python scripts/cluster_exclusion_sweep.py \
      --runs 'hybrid_yolo26_sd35cut_lora_v2_r125_seed[0-9]'

One ap_analysis pass per cell: 5 exclusion settings x 4 adjustments = 20 passes.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNITS = re.compile(r"\[ap\]\s+(\d+)\s+\(tile, seed\) units")

ADJUST = [
    ("none", []),
    ("scope", ["--fp-bkg-verified", "0.95", "--fp-bkg-recall", "0.75"]),
    ("extent", ["--match-mode", "ioa"]),
    ("both", ["--fp-bkg-verified", "0.95", "--fp-bkg-recall", "0.75",
              "--match-mode", "ioa"]),
]


def one(runs, split, metric, boot, excl, adj, tmp):
    out = tmp / "r.csv"
    cmd = [sys.executable, str(HERE / "ap_analysis.py"),
           "--runs", runs, "--split", split, "--metric", metric,
           "--boot", str(boot), "--csv", str(out)] + list(excl) + list(adj)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not out.exists():
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit(f"ap_analysis failed for {excl} {adj}")
    m = UNITS.search(p.stdout)
    units = int(m.group(1)) if m else 0
    per, n_gt = {}, 0
    with open(out, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("size_bin") != "all" or "MINUS" in row.get("config", ""):
                continue
            if row["class"] == "MEAN":
                n_gt = int(row["n_gt"])
            if row.get(metric) not in (None, ""):
                per[row["class"]] = float(row[metric])
    out.unlink()
    return units, n_gt, per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--metric", default="map50", choices=["map50", "map50_95"])
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--overlap", type=float, nargs="*",
                    default=[1.0, 0.75, 0.50, 0.25, 0.10])
    ap.add_argument("--crowded", type=int, nargs="*", default=[])
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    rows = [("full set", [])]
    rows += [("overlap %g" % v, ["--drop-overlap", str(v)])
             for v in args.overlap if v < 1.0]
    rows += [("crowded %d" % v, ["--drop-crowded", str(v)])
             for v in args.crowded if v > 0]

    grid, meta = {}, {}
    with tempfile.TemporaryDirectory() as td:
        for rname, excl in rows:
            for aname, adj in ADJUST:
                print(f"  {rname} / {aname} ...", file=sys.stderr)
                u, g, per = one(args.runs, args.split, args.metric, args.boot,
                                excl, adj, Path(td))
                grid[(rname, aname)] = per
                if aname == "none":
                    meta[rname] = (u, g)

    bu, bg = meta["full set"]
    base = grid[("full set", "none")].get("MEAN", float("nan"))
    order = [c for c in ("SOLNI", "POROL", "SETVE", "CYPRO", "ECHCG")
             if c in grid[("full set", "none")]]

    print(f"\n{args.metric}, {args.split}, {args.runs}")
    print(f"full set = {bu} (tile,seed) units, {bg} instances\n")

    print("EXCLUSION COST vs ADJUSTMENT, mAP@50\n")
    h = f"{'setting':14s}{'units':>7s}{'inst':>8s}{'kept':>7s}"
    print(h + "".join(f"{a:>9s}" for a, _ in ADJUST) + f"{'both-none':>11s}")
    print("-" * (len(h) + 9 * len(ADJUST) + 11))
    for rname, _ in rows:
        u, g = meta[rname]
        line = f"{rname:14s}{u:7d}{g:8d}{g / max(bg, 1) * 100:6.0f}%"
        for aname, _ in ADJUST:
            line += f"{grid[(rname, aname)].get('MEAN', float('nan')):9.4f}"
        d = (grid[(rname, 'both')].get("MEAN", float("nan"))
             - grid[(rname, 'none')].get("MEAN", float("nan")))
        print(line + f"{d:+11.4f}")

    print("\n\nPER CLASS, with both adjustments applied\n")
    h2 = f"{'setting':14s}{'inst':>8s}{'kept':>7s}"
    print(h2 + "".join(f"{c:>9s}" for c in order) + f"{'MEAN':>9s}")
    print("-" * (len(h2) + 9 * (len(order) + 1)))
    for rname, _ in rows:
        u, g = meta[rname]
        per = grid[(rname, "both")]
        print(f"{rname:14s}{g:8d}{g / max(bg, 1) * 100:6.0f}%"
              + "".join(f"{per.get(c, float('nan')):9.3f}" for c in order)
              + f"{per.get('MEAN', float('nan')):9.3f}")

    ex = grid[("full set", "both")].get("MEAN", float("nan"))
    print(f"\nfull set, uncorrected            {base:.4f}")
    print(f"full set, both adjustments       {ex:.4f}   "
          f"({ex - base:+.4f}, no data discarded)")
    for rname, _ in rows[1:]:
        v = grid[(rname, "both")].get("MEAN", float("nan"))
        g = meta[rname][1]
        print(f"{rname + ', both adjustments':33s}{v:.4f}   ({v - ex:+.4f} "
              f"on top, at the cost of {100 - g / max(bg, 1) * 100:.0f}% of instances)")
    print("\nIf the last column adds little once both adjustments are in, the")
    print("clustering effect was already accounted for and exclusion is not")
    print("needed — which is a better answer than exclusion being too expensive.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["setting", "units", "instances", "instances_pct",
                        "adjustment", args.metric] + order)
            for rname, _ in rows:
                u, g = meta[rname]
                for aname, _ in ADJUST:
                    per = grid[(rname, aname)]
                    w.writerow([rname, u, g, round(g / max(bg, 1) * 100, 1),
                                aname, round(per.get("MEAN", float("nan")), 4)]
                               + [round(per.get(c, float("nan")), 4) for c in order])
        print(f"-> {args.csv}")


if __name__ == "__main__":
    main()
