"""Per-class AP@50 under each measurement adjustment, side by side.

The adjustments are not corrections to the model. They are answers to "what
would this number be if the metric and the annotations agreed", and each rests
on a different assumption, so they belong in one table where the assumptions are
visible rather than quoted one at a time.

  uncorrected   AP against the annotations as they are.
  scope lo/est/hi
                credits background false positives (max IoU < 0.1 against every
                GT box of every class) as unannotated real plants. P = fraction
                a blind sample verified as real, R = assumed recall on those
                plants. Both assumed, hence a range.
  box extent    matches on intersection-over-GT-area rather than IoU, so a
                prediction containing the GT box counts however much larger it
                is. Addresses annotations that split one clumped plant into
                parts, which is the nutsedge problem.
  both          scope + box extent together.

  python scripts/adjusted_perclass.py \
      --runs 'hybrid_yolo26_sd35cut_lora_v2_r125_seed[0-9]'

Needs cached predictions (scripts/slurm/cache_preds.slurm). Runs ap_analysis
once per column, so give it a few minutes with --boot 0 (default here).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# label -> extra ap_analysis flags. Order is the column order.
VARIANTS = [
    ("uncorrected", []),
    ("scope lo\nP=.50 R=.75", ["--fp-bkg-verified", "0.50", "--fp-bkg-recall", "0.75"]),
    ("scope est\nP=.95 R=.75", ["--fp-bkg-verified", "0.95", "--fp-bkg-recall", "0.75"]),
    ("scope hi\nP=.95 R=1.0", ["--fp-bkg-verified", "0.95", "--fp-bkg-recall", "1.0"]),
    ("box extent\nIoA match", ["--match-mode", "ioa"]),
    ("both\nest + IoA", ["--fp-bkg-verified", "0.95", "--fp-bkg-recall", "0.75",
                         "--match-mode", "ioa"]),
]


def run(runs: str, split: str, metric: str, boot: int, extra, tmp: Path):
    out = tmp / "r.csv"
    cmd = [sys.executable, str(HERE / "ap_analysis.py"),
           "--runs", runs, "--split", split, "--metric", metric,
           "--boot", str(boot), "--csv", str(out)] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not out.exists():
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit(f"ap_analysis failed for flags {extra}")
    got = {}
    with open(out, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # 'all' is always emitted whatever --bins is set to; the size-
            # stratified rows belong in ap_analysis' own output, not here.
            if (row.get("size_bin") == "all" and row.get("class")
                    and row.get(metric) not in (None, "")
                    and "MINUS" not in row.get("config", "")):
                got[row["class"]] = float(row[metric])
    out.unlink()
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--metric", default="map50", choices=["map50", "map50_95"])
    ap.add_argument("--boot", type=int, default=0,
                    help="bootstrap resamples per column; 0 for speed, the "
                         "point of this table is the comparison across columns")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    cols = []
    with tempfile.TemporaryDirectory() as td:
        for label, extra in VARIANTS:
            print(f"  running {label.replace(chr(10), ' ')} ...", file=sys.stderr)
            cols.append((label, run(args.runs, args.split, args.metric,
                                    args.boot, extra, Path(td))))

    order = [c for c in ("SOLNI", "POROL", "SETVE", "CYPRO", "ECHCG")
             if c in cols[0][1]]
    order += [c for c in cols[0][1] if c not in order and c != "MEAN"]
    order.append("MEAN")

    w = 13
    head2 = [l.split("\n")[1] if "\n" in l else "" for l, _ in cols]
    print(f"\n{args.metric} per class, {args.split}, {args.runs}\n")
    print(f"{'':9s}" + "".join(f"{l.split(chr(10))[0]:>{w}s}" for l, _ in cols))
    print(f"{'':9s}" + "".join(f"{h:>{w}s}" for h in head2))
    print("-" * (9 + w * len(cols)))
    for c in order:
        base = cols[0][1].get(c)
        line = f"{c:9s}"
        for _, d in cols:
            v = d.get(c)
            line += f"{'-':>{w}s}" if v is None else f"{v:>{w}.3f}"
        if base is not None and cols[-1][1].get(c) is not None:
            line += f"   {cols[-1][1][c] - base:+.3f}"
        print(("-" * (9 + w * len(cols))) if c == "MEAN" else "", end="")
        if c == "MEAN":
            print()
        print(line)

    print("\nlast column is 'both' minus 'uncorrected'.")
    print("scope columns assume P and R; the box-extent column assumes nothing")
    print("beyond the matching rule. Quote the range, not a single figure.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["class"] + [l.replace("\n", " ") for l, _ in cols])
            for c in order:
                wr.writerow([c] + [cols[i][1].get(c, "") for i in range(len(cols))])
        print(f"-> {args.csv}")


if __name__ == "__main__":
    main()
