import csv
import re
import subprocess
from itertools import combinations
from pathlib import Path

PREDS = Path("results/tables/preds")
OUT = Path("results/tables/run_means.csv")
SPLIT = "test"

TARGETS = [
    ("v7 photometric, paper says -0.026", -0.026, "v7"),
    ("per-species conditioning, paper says -0.022", -0.022, "perspecies"),
    ("background oracle, paper says +0.009", +0.009, "v9B"),
    ("v8 defect correction, paper says flat", 0.000, "v8"),
    ("base model without adapter, paper says -0.010", -0.010, None),
]
TOL = 0.0035
MEAN = re.compile(r"^MEAN\s+([\d.]+)")


def families():
    seen = {}
    for f in sorted(PREDS.glob(f"*_{SPLIT}.npz")):
        fam = re.sub(rf"_seed\d+_{SPLIT}$", "", f.stem)
        seen.setdefault(fam, 0)
        seen[fam] += 1
    return {k: v for k, v in seen.items() if v >= 2}


def mean_ap(fam):
    r = subprocess.run(["python3", "scripts/ap_analysis.py", "--runs", f"{fam}_seed*",
                        "--split", SPLIT, "--boot", "0"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        m = MEAN.match(line)
        if m:
            return float(m.group(1))
    return None


def ratio_of(fam):
    m = re.search(r"_r(\d+)(_x\d+)?$", fam)
    return (m.group(1), m.group(2) or "") if m else (None, "")


if __name__ == "__main__":
    fams = families()
    print(f"{len(fams)} run families with >=2 seeds\n")
    means = {}
    for i, fam in enumerate(sorted(fams), 1):
        v = mean_ap(fam)
        if v is not None:
            means[fam] = v
        print(f"  [{i}/{len(fams)}] {fam:52s} {v}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "map50", "n_seeds"])
        for k, v in sorted(means.items()):
            w.writerow([k, v, fams[k]])
    print(f"\n-> {OUT}\n")

    print("=" * 74)
    print("CANDIDATE PAIRS REPRODUCING EACH PUBLISHED NUMBER")
    print("(delta = B - A, matched on ratio suffix and crowding-filter suffix)")
    print("=" * 74)
    for label, target, must in TARGETS:
        print(f"\n{label}")
        hits = 0
        for a, b in combinations(sorted(means), 2):
            if ratio_of(a) != ratio_of(b) or ratio_of(a)[0] is None:
                continue
            if must and must not in a and must not in b:
                continue
            for x, y in ((a, b), (b, a)):
                d = means[y] - means[x]
                if abs(d - target) <= TOL:
                    print(f"    {d:+.4f}   A={x}\n              B={y}")
                    hits += 1
        if not hits:
            print("    no pair within tolerance")
