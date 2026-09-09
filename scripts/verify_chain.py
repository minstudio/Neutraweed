import csv
import subprocess
import sys
from pathlib import Path

RUN = sys.argv[1] if len(sys.argv) > 1 else "hybrid_yolo26_sd35cut_lora_v2b_r200_seed*"
OUT = Path("/tmp/chain.csv")

FIVE = ["SOLNI", "POROL", "SETVE", "CYPRO", "ECHCG"]
THREE = ["SOLNI", "POROL", "SETVE"]

cmd = ["python3", "scripts/ap_analysis.py", "--runs", RUN,
       "--bins", "60", "--boot", "0", "--csv", str(OUT)]
print("$ " + " ".join(cmd))
subprocess.run(cmd, check=True)

rows = list(csv.DictReader(OUT.open()))
bins = []
for r in rows:
    if r["size_bin"] not in bins:
        bins.append(r["size_bin"])


def mean(size_bin, keep):
    vals = [(r["class"], float(r["map50"]), int(r["n_gt"])) for r in rows
            if r["size_bin"] == size_bin and r["class"] in keep and r["map50"] != ""]
    if not vals:
        return None, []
    return sum(v for _, v, _ in vals) / len(vals), vals


print()
for b in bins:
    for label, keep in (("five", FIVE), ("three", THREE)):
        m, vals = mean(b, keep)
        if m is None:
            continue
        detail = " ".join(f"{c}={v:.3f}[{g}]" for c, v, g in vals)
        print(f"{b:>10s}  {label:5s}  mean={m:.3f}   {detail}")
