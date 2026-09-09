import csv
import re
import subprocess
import sys
from pathlib import Path

# label, baseline arm (A), intervention arm (B). Delta is reported as B - A.
PAIRS = [
    ("Rendering (compositor settings)", "hybrid_yolo26_sd35cut_lora_v2_r200_seed*",
     "hybrid_yolo26_sd35cut_lora_v2p_r200_seed*"),
    ("Photometric harmonisation (v7)", "hybrid_yolo26_sd35cut_lora_v2_r125_seed*",
     "hybrid_yolo26_sd35cut_v7_r125_seed*"),
    ("Further defect correction (v8)", "hybrid_yolo26_sd35cut_lora_v2_r125_seed*",
     "hybrid_yolo26_sd35cut_v8_r125_seed*"),
    ("Per-species conditioning", "hybrid_yolo26_sd35cut_lora_v2_r125_seed*",
     "hybrid_yolo26_sd35cut_perspecies_r125_seed*"),
    ("Instance sharpness matched", "hybrid_yolo26_sd35cut_v9_r125_x50_seed*",
     "hybrid_yolo26_sd35cut_v9_blur_r125_x50_seed*"),
    ("Background oracle", "hybrid_yolo26_sd35cut_v9_r125_x50_seed*",
     "hybrid_yolo26_sd35cut_v9B_r125_x50_seed*"),
    ("Photorealistic pipeline, full pool", "hybrid_yolo26_sd35cut_lora_v2b_r200_seed*",
     "hybrid_yolo26_sd35cut_v9_r200_seed*"),
    ("Distinct scenes, 2,500 to 26,220", "hybrid_yolo26_sd35cut_lora_v2_r200_seed*",
     "hybrid_yolo26_sd35cut_lora_v2b_r200_seed*"),
]

LINE = re.compile(r"^\s*all\s+A\s+([\d.]+)\s+B\s+([\d.]+)\s+delta\s+([-+][\d.]+)"
                  r".*?\[([-+][\d.]+),\s*([-+][\d.]+)\].*?p=([\d.]+)")

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
OUT = Path(f"results/tables/forest_{SPLIT}.csv")

rows = []
for label, a, b in PAIRS:
    r = subprocess.run(["python3", "scripts/ap_analysis.py", "--runs", a, "--vs", b,
                        "--split", SPLIT, "--boot", "1000"],
                       capture_output=True, text=True)
    hit = next((LINE.match(l) for l in r.stdout.splitlines() if LINE.match(l)), None)
    if not hit:
        print(f"MISSING  {label}\n         A={a}\n         B={b}")
        continue
    A, B, d, lo, hi, pv = hit.groups()
    rows.append({"label": label, "A": A, "B": B, "delta": d,
                 "lo": lo, "hi": hi, "p": pv})
    print(f"{label:38s} {d:>8s}  [{lo}, {hi}]  p={pv}")

if rows:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {OUT}")
