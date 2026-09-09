import csv
import re
import subprocess
from pathlib import Path

MEANS = Path("results/tables/run_means.csv")
OUT = Path("results/tables/effects.csv")
SPLIT = "test"
BOOT = "1000"

# intervention family -> (reference family, human label)
CONTRASTS = {
    "sd35cut_lora_v2p":  ("sd35cut_lora_v2", "Compositor settings (rendering control)"),
    "sd35cut_lora_v4":   ("sd35cut_lora_v2", "Scene geometry"),
    "sd35cut_lora_v5":   ("sd35cut_lora_v2", "Instance scale sampling"),
    "sd35cut_perspecies": ("sd35cut_lora_v2", "Per-species conditioning"),
    "sd35cut_v7":        ("sd35cut_lora_v2", "Photometric harmonisation"),
    "sd35cut_v8":        ("sd35cut_lora_v2", "Further defect correction"),
    "sd35cut_lora":      ("sd35cut_lora_v2", "Earlier compositor"),
    "sd35cut_v9B":       ("sd35cut_v9", "Background oracle"),
    "sd35cut_v9_blur":   ("sd35cut_v9", "Instance sharpness matched"),
    "sd35cut_lora_v2b":  ("sd35cut_lora_v2", "Distinct scenes, 2,500 to 26,220"),
    "sd35cut_v9":        ("sd35cut_lora_v2b", "Photorealistic pipeline"),
}

PAT = re.compile(r"^hybrid_yolo26_(.+?)_r(\d+)(_x\d+)?$")
LINE = re.compile(r"^\s*all\s+A\s+([\d.]+)\s+B\s+([\d.]+)\s+delta\s+([-+][\d.]+)"
                  r".*?\[([-+][\d.]+),\s*([-+][\d.]+)\].*?p=([\d.]+)")


def index():
    idx = {}
    for row in csv.DictReader(MEANS.open()):
        m = PAT.match(row["family"])
        if m:
            gen, ratio, filt = m.group(1), m.group(2), m.group(3) or ""
            idx[(gen, ratio, filt)] = row["family"]
    return idx


def run(a, b):
    r = subprocess.run(["python3", "scripts/ap_analysis.py",
                        "--runs", f"{a}_seed*", "--vs", f"{b}_seed*",
                        "--split", SPLIT, "--boot", BOOT],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        m = LINE.match(line)
        if m:
            return m.groups()
    return None


if __name__ == "__main__":
    idx = index()
    rows = []
    for gen, (ref, label) in CONTRASTS.items():
        for (g, ratio, filt), fam in sorted(idx.items()):
            if g != gen:
                continue
            ref_fam = idx.get((ref, ratio, filt))
            if not ref_fam:
                print(f"  skip {label} r{ratio}{filt}: no matched {ref}")
                continue
            res = run(ref_fam, fam)
            if not res:
                print(f"  FAIL {label} r{ratio}{filt}")
                continue
            A, B, d, lo, hi, pv = res
            rows.append({"label": label, "generator": gen, "reference": ref,
                         "ratio": int(ratio) / 100, "filter": filt or "none",
                         "A": A, "B": B, "delta": d, "lo": lo, "hi": hi, "p": pv})
            print(f"{label:42s} r{int(ratio)/100:<5.2f}{filt:<5s} "
                  f"{d:>8s}  [{lo}, {hi}]  p={pv}")

    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {OUT}")
