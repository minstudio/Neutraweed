import csv
import re
import subprocess
from pathlib import Path

POOLS = [
    "sd35cut_lora", "sd35cut_lora_v2", "sd35cut_lora_v2b", "sd35cut_lora_v2p",
    "sd35cut_lora_v4", "sd35cut_lora_v5", "sd35cut_perspecies",
    "sd35cut_v7", "sd35cut_v8", "sd35cut_v9", "sd35cut_v9B", "sd35cut_v9_blur",
]
LABELS = [
    ("fol_s", "foliage HSV saturation"),
    ("fol_v", "foliage HSV value"),
    ("fol_exg", "foliage ExG"),
    ("fol_contrast", "plant self-shading p90/p10"),
    ("rel_v", "plant V / local soil V"),
    ("rel_s", "plant S / local soil S"),
    ("soil_v", "soil V around plants"),
    ("sh_depth", "shadow depth"),
    ("sh_area", "shadow area"),
    ("block", "2x2 blockiness R2"),
]
TRIPLE = re.compile(r"([-\d.]+)\s+\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]")
OUT = Path("results/tables/appearance.csv")
ARGS = ["--split", "train", "--tile", "1536", "--imgsz", "1024",
        "--n-real", "300", "--n-pool", "300", "--seed", "0"]


def score(pool):
    r = subprocess.run(
        ["python3", "scripts/fit_appearance_prior.py", "--score", pool, *ARGS,
         "--out", f"results/tables/appearance_{pool}.json"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL {pool}: {r.stderr.strip().splitlines()[-1:]}")
        return None
    out = {}
    for line in r.stdout.splitlines():
        for key, label in LABELS:
            if line.startswith(label):
                m = TRIPLE.findall(line)
                if len(m) >= 2:
                    out[key] = {
                        "real": tuple(float(x) for x in m[0]),
                        "pool": tuple(float(x) for x in m[1]),
                    }
                break
    return out or None


if __name__ == "__main__":
    rows = []
    for pool in POOLS:
        print(f"scoring {pool}")
        d = score(pool)
        if not d:
            continue
        inband = 0
        row = {"pool": pool}
        for key, _ in LABELS:
            if key not in d:
                row[key] = ""
                continue
            rp50, rp10, rp90 = d[key]["real"]
            pp50 = d[key]["pool"][0]
            ok = rp10 <= pp50 <= rp90
            inband += ok
            row[key] = f"{pp50:g}"
            row[f"{key}_in"] = int(ok)
            row[f"{key}_real"] = f"{rp50:g}"
            row[f"{key}_band"] = f"[{rp10:g},{rp90:g}]"
        row["n_in_band"] = inband
        row["n_stats"] = sum(1 for k, _ in LABELS if k in d)
        rows.append(row)
        print(f"  {inband}/{row['n_stats']} inside the real p10-p90 band")

    if rows:
        cols, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    cols.append(k)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {OUT}")
        print(f"{'pool':24s} in-band")
        for r in rows:
            print(f"{r['pool']:24s} {r['n_in_band']}/{r['n_stats']}")
