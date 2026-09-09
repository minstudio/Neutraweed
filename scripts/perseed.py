import re
import subprocess
from itertools import product

MEAN = re.compile(r"^MEAN\s+([\d.]+)")
SPLIT = "test"

PAIRS = [
    ("scene count, v2b - v2",
     "hybrid_yolo26_sd35cut_lora_v2_r{r}", "hybrid_yolo26_sd35cut_lora_v2b_r{r}",
     ["25", "50", "75", "100", "125", "150", "175", "200"]),
    ("headline, hybrid - real only",
     "realonly_yolo26", "hybrid_yolo26_sd35cut_lora_v2b_r{r}", ["200"]),
    ("photoreal, v9 - v2b",
     "hybrid_yolo26_sd35cut_lora_v2b_r{r}", "hybrid_yolo26_sd35cut_v9_r{r}", ["200"]),
]
SEEDS = ["0", "1", "2"]


def ap(fam, seed):
    r = subprocess.run(["python3", "scripts/ap_analysis.py",
                        "--runs", f"{fam}_seed{seed}", "--split", SPLIT, "--boot", "0"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        m = MEAN.match(line)
        if m:
            return float(m.group(1))
    return None


if __name__ == "__main__":
    for title, a_t, b_t, ratios in PAIRS:
        print(f"\n{title}")
        print(f"  {'ratio':>6s}  {'A seeds':>22s}   {'B seeds':>22s}   B>A")
        wins = tot = 0
        for r in ratios:
            a_f, b_f = a_t.format(r=r), b_t.format(r=r)
            A = [ap(a_f, s) for s in SEEDS]
            B = [ap(b_f, s) for s in SEEDS]
            if any(v is None for v in A + B):
                print(f"  r{int(r)/100:<5.2f}  missing cache")
                continue
            w = sum(1 for x, y in product(A, B) if y > x)
            wins += sum(1 for x, y in zip(A, B) if y > x)
            tot += len(SEEDS)
            sa = " ".join(f"{v:.3f}" for v in A)
            sb = " ".join(f"{v:.3f}" for v in B)
            print(f"  r{int(r)/100:<5.2f}  {sa:>22s}   {sb:>22s}   "
                  f"{sum(1 for x, y in zip(A, B) if y > x)}/3  "
                  f"(all-pairs {w}/9)")
        if tot:
            print(f"  seed-matched wins: {wins}/{tot}")
