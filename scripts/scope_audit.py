import re
import subprocess
import sys

ARMS = [
    "realonly_yolo26",
    "hybrid_yolo26_sd35cut_lora_v2b_r200",
    "hybrid_yolo26_sd35cut_lora_v2_r200",
    "matched_yolo26_sd35cut_lora_v2b_r200",
    "hybrid_yolo26_sd35cut_v9_r200",
]
ROW = re.compile(r"^(SOLNI|POROL|SETVE|CYPRO|ECHCG)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+"
                 r"([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"


def run(arm):
    r = subprocess.run(["python3", "scripts/error_breakdown.py",
                        "--runs", f"{arm}_seed*", "--split", SPLIT],
                       capture_output=True, text=True)
    rows = [m.groups() for line in r.stdout.splitlines() if (m := ROW.match(line))]
    if not rows:
        return None
    gt = sum(int(x[1]) for x in rows)
    fn = sum(int(x[5]) for x in rows)
    fp_cls = sum(int(x[6]) for x in rows)
    fp_loc = sum(int(x[7]) for x in rows)
    fp_bkg = sum(int(x[8]) for x in rows)
    tp = gt - fn
    fp = fp_cls + fp_loc + fp_bkg
    return {
        "gt": gt, "fn": fn, "fp_bkg": fp_bkg, "fp": fp,
        "micro_rec": tp / gt, "micro_prec": tp / (tp + fp),
        "macro_rec": sum(float(x[2]) for x in rows) / len(rows),
        "macro_prec": sum(float(x[4]) for x in rows) / len(rows),
    }


if __name__ == "__main__":
    res = {}
    print(f"split = {SPLIT}\n")
    hdr = ("arm", "FP-Bkg", "FN", "micro P", "micro R", "macro P", "macro R")
    print(f"{hdr[0]:40s} {hdr[1]:>7s} {hdr[2]:>6s} "
          f"{hdr[3]:>8s} {hdr[4]:>8s} {hdr[5]:>8s} {hdr[6]:>8s}")
    for a in ARMS:
        d = run(a)
        if not d:
            print(f"{a:40s} missing")
            continue
        res[a] = d
        print(f"{a:40s} {d['fp_bkg']:7d} {d['fn']:6d} "
              f"{d['micro_prec']:8.4f} {d['micro_rec']:8.4f} "
              f"{d['macro_prec']:8.4f} {d['macro_rec']:8.4f}")

    if len(res) >= 2:
        def span(k, pts=False):
            v = [d[k] for d in res.values()]
            s = max(v) - min(v)
            return f"{min(v):.4f} to {max(v):.4f}  span {s*100:.1f} pts" if pts \
                else f"{min(v)} to {max(v)}  span {s}"
        print(f"\nacross {len(res)} arms")
        print("  FP-Bkg     ", span("fp_bkg"))
        print("  FN         ", span("fn"))
        fns = [d["fn"] for d in res.values()]
        print(f"  FN swing    {(max(fns)-min(fns))/max(fns)*100:.1f}% of the largest, "
              f"{(max(fns)-min(fns))/min(fns)*100:.1f}% of the smallest")
        print("  micro prec ", span("micro_prec", True))
        print("  micro rec  ", span("micro_rec", True))
        print("  macro prec ", span("macro_prec", True))
        print("  macro rec  ", span("macro_rec", True))
