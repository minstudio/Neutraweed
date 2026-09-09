import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
CKPT = ROOT / "results" / "checkpoints"
TABLES = ROOT / "results" / "tables"
OUT = TABLES / "recipe_audit.csv"

SEED_RE = re.compile(r"_seed\d+$")
RATIO_RE = re.compile(r"_r(\d+)")


def classify(peak_lr):
    if peak_lr is None:
        return "unknown"
    if peak_lr > 2e-2:
        return "MuSGD_lr0.01"
    if peak_lr > 5e-4:
        return "AdamW_lr0.00111"
    if peak_lr > 5e-5:
        return "AdamW_lr0.0001"
    return "explicit_lr1e-05"


def read_run(d):
    f = d / "results.csv"
    if not f.exists():
        return None
    with f.open() as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return None
    header = [h.strip() for h in rows[0]]
    col = None
    for i, h in enumerate(header):
        if "lr/pg0" in h:
            col = i
    if col is None:
        return None
    peak = 0.0
    n = 0
    for r in rows[1:]:
        if len(r) <= col:
            continue
        n += 1
        try:
            v = float(r[col])
        except ValueError:
            continue
        peak = max(peak, v)
    return n, peak


def load_metrics():
    m = defaultdict(dict)
    f = TABLES / "metrics.csv"
    if not f.exists():
        f = ROOT / "metrics.csv"
    if not f.exists():
        return m
    with f.open() as fh:
        for row in csv.DictReader(fh):
            m[row["run_name"]][row["split"]] = row
    return m


metrics = load_metrics()
runs = {}
for d in sorted(p for p in CKPT.iterdir() if p.is_dir()):
    got = read_run(d)
    if got is None:
        continue
    epochs, peak = got
    runs[d.name] = {
        "epochs": epochs,
        "peak_lr": peak,
        "recipe": classify(peak),
        "test_map50": metrics.get(d.name, {}).get("test", {}).get("map50", ""),
        "val_map50": metrics.get(d.name, {}).get("val", {}).get("map50", ""),
    }

TABLES.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run_name", "epochs", "peak_lr", "recipe", "test_map50", "val_map50"])
    for k in sorted(runs):
        r = runs[k]
        w.writerow([k, r["epochs"], f"{r['peak_lr']:.6g}", r["recipe"],
                    r["test_map50"], r["val_map50"]])
print(f"wrote {OUT}  ({len(runs)} runs)\n")

groups = defaultdict(list)
for k, r in runs.items():
    groups[SEED_RE.sub("", k)].append((k, r))

print("=" * 78)
print("A. CONDITIONS WHERE SEEDS DISAGREE ON RECIPE")
print("   (same data, same condition, different training recipe -> free measurement)")
print("=" * 78)
found = 0
deltas = []
for cond in sorted(groups):
    byrec = defaultdict(list)
    for k, r in groups[cond]:
        byrec[r["recipe"]].append((k, r))
    if len(byrec) < 2:
        continue
    found += 1
    print(f"\n{cond}")
    means = {}
    for rec in sorted(byrec):
        vals = []
        for k, r in byrec[rec]:
            v = r["test_map50"]
            print(f"    {rec:22s} epochs={r['epochs']:<4d} {k:52s} test_map50={v or 'NA'}")
            if v:
                vals.append(float(v))
        if vals:
            means[rec] = sum(vals) / len(vals)
    if len(means) == 2:
        (a, ma), (b, mb) = sorted(means.items())
        d = mb - ma
        deltas.append((cond, a, b, d))
        print(f"    -> {b} minus {a} = {d:+.4f}")
if not found:
    print("\n  none found")
if deltas:
    print("\n  recipe deltas observed:")
    for cond, a, b, d in deltas:
        print(f"    {d:+.4f}   {cond}   ({b} vs {a})")
    mean_abs = sum(abs(d) for _, _, _, d in deltas) / len(deltas)
    print(f"\n  mean |delta| across {len(deltas)} conditions = {mean_abs:.4f}")

print("\n" + "=" * 78)
print("B. RECIPE ALONG EACH RATIO CURVE")
print("=" * 78)
fam = defaultdict(dict)
for k, r in runs.items():
    base = SEED_RE.sub("", k)
    m = RATIO_RE.search(base)
    if not m:
        continue
    family = base[:m.start()] + base[m.end():]
    fam[family].setdefault(int(m.group(1)), []).append(r)
for family in sorted(fam):
    ratios = fam[family]
    if len(ratios) < 3:
        continue
    print(f"\n{family}")
    prev = None
    for rr in sorted(ratios):
        recs = sorted({x["recipe"] for x in ratios[rr]})
        eps = sorted({x["epochs"] for x in ratios[rr]})
        flag = "  <== RECIPE CHANGES HERE" if prev is not None and recs != prev else ""
        vals = [float(x["test_map50"]) for x in ratios[rr] if x["test_map50"]]
        ap = f"{sum(vals)/len(vals):.4f}" if vals else "NA"
        print(f"    r{rr/100:<5.2f} ap={ap}  epochs={eps}  {'/'.join(recs)}{flag}")
        prev = recs

print("\n" + "=" * 78)
print("C. PAIRWISE CONTRASTS: RECIPE MATCHED OR NOT")
print("=" * 78)
CONTRASTS = [
    ("hybrid_yolo26_sd35cut_lora_v2_r200", "hybrid_yolo26_sd35cut_lora_v2b_r200"),
    ("hybrid_yolo26_sd35cut_lora_v2_r200", "hybrid_yolo26_sd35cut_lora_v2p_r200"),
    ("hybrid_yolo26_sd35cut_v9_r125_x50", "hybrid_yolo26_sd35cut_v9B_r125_x50"),
    ("hybrid_yolo26_sd35cut_v9_r125_x50", "hybrid_yolo26_sd35cut_v9_blur_r125_x50"),
    ("hybrid_yolo26_sd35cut_v9_r125_x50", "realonly_yolo26_x50"),
    ("hybrid_yolo26_sd35cut_v9_r100_x50", "realonly_yolo26_x50"),
    ("hybrid_yolo26_sd35cut_lora_v2_r125", "realonly_yolo26_real"),
    ("hybrid_yolo26_sd35cut_lora_v2_r125", "hybrid_yolo26_sd35cut_v8_r125"),
    ("hybrid_yolo26_sd35cut_lora_v2_r125", "hybrid_yolo26_sd35cut_v7_r125"),
    ("hybrid_yolo26_sd35cut_lora_v2_r125", "hybrid_yolo26_sd35cut_perspecies_r125"),
    ("hybrid_yolo26_sd35cut_lora_v5_r125", "hybrid_yolo26_sd35cut_v8_r125"),
    ("realonly_yolo26_hires", "hybrid_yolo26_v2_hires"),
]


def summary(cond):
    items = groups.get(cond, [])
    if not items:
        return None
    recs = sorted({r["recipe"] for _, r in items})
    eps = sorted({r["epochs"] for _, r in items})
    vals = [float(r["test_map50"]) for _, r in items if r["test_map50"]]
    return recs, eps, (sum(vals) / len(vals) if vals else None), len(items)


for a, b in CONTRASTS:
    sa, sb = summary(a), summary(b)
    if sa is None or sb is None:
        print(f"\n  MISSING  {a}  vs  {b}")
        continue
    ok = sa[0] == sb[0] and len(sa[0]) == 1
    same_ep = sa[1] == sb[1] and len(sa[1]) == 1
    verdict = "MATCHED" if ok and same_ep else ("optimizer MATCHED, budget DIFFERS" if ok else "RECIPE MISMATCH")
    d = ""
    if sa[2] is not None and sb[2] is not None:
        d = f"  delta={sb[2]-sa[2]:+.4f}"
    print(f"\n  [{verdict}]{d}")
    print(f"      {a:52s} n={sa[3]} epochs={sa[1]} {'/'.join(sa[0])}")
    print(f"      {b:52s} n={sb[3]} epochs={sb[1]} {'/'.join(sb[0])}")