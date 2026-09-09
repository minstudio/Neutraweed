import argparse
import sys
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

CLASSES = ["SOLNI", "POROL", "SETVE", "CYPRO", "ECHCG"]

GOMEZ = {
    "A1 (2021)": [695, 546, 816, 1070, 208],
    "B  (2021)": [958, 14, 299, 990, 14],
    "C  (2021)": [1001, 95, 140, 477, 14],
    "D  (2022)": [1360, 333, 133, 1791, 240],
    "E  (2022)": [1008, 213, 156, 761, 2415],
    "A2 (2022)": [1140, 116, 47, 74, 71],
}

SEARCH = [
    "data/real", "data/raw", "data", "datasets", "raw",
    "~/TOMATO", "~/data", "..",
]


def discover(extra):
    roots = [Path(p).expanduser() for p in (extra or SEARCH)]
    hits = defaultdict(int)
    for r in roots:
        if not r.exists():
            continue
        for x in r.rglob("*.xml"):
            hits[str(x.parent)] += 1
    return hits


def scan(roots, depth_from_end):
    per = defaultdict(lambda: defaultdict(int))
    n = 0
    for r in roots:
        for x in Path(r).expanduser().rglob("*.xml"):
            try:
                root = ET.parse(x).getroot()
            except ET.ParseError:
                continue
            n += 1
            folder = (root.findtext("folder") or "").strip()
            if not folder or folder.lower() in {"images", "annotations", "jpegimages"}:
                parts = x.resolve().parts
                folder = "/".join(parts[-(depth_from_end + 1):-1])
            for o in root.findall("object"):
                name = (o.findtext("name") or "").strip().upper()
                if name in CLASSES:
                    per[folder][name] += 1
    return per, n


def profile(counts):
    t = sum(counts)
    return [round(100 * c / t) for c in counts] if t else [0] * 5


def l1(a, b):
    return sum(abs(x - y) for x, y in zip(profile(a), profile(b)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--find", action="store_true")
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    if args.find or not args.roots:
        hits = discover(args.roots)
        if not hits:
            print("no .xml found under: " + ", ".join(args.roots or SEARCH))
            print("locate the VOC tree with:")
            print("  find ~ /media/beegfs -maxdepth 8 -name '*.xml' 2>/dev/null | head")
            sys.exit(1)
        print(f"{'directory':70s} {'xml':>7s}")
        for d, c in sorted(hits.items(), key=lambda kv: -kv[1])[:40]:
            print(f"{d:70s} {c:7d}")
        if args.find:
            sys.exit(0)

    per, n = scan(args.roots, args.depth)
    print(f"parsed {n} xml files, {len(per)} folders\n")
    mine = {k: [v[c] for c in CLASSES] for k, v in sorted(per.items())}

    print(f"{'folder':44s} {'total':>7s}  " + " ".join(f"{c:>6s}" for c in CLASSES))
    for k, v in mine.items():
        print(f"{k:44s} {sum(v):7d}  " + " ".join(f"{x:6d}" for x in v))

    print(f"\n{'folder':44s} {'best gomez match':>18s} {'L1':>5s}")
    for k, v in mine.items():
        if not sum(v):
            continue
        ranked = sorted(GOMEZ.items(), key=lambda kv: l1(v, kv[1]))
        print(f"{k:44s} {ranked[0][0]:>18s} {l1(v, ranked[0][1]):5d}"
              f"   next: {ranked[1][0]} at {l1(v, ranked[1][1])}")
