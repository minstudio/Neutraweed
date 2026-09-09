from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import classes, paths


def count_split(labels_dir: Path) -> tuple[int, int, dict[int, int]]:
    per_class: dict[int, int] = defaultdict(int)
    n_images = 0
    n_boxes = 0
    for txt in sorted(labels_dir.glob("*.txt")):
        n_images += 1
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cid = int(float(line.split()[0]))
            per_class[cid] += 1
            n_boxes += 1
    return n_images, n_boxes, per_class


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Instance (bounding-box) counts per class per split, from the "
                    "frozen YOLO label files. Pure file reads, safe on the login node.")
    ap.add_argument("--labels-root", type=Path, default=paths.REAL / "labels",
                    help="dir containing train/ val/ test/ label folders")
    ap.add_argument("--out", type=Path, default=paths.TABLES / "instance_counts.csv")
    args = ap.parse_args()

    splits = [s for s in paths.SPLITS if (args.labels_root / s).is_dir()]
    if not splits:
        raise SystemExit(f"No split dirs under {args.labels_root}")

    totals: dict[str, tuple[int, int, dict[int, int]]] = {}
    for s in splits:
        totals[s] = count_split(args.labels_root / s)

    names = classes.CLASS_NAMES
    col = 9
    header = f"{'class':<8}" + "".join(f"{s:>{col}}" for s in splits) + f"{'TOTAL':>{col}}"
    print("\nInstances (bounding boxes) per class per split")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    grand = defaultdict(int)
    for cid, name in enumerate(names):
        row = f"{name:<8}"
        rt = 0
        for s in splits:
            c = totals[s][2].get(cid, 0)
            row += f"{c:>{col}}"
            rt += c
            grand[s] += c
        row += f"{rt:>{col}}"
        print(row)
    print("-" * len(header))
    tr = f"{'TOTAL':<8}" + "".join(f"{grand[s]:>{col}}" for s in splits)
    tr += f"{sum(grand.values()):>{col}}"
    print(tr)
    imgs = f"{'images':<8}" + "".join(f"{totals[s][0]:>{col}}" for s in splits)
    imgs += f"{sum(totals[s][0] for s in splits):>{col}}"
    print(imgs)
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class"] + splits + ["total"])
        for cid, name in enumerate(names):
            r = [totals[s][2].get(cid, 0) for s in splits]
            w.writerow([cid, name] + r + [sum(r)])
        w.writerow(["", "images"] + [totals[s][0] for s in splits]
                   + [sum(totals[s][0] for s in splits)])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
