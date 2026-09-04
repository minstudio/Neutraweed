"""Inventory the raw VOC sources: image/object counts, class and plot stats.

Pure read-only sanity check — run it before/after building splits.

Run:  python -m src.data_prep.scan_dataset
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from ..common import paths
from .voc import iter_annotations


def scan() -> dict:
    per_source = {}
    global_classes: Counter = Counter()
    for source, src_dir in paths.RAW_SOURCES.items():
        if not src_dir.exists():
            print(f"  (skip) {source}: not found at {src_dir}")
            continue
        n_img = 0
        n_obj = 0
        class_counts: Counter = Counter()
        plots: dict[str, int] = defaultdict(int)
        missing_img = 0
        for ann in iter_annotations(source, src_dir):
            n_img += 1
            plots[ann.plot] += 1
            if ann.image_path is None:
                missing_img += 1
            for o in ann.objects:
                class_counts[o.name] += 1
                n_obj += 1
        global_classes.update(class_counts)
        per_source[source] = {
            "images": n_img,
            "objects": n_obj,
            "plots": len(plots),
            "missing_images": missing_img,
            "classes": dict(class_counts.most_common()),
        }

    summary = {
        "per_source": per_source,
        "classes_total": dict(global_classes.most_common()),
        "images_total": sum(s["images"] for s in per_source.values()),
        "objects_total": sum(s["objects"] for s in per_source.values()),
    }

    print(json.dumps(summary, indent=2))
    paths.ensure_dirs()
    out = paths.TABLES / "dataset_scan.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten -> {out}")
    return summary


if __name__ == "__main__":
    scan()
