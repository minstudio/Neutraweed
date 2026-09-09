"""Build the ONE frozen train/val/test split (Stage A).

Protocol: **spatio-temporal field holdout**, following Gómez et al. 2025 (the
study this dataset is from). Whole 2022 FIELDS are held out for val/test; the
detector trains on the 2021+2022 mix. Reasoning:
  * The dataset's value is its temporal axis (TOMATO_1=2021, TOMATO_2=2022).
    The paper shows training on the year+field mix (mAP 0.91) beats single-year
    (0.70). So years must NOT be split into train-vs-test.
  * ECHCG lives almost entirely in 2022 (238 instances in 2021 vs 5084 in 2022),
    so 2022 must be in training or ECHCG is untrainable. We hold out 2022 *fields*
    (Parcela C -> test, Parcela B -> val) while keeping Finca Santa Amalia (2022)
    + all 2021 for training.

Hard invariants enforced here:
  * Field grouping: a field (plot key) is never split across train/val/test ->
    no spatio-temporal leakage.
  * Val and test are 100% real (all source data is real; synthetic builders are
    forbidden from touching them — see src/datasets/build_hybrid.py).
  * Deterministic: the protocol is a fixed field assignment, not a random draw.

Run:  python -m src.data_prep.make_splits
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..common import classes as class_registry
from ..common import paths
from .voc import VocAnnotation, iter_annotations


@dataclass
class PlotGroup:
    key: str                      # "TOMATO_2/Parcela C_alta densidad"
    year: int
    uids: list[str]
    n_images: int
    n_objects: int                # counted over target classes only
    class_counts: Counter = field(default_factory=Counter)


def _collect_groups() -> tuple[list[PlotGroup], dict[str, VocAnnotation]]:
    anns: dict[str, VocAnnotation] = {}
    by_plot: dict[str, list[str]] = defaultdict(list)
    year_of: dict[str, int] = {}
    for source, src_dir in paths.RAW_SOURCES.items():
        if not src_dir.exists():
            continue
        for ann in iter_annotations(source, src_dir):
            anns[ann.uid] = ann
            by_plot[ann.plot_key].append(ann.uid)
            year_of[ann.plot_key] = ann.year

    groups = []
    for key, uids in by_plot.items():
        cc: Counter = Counter()
        for u in uids:
            for o in anns[u].objects:
                if class_registry.is_known(o.name):   # 5 weeds only
                    cc[o.name] += 1
        groups.append(
            PlotGroup(
                key=key,
                year=year_of[key],
                uids=uids,
                n_images=len(uids),
                n_objects=sum(cc.values()),
                class_counts=cc,
            )
        )
    return groups, anns


def _assign_field_holdout(
    groups: list[PlotGroup], test_fields: list[str], val_fields: list[str]
) -> dict[str, str]:
    """Map each field (plot key) to a split. Listed fields -> val/test, rest -> train."""
    test_set, val_set = set(test_fields), set(val_fields)
    known = {g.key for g in groups}
    for f in test_set | val_set:
        if f not in known:
            raise SystemExit(
                f"Configured holdout field {f!r} not found in the data. "
                f"Known fields: {sorted(known)}"
            )
    plot_to_split: dict[str, str] = {}
    for g in groups:
        if g.key in test_set:
            plot_to_split[g.key] = "test"
        elif g.key in val_set:
            plot_to_split[g.key] = "val"
        else:
            plot_to_split[g.key] = "train"
    return plot_to_split


def build_split() -> dict:
    from ..common.config import load_config

    cfg = load_config("base.yaml")
    sp_cfg = cfg["split"]
    if sp_cfg.get("protocol") != "field_holdout":
        raise SystemExit(
            f"Unsupported split protocol {sp_cfg.get('protocol')!r}; "
            "this build expects 'field_holdout'."
        )

    groups, anns = _collect_groups()
    if not groups:
        raise SystemExit("No annotations found under TOMATO_1/ and TOMATO_2/.")

    plot_to_split = _assign_field_holdout(
        groups, sp_cfg["test_fields"], sp_cfg["val_fields"]
    )
    uid_to_split = {uid: plot_to_split[g.key] for g in groups for uid in g.uids}

    split = {
        "protocol": "field_holdout",
        "seed": sp_cfg.get("seed"),
        "group_by": "field",
        "classes": class_registry.CLASS_NAMES,
        "test_fields": sp_cfg["test_fields"],
        "val_fields": sp_cfg["val_fields"],
        "n_images": len(anns),
        "n_fields": len(groups),
        "plot_to_split": plot_to_split,
        "field_year": {g.key: g.year for g in groups},
        "uid_to_split": uid_to_split,
    }

    paths.ensure_dirs()
    with open(paths.SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    _print_summary(split, groups)
    return split


def _print_summary(split: dict, groups: list[PlotGroup]) -> None:
    from collections import Counter as C

    g_by_key = {g.key: g for g in groups}
    per_split_imgs = C(split["uid_to_split"].values())
    print(f"Frozen split (field holdout) -> {paths.SPLIT_FILE}")
    print(f"  fields={split['n_fields']}  images={split['n_images']}  classes={split['classes']}")
    for s in paths.SPLITS:
        fields = [k for k, v in split["plot_to_split"].items() if v == s]
        n_img = per_split_imgs.get(s, 0)
        years = sorted({g_by_key[k].year for k in fields})
        cc: C = C()
        for k in fields:
            cc.update(g_by_key[k].class_counts)
        print(f"  {s:5s}: {n_img:4d} imgs | years={years} | "
              f"objects={[cc[c] for c in split['classes']]} {split['classes']}")
        for k in fields if s != "train" else []:
            print(f"          field: {k} ({g_by_key[k].year})")

    # leakage assertion: every field in exactly one split
    assert len(split["plot_to_split"]) == split["n_fields"], "field assigned twice"
    # temporal sanity: train must contain 2022 (or ECHCG is untrainable)
    train_years = {g.year for g in groups if split["plot_to_split"][g.key] == "train"}
    assert 2022 in train_years, "2022 missing from train — ECHCG would be untrainable"
    print("  leakage check: OK (each field in exactly one split; train spans both years)")


if __name__ == "__main__":
    build_split()
