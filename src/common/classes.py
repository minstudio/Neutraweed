"""Canonical class registry for the tomato-weed dataset.

Detection target = the FIVE weed species from Gómez et al. 2025 (Agricultural
Systems 228), the study this dataset comes from. The crop (LYPES) and the
"not recognised" placeholder (NR) are deliberately EXCLUDED: the crop is not a
detection target here and NR is a non-class. Both are dropped to match the
published protocol and to remove a cross-season annotation inconsistency,
since TOMATO_2/2022 never labels the crop.

The class order is FROZEN here: changing it silently invalidates every trained
checkpoint and YOLO label file. Order mirrors the paper's table order so our
per-class numbers line up with theirs. If you must add a class, append it.

Codes are EPPO Bayer codes as used in the source VOC annotations.
"""

from __future__ import annotations

# Frozen, ordered list. Index == YOLO class id. (Paper table order.)
CLASS_NAMES: list[str] = [
    "SOLNI",  # Solanum nigrum        — black nightshade
    "POROL",  # Portulaca oleracea    — common purslane
    "SETVE",  # Setaria verticillata  — bristly foxtail
    "CYPRO",  # Cyperus rotundus      — purple nutsedge
    "ECHCG",  # Echinochloa crus-galli — barnyardgrass
]

# Human-readable names, for plots and reports.
CLASS_DISPLAY: dict[str, str] = {
    "SOLNI": "Solanum nigrum (nightshade)",
    "POROL": "Portulaca oleracea (purslane)",
    "SETVE": "Setaria verticillata (foxtail)",
    "CYPRO": "Cyperus rotundus (nutsedge)",
    "ECHCG": "Echinochloa crus-galli (barnyardgrass)",
}

# Codes present in the raw VOC data but intentionally NOT detection targets.
# voc_to_yolo skips boxes with these names.
EXCLUDED_CODES: set[str] = {
    "LYPES",  # Solanum lycopersicum — the tomato CROP (only labelled in 2021)
    "NR",     # not recognised — placeholder, not a real class
}

NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


def is_known(name: str) -> bool:
    """True if `name` is a detection target (one of the 5 weeds)."""
    return name in NAME_TO_ID


def class_id(name: str) -> int:
    """Map an EPPO code to its frozen YOLO id. Raises on non-target codes."""
    try:
        return NAME_TO_ID[name]
    except KeyError as exc:
        hint = " (excluded by design)" if name in EXCLUDED_CODES else ""
        raise KeyError(
            f"Unknown class code {name!r}{hint}; targets: {CLASS_NAMES}."
        ) from exc


def num_classes() -> int:
    return len(CLASS_NAMES)
