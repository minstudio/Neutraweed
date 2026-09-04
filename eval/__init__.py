"""Stage F — evaluation on the REAL test set (CLAUDE.md §4/§6).

metrics : P / R / F1 / mAP@50 / mAP@50:95 via Ultralytics val.
stats   : paired tests across seeds (report mean±std, not point diffs — §8).
plots   : the hybrid ratio curve and per-detector deltas.
"""

from . import metrics, plots, stats

__all__ = ["metrics", "stats", "plots"]
