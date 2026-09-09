"""Why each class fails: TP/FN and a split of the false positives (CPU).

mAP tells you a class is weak. It does not tell you whether the model misses the
plants, hallucinates them, calls them the wrong species, or finds them and boxes
them badly. Those need different fixes, so guessing is expensive.

Reads the caches from scripts/cache_preds.py and reports, per class:

  recall / precision at IoU 0.5 and at a loose IoU
  FN                 ground truth nothing was matched to
  FP-Cls             detection sitting on a real weed of a DIFFERENT species
  FP-Loc             detection on the right species, overlapping but under IoU 0.5
  FP-Bkg             detection on nothing at all
  confused-with      which species the FP-Cls errors landed on

Read it like this:
  high FP-Cls        -> species confusion; more data of that class may not help
  high FP-Loc, and a
  big recall jump at
  the loose IoU      -> the model finds the plant but the box extent is ambiguous;
                        suspect annotation consistency, not the detector
  high FN, low FP    -> genuine misses; rarity / signal, where targeted synthetic
                        or class weighting is the lever
  high FP-Bkg        -> hallucination, usually a precision/threshold issue

  python scripts/error_breakdown.py --runs 'hybrid_yolo26_sd35cut_lora_v2_r125_seed[0-9]'
  python scripts/error_breakdown.py --runs 'realonly_yolo26_seed*' --split val --max-size 64
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common import classes, paths  # noqa: E402

PREDS = paths.TABLES / "preds"


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)).astype(np.float32)


def _size_key(box: np.ndarray) -> np.ndarray:
    if box.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.sqrt(np.clip(box[:, 2] - box[:, 0], 0, None)
                   * np.clip(box[:, 3] - box[:, 1], 0, None))


def _greedy(ious: np.ndarray, thr: float) -> np.ndarray:
    """-> matched GT index per detection (-1 = none), greedy by detection order."""
    P, G = ious.shape
    out = np.full(P, -1, dtype=np.int32)
    if P == 0 or G == 0:
        return out
    taken = np.zeros(G, dtype=bool)
    for di in range(P):
        row = np.where(taken, -1.0, ious[di])
        gj = int(np.argmax(row))
        if row[gj] >= thr:
            taken[gj] = True
            out[di] = gj
    return out


def load_runs(pattern: str, split: str) -> list[Path]:
    if not PREDS.exists():
        raise SystemExit(f"{PREDS} missing — run scripts/cache_preds.py first")
    hits = [p for p in sorted(PREDS.glob(f"*_{split}.npz"))
            if fnmatch.fnmatch(p.name[: -len(f"_{split}.npz")], pattern)]
    if not hits:
        raise SystemExit(f"no caches match {pattern!r} (split={split}) in {PREDS}")
    return hits


def analyse(files, conf: float, loose: float, scale_to, min_size, max_size):
    nc = len(classes.CLASS_NAMES)
    z0 = lambda: np.zeros(nc, dtype=np.int64)  # noqa: E731
    st = {k: z0() for k in ("gt", "pred", "tp", "tp_loose", "fp_cls", "fp_loc", "fp_bkg")}
    confused = np.zeros((nc, nc), dtype=np.int64)

    for f in files:
        z = np.load(f, allow_pickle=False)
        names = z["image_names"]
        imgsz = int(z["imgsz"][0])
        for i in range(len(names)):
            k = f"{i:06d}"
            pred, gt, wh = z[f"p{k}"], z[f"g{k}"], z[f"s{k}"]
            scale = (scale_to / max(wh[0], wh[1])) if scale_to else (imgsz / max(wh[0], wh[1]))

            if gt.size:
                gsz = _size_key(gt[:, :4]) * scale
                gt = gt[(gsz >= min_size) & (gsz < max_size)]
            if pred.size:
                pred = pred[pred[:, 4] >= conf]
                psz = _size_key(pred[:, :4]) * scale
                pred = pred[(psz >= min_size) & (psz < max_size)]
            if pred.size:
                pred = pred[np.argsort(-pred[:, 4], kind="stable")]

            all_iou = _iou_matrix(pred[:, :4], gt[:, :4]) if (pred.size and gt.size) \
                else np.zeros((pred.shape[0], gt.shape[0]), np.float32)

            for cid in range(nc):
                gsel = np.flatnonzero(gt[:, 4] == cid) if gt.size else np.zeros(0, int)
                psel = np.flatnonzero(pred[:, 5] == cid) if pred.size else np.zeros(0, int)
                st["gt"][cid] += len(gsel)
                st["pred"][cid] += len(psel)
                if len(psel) == 0:
                    continue

                sub = all_iou[np.ix_(psel, gsel)] if len(gsel) else \
                    np.zeros((len(psel), 0), np.float32)
                m50 = _greedy(sub, 0.5)
                st["tp"][cid] += int((m50 >= 0).sum())
                st["tp_loose"][cid] += int((_greedy(sub, loose) >= 0).sum())

                for j, di in enumerate(psel):
                    if m50[j] >= 0:
                        continue
                    other = np.delete(np.arange(gt.shape[0]), gsel) if gt.size \
                        else np.zeros(0, int)
                    best_o, best_oi = 0.0, -1
                    if other.size:
                        row = all_iou[di, other]
                        bi = int(np.argmax(row))
                        best_o, best_oi = float(row[bi]), int(other[bi])
                    best_s = float(sub[j].max()) if len(gsel) else 0.0
                    if best_o >= 0.5:
                        st["fp_cls"][cid] += 1
                        confused[cid, int(gt[best_oi, 4])] += 1
                    elif best_s >= 0.1:
                        st["fp_loc"][cid] += 1
                    else:
                        st["fp_bkg"][cid] += 1
    return st, confused


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detection confidence floor (default 0.25, the usual "
                         "operating point; AP uses everything, this does not)")
    ap.add_argument("--loose-iou", type=float, default=0.25,
                    help="second, looser IoU. A big recall gain here means the "
                         "plant is found but the box extent disagrees.")
    ap.add_argument("--min-size", type=float, default=0.0)
    ap.add_argument("--max-size", type=float, default=float("inf"),
                    help="restrict to boxes in [min-size, max-size) sqrt-area px, "
                         "e.g. --max-size 64 to look only at the small failures")
    ap.add_argument("--scale-to", type=int, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    files = load_runs(args.runs, args.split)
    print(f"[err] {len(files)} cache(s), split={args.split}, conf>={args.conf}"
          + (f", size [{args.min_size:.0f}, {args.max_size:.0f})"
             if args.max_size != float("inf") or args.min_size else ""))
    st, confused = analyse(files, args.conf, args.loose_iou, args.scale_to,
                           args.min_size, args.max_size)

    print(f"\n{'class':8s}{'nGT':>7s}{'rec':>7s}{f'rec@{args.loose_iou:.2f}':>9s}"
          f"{'prec':>7s}{'FN':>7s}{'FP-Cls':>8s}{'FP-Loc':>8s}{'FP-Bkg':>8s}"
          f"   confused-with")
    rows = []
    for cid, name in enumerate(classes.CLASS_NAMES):
        g, p = st["gt"][cid], st["pred"][cid]
        tp, tl = st["tp"][cid], st["tp_loose"][cid]
        rec = tp / g if g else float("nan")
        recl = tl / g if g else float("nan")
        prec = tp / p if p else float("nan")
        fn = g - tp
        top = np.argsort(-confused[cid])[:2]
        cw = ", ".join(f"{classes.CLASS_NAMES[t]} {confused[cid, t]}"
                       for t in top if confused[cid, t] > 0) or "-"
        print(f"{name:8s}{g:7d}{rec:7.3f}{recl:9.3f}{prec:7.3f}{fn:7d}"
              f"{st['fp_cls'][cid]:8d}{st['fp_loc'][cid]:8d}{st['fp_bkg'][cid]:8d}   {cw}")
        rows.append({"class": name, "n_gt": int(g), "n_pred": int(p),
                     "recall": round(rec, 4), f"recall_iou{args.loose_iou}": round(recl, 4),
                     "precision": round(prec, 4), "FN": int(fn),
                     "FP_cls": int(st["fp_cls"][cid]), "FP_loc": int(st["fp_loc"][cid]),
                     "FP_bkg": int(st["fp_bkg"][cid]), "confused_with": cw})

    print("\nrecall gain from the looser IoU (box-extent disagreement):")
    for cid, name in enumerate(classes.CLASS_NAMES):
        g = st["gt"][cid]
        if not g:
            continue
        d = (st["tp_loose"][cid] - st["tp"][cid]) / g
        flag = "  <-- extent ambiguity" if d >= 0.10 else ""
        print(f"  {name:8s} +{d:.3f}{flag}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
