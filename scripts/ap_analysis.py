"""AP by class and object size, with bootstrap CIs over test tiles (CPU).

Reads the caches written by scripts/cache_preds.py. Nothing here touches a GPU
or a checkpoint, so it is safe on the login node and cheap to re-run.

Three things the current readouts cannot do:

1. AP by object SIZE. The project's claim is that synthetic data helps the
   small, under-represented instances. Per-class AP only proxies that;
   size-stratified AP measures it.
2. Confidence intervals from the ~530 test tiles rather than from 3 seeds. A
   paired t-test on 3 seeds has 2 degrees of freedom; a tile bootstrap has
   hundreds of resampling units and is the honest way to report a small field.
3. A paired A-vs-B comparison that resamples the SAME tiles for both runs, so
   the interval is on the difference and tile-difficulty variance cancels.

Seeds of one config are pooled: a resampling unit is a (tile, seed) pair, so
tile difficulty and seed noise both enter the interval.

Matching follows COCO: greedy by confidence, one GT per detection, area ranges
implemented as ignore regions. It is done once per (image, class) and reused
across size bins, which is a hair more lenient than pycocotools re-matching per
range, and far faster.

  python scripts/ap_analysis.py --runs 'realonly_yolo26_seed*' --metric map50
  python scripts/ap_analysis.py --runs 'realonly_yolo26_seed*' \
      --vs 'hybrid_yolo26_sd35cut_lora_v2_r125_seed*' --metric map50 --boot 1000
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common import classes, paths  # noqa: E402

PREDS = paths.TABLES / "preds"
REC = np.linspace(0, 1, 101)


def _thresholds(metric: str) -> np.ndarray:
    return np.asarray([0.5]) if metric == "map50" else np.arange(0.5, 0.96, 0.05)


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


def _ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the GROUND TRUTH area.

    A prediction that fully contains a ground-truth box scores 1.0 however much
    bigger it is. IoU penalises that; for species annotated inconsistently as one
    clump or several parts (CYPRO), the penalty is an artefact of the annotation
    convention rather than a detection error."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(bb[None, :], 1e-9)).astype(np.float32)


def _score_matrix(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    if mode == "iou":
        return _iou_matrix(a, b)
    if mode == "ioa":
        return _ioa_matrix(a, b)
    if mode == "max":
        return np.maximum(_iou_matrix(a, b), _ioa_matrix(a, b))
    raise SystemExit(f"unknown match mode {mode!r}")


def _size_key(box: np.ndarray) -> np.ndarray:
    if box.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.sqrt(np.clip(box[:, 2] - box[:, 0], 0, None)
                   * np.clip(box[:, 3] - box[:, 1], 0, None))


def _match(pred: np.ndarray, gt: np.ndarray, thr: np.ndarray,
           mode: str = "iou") -> np.ndarray:
    """-> matched[T, P] = index of the GT each detection took, or -1.

    `pred` must already be sorted by descending confidence. Detections whose
    best IoU is below the loosest threshold can never match, so only those
    candidates enter the greedy loop."""
    T, P, G = len(thr), pred.shape[0], gt.shape[0]
    matched = np.full((T, P), -1, dtype=np.int32)
    if P == 0 or G == 0:
        return matched

    ious = _score_matrix(pred[:, :4], gt[:, :4], mode)
    cand = np.flatnonzero(ious.max(axis=1) >= thr.min())
    for ti in range(T):
        taken = np.zeros(G, dtype=bool)
        t = thr[ti]
        for di in cand:
            row = ious[di]
            row = np.where(taken, -1.0, row)
            gj = int(np.argmax(row))
            if row[gj] >= t:
                taken[gj] = True
                matched[ti, di] = gj
    return matched


class Cell:
    """Flattened detections for one (class, size-bin), ready to bootstrap.

    Everything is concatenated in one global confidence-descending order, so a
    bootstrap resample is a weighted cumsum instead of a re-sort.
    """

    def __init__(self, conf, tp, ig, unit, n_units, n_gt_unit):
        o = np.argsort(-conf, kind="stable")
        self.tp = tp[:, o]
        self.ig = ig[:, o]
        self.unit = unit[o]
        self.n_units = n_units
        self.n_gt_unit = n_gt_unit
        self.n_gt = float(n_gt_unit.sum())

    def ap(self, counts: np.ndarray | None = None) -> np.ndarray:
        if counts is None:
            n_gt = self.n_gt
            w = None
        else:
            n_gt = float(counts @ self.n_gt_unit)
            w = counts[self.unit].astype(np.float32)
        T = self.tp.shape[0]
        if n_gt <= 0:
            return np.full(T, np.nan)
        out = np.zeros(T)
        for ti in range(T):
            valid = (~self.ig[ti]).astype(np.float32)
            if w is not None:
                valid = valid * w
            t = np.asarray(self.tp[ti], dtype=np.float32) * valid
            ctp = np.cumsum(t)
            cfp = np.cumsum(valid - t)
            if ctp.size == 0 or ctp[-1] == 0:
                continue
            rec = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            prec = np.maximum.accumulate(prec[::-1])[::-1]
            out[ti] = np.interp(REC, rec, prec, left=prec[0], right=0.0).mean()
        return out


def load_runs(pattern: str, split: str) -> list[Path]:
    if not PREDS.exists():
        raise SystemExit(f"{PREDS} missing — run scripts/cache_preds.py first")
    hits = [p for p in sorted(PREDS.glob(f"*_{split}.npz"))
            if fnmatch.fnmatch(p.name[: -len(f"_{split}.npz")], pattern)]
    if not hits:
        raise SystemExit(f"no caches match {pattern!r} (split={split}) in {PREDS}")
    return hits


def _points_in_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon, vectorised over points."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(pts.shape[0], dtype=bool)
    n = poly.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cross = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / np.where(yj - yi == 0, 1e-12, yj - yi) + xi)
        inside ^= cross
        j = i
    return inside


def load_valid_regions(path):
    """-> {image_name: ('all'|'none'|'poly', [np.ndarray(K,2), ...])}"""
    import json
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, rec in d.get("regions", {}).items():
        mode = rec.get("mode", "all")
        polys = [np.asarray(p, dtype=np.float64) for p in rec.get("polys", [])]
        out[name] = (mode, polys)
    return out


def _valid_mask(boxes: np.ndarray, entry) -> np.ndarray:
    """True where a box's centre falls in the valid region."""
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if entry is None:
        return np.ones(boxes.shape[0], dtype=bool)
    mode, polys = entry
    if mode == "all":
        return np.ones(boxes.shape[0], dtype=bool)
    if mode == "none" or not polys:
        return np.zeros(boxes.shape[0], dtype=bool)
    c = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    m = np.zeros(boxes.shape[0], dtype=bool)
    for p in polys:
        if p.shape[0] >= 3:
            m |= _points_in_poly(c, p)
    return m


def crowding(gt: np.ndarray, iou_thr: float = 0.10) -> float:
    """Fraction of ground-truth boxes that overlap at least one other.

    A count of boxes measures busy-ness; this measures clustering, which is the
    thing that makes a scene ambiguous to annotate and to score."""
    if gt.shape[0] < 2:
        return 0.0
    m = _iou_matrix(gt[:, :4], gt[:, :4])
    np.fill_diagonal(m, 0.0)
    return float((m.max(axis=1) >= iou_thr).mean())


def build(files, bins, thr, scale_to, topk, verified=0.0, assume_recall=1.0,
          verified_conf=0.25, match_mode="iou", max_gt=0, max_overlap=1.0,
          valid_regions=None):
    edges = [0.0] + [float(b) for b in bins] + [float("inf")]
    ranges = {"all": (0.0, float("inf"))}
    for i in range(len(edges) - 1):
        hi = "inf" if np.isinf(edges[i + 1]) else str(int(edges[i + 1]))
        ranges[f"{int(edges[i])}-{hi}"] = (edges[i], edges[i + 1])
    labels = list(ranges)
    nc = len(classes.CLASS_NAMES)

    acc = {(c, lab): {"conf": [], "tp": [], "ig": [], "unit": [], "gt": []}
           for c in range(nc) for lab in labels}
    uid = 0
    dropped = 0
    n_out = 0
    for f in files:
        z = np.load(f, allow_pickle=False)
        names = z["image_names"]
        imgsz = int(z["imgsz"][0])
        for i in range(len(names)):
            k = f"{i:06d}"
            pred, gt, wh = z[f"p{k}"], z[f"g{k}"], z[f"s{k}"]
            if max_gt and gt.shape[0] > max_gt:
                dropped += 1
                continue
            if max_overlap < 1.0 and crowding(gt) > max_overlap:
                dropped += 1
                continue
            vr = valid_regions.get(str(names[i])) if valid_regions else None
            if vr is not None and vr[0] == "none":
                dropped += 1
                continue
            gt_ok = _valid_mask(gt[:, :4], vr) if gt.size else np.zeros(0, bool)
            pr_ok = _valid_mask(pred[:, :4], vr) if pred.size else np.zeros(0, bool)
            if vr is not None and vr[0] != "all":
                n_out += int((~gt_ok).sum())
                gt = gt[gt_ok] if gt.size else gt
                pred = pred[pr_ok] if pred.size else pred
            scale = (scale_to / max(wh[0], wh[1])) if scale_to else (imgsz / max(wh[0], wh[1]))
            if verified > 0 and pred.size and gt.size:
                gmax_all = _iou_matrix(pred[:, :4], gt[:, :4]).max(axis=1)
            elif verified > 0 and pred.size:
                gmax_all = np.zeros(pred.shape[0], dtype=np.float32)
            else:
                gmax_all = None
            for cid in range(nc):
                p = pred[pred[:, 5] == cid][:, :5] if pred.size else np.zeros((0, 5), np.float32)
                g = gt[gt[:, 4] == cid][:, :4] if gt.size else np.zeros((0, 4), np.float32)
                if p.shape[0]:
                    p = p[np.argsort(-p[:, 4], kind="stable")][:topk]
                matched = _match(p, g, thr, match_mode)
                psz = _size_key(p) * scale
                gsz = _size_key(g) * scale
                gmax_c = gmax_all[pred[:, 5] == cid][
                    np.argsort(-pred[pred[:, 5] == cid][:, 4], kind="stable")][:topk] \
                    if (gmax_all is not None and p.shape[0]) else None
                for lab in labels:
                    lo, hi = ranges[lab]
                    keep_g = (gsz >= lo) & (gsz < hi)
                    a = acc[(cid, lab)]
                    n_gt_unit = float(keep_g.sum())
                    a["unit"].append(np.full(p.shape[0], uid, dtype=np.int64))
                    a["conf"].append(p[:, 4] if p.shape[0] else np.zeros(0, np.float32))
                    if p.shape[0] == 0:
                        a["gt"].append(n_gt_unit)
                        a["tp"].append(np.zeros((len(thr), 0), np.float32))
                        a["ig"].append(np.zeros((len(thr), 0), bool))
                        continue
                    hit = matched >= 0
                    gk = (keep_g[np.clip(matched, 0, None)] & hit) if keep_g.size \
                        else np.zeros_like(hit)
                    p_in = (psz >= lo) & (psz < hi)
                    tp = (hit & gk).astype(np.float32)
                    ig = (hit & ~gk) | (~hit & ~p_in[None, :])
                    if gmax_c is not None:
                        bkg = (gmax_c < 0.1) & p_in & (p[:, 4] >= verified_conf)
                        if bkg.any():
                            tp[:, bkg] = verified
                            n_gt_unit += verified * int(bkg.sum()) / max(assume_recall, 1e-6)
                    a["gt"].append(n_gt_unit)
                    a["tp"].append(tp)
                    a["ig"].append(ig)
            uid += 1

    if valid_regions:
        print(f"[ap] valid regions applied: {n_out} GT boxes outside the region "
              f"excluded; detections outside are ignored, not penalised")
    if max_gt or max_overlap < 1.0 or valid_regions:
        print(f"[ap] dropped {dropped} crowded tiles "
              f"(max_gt={max_gt or '-'}, max_overlap={max_overlap if max_overlap < 1 else '-'})")
    cells = {}
    for key, a in acc.items():
        cells[key] = Cell(np.concatenate(a["conf"]) if a["conf"] else np.zeros(0, np.float32),
                          np.concatenate(a["tp"], axis=1),
                          np.concatenate(a["ig"], axis=1),
                          np.concatenate(a["unit"]) if a["unit"] else np.zeros(0, np.int64),
                          uid, np.asarray(a["gt"], dtype=np.float64))
    return cells, labels, uid


def _cells_metric(cells, labels, counts, present, reducer):
    out = {}
    for lab in labels:
        vals = []
        for c in present:
            ap = cells[(c, lab)].ap(counts)
            vals.append(reducer(ap))
        out[lab] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="glob over cached run names (all seeds)")
    ap.add_argument("--vs", default=None, help="second config to compare against --runs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--metric", default="map50", choices=["map50", "map50_95"])
    ap.add_argument("--bins", type=float, nargs="*", default=[64, 128],
                    help="sqrt(area) bin edges in imgsz px (default 64 128)")
    ap.add_argument("--boot", type=int, default=1000,
                    help="bootstrap resamples (0 = off). map50_95 is 10x slower; "
                         "use ~200 there.")
    ap.add_argument("--topk", type=int, default=300,
                    help="keep at most this many detections per image per class")
    ap.add_argument("--scale-to", type=int, default=None,
                    help="express sizes at this render width (default: cached imgsz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp-bkg-verified", type=float, default=0.0, metavar="P",
                    help="fraction of background false positives (detections with "
                         "max IoU < 0.1 against every GT box of every class) that a "
                         "blind sample verified as real plants of that class. Those "
                         "detections get P true-positive credit and P phantom ground "
                         "truth, approximating AP against the plants that are "
                         "actually present rather than the ones annotated. 0 = off.")
    ap.add_argument("--fp-bkg-recall", type=float, default=1.0, metavar="R",
                    help="assumed recall on the unannotated plants. 1.0 (default) "
                         "assumes every one was detected, i.e. the optimistic bound. "
                         "Set to the class's measured recall for the realistic "
                         "estimate; the truth is between them.")
    ap.add_argument("--match-mode", default="iou", choices=["iou", "ioa", "max"],
                    help="'iou' standard. 'ioa' = intersection over the GROUND "
                         "TRUTH area, so a prediction that contains the GT box "
                         "counts as correct however much larger it is — use when "
                         "the annotation splits one plant into parts (CYPRO). "
                         "'max' takes whichever of the two is higher.")
    ap.add_argument("--drop-crowded", type=int, default=0, metavar="N",
                    help="exclude tiles with more than N ground-truth boxes. "
                         "Measures busy-ness, not clustering — prefer "
                         "--drop-overlap.")
    ap.add_argument("--valid-regions", type=Path, default=None,
                    help="JSON from scripts/draw_valid_region.py. Detections whose "
                         "centre falls outside the drawn region are ignored rather "
                         "than counted as false positives, so AP is measured over "
                         "the region the annotations actually cover. Images with no "
                         "entry are treated as fully valid.")
    ap.add_argument("--drop-overlap", type=float, default=1.0, metavar="F",
                    help="exclude tiles where more than fraction F of the ground "
                         "truth boxes overlap another box (IoU >= 0.10). This is "
                         "clustering proper: 0.5 drops tiles where over half the "
                         "plants touch a neighbour. 1.0 = off.")
    ap.add_argument("--fp-bkg-conf", type=float, default=0.25, metavar="C",
                    help="only credit background FPs at or above this confidence "
                         "— it MUST match the confidence floor used when the crops "
                         "were verified (error_breakdown.py default 0.25). The "
                         "cache holds detections down to 0.001, and crediting those "
                         "unverified low-confidence detections inflates ground truth "
                         "by an order of magnitude.")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    thr = _thresholds(args.metric)
    reducer = (lambda a: float(a[0])) if args.metric == "map50" else (lambda a: float(np.nanmean(a)))

    vregions = load_valid_regions(args.valid_regions) if args.valid_regions else None
    if vregions:
        modes = {}
        for m, _ in vregions.values():
            modes[m] = modes.get(m, 0) + 1
        print(f"[ap] valid regions for {len(vregions)} images: " +
              ", ".join(f"{k}={v}" for k, v in sorted(modes.items())))

    t0 = time.time()
    fa = load_runs(args.runs, args.split)
    print(f"[ap] A = {len(fa)} cache(s): {', '.join(p.stem for p in fa)}")
    ca, labels, na = build(fa, args.bins, thr, args.scale_to, args.topk,
                           args.fp_bkg_verified, args.fp_bkg_recall,
                           args.fp_bkg_conf, args.match_mode, args.drop_crowded,
                           args.drop_overlap, vregions)
    present = [c for c in range(len(classes.CLASS_NAMES)) if ca[(c, "all")].n_gt > 0]
    if args.fp_bkg_verified > 0:
        print(f"[ap] SCOPE-CORRECTED: background FPs credited at "
              f"P={args.fp_bkg_verified}, R={args.fp_bkg_recall}, "
              f"conf>={args.fp_bkg_conf}")
    if args.match_mode != "iou":
        print(f"[ap] MATCH MODE = {args.match_mode}")
    print(f"[ap] {na} (tile, seed) units, classes present: "
          f"{', '.join(classes.CLASS_NAMES[c] for c in present)} "
          f"({time.time() - t0:.1f}s)")

    rows = []
    print(f"\n{args.metric} by class x size bin (sqrt-area in imgsz px), [n GT]\n")
    print(f"{'class':8s}" + "".join(f"{lab:>18s}" for lab in labels))
    for cid, name in enumerate(classes.CLASS_NAMES):
        cells = []
        for lab in labels:
            cell = ca[(cid, lab)]
            g = int(cell.n_gt)
            v = reducer(cell.ap()) if g else float("nan")
            cells.append("        -  " if g == 0 else f"{v:.3f} [{g}]")
            rows.append({"config": args.runs, "class": name, "size_bin": lab,
                         "n_gt": g, args.metric: round(v, 4) if g else ""})
        print(f"{name:8s}" + "".join(f"{c:>18s}" for c in cells))
    means = _cells_metric(ca, labels, None, present, reducer)
    print(f"{'MEAN':8s}" + "".join(f"{means[lab]:>18.3f}" for lab in labels))
    for lab in labels:
        rows.append({"config": args.runs, "class": "MEAN", "size_bin": lab,
                     "n_gt": int(sum(ca[(c, lab)].n_gt for c in present)),
                     args.metric: round(means[lab], 4)})

    rng = np.random.default_rng(args.seed)
    if args.boot:
        t0 = time.time()
        boots = {lab: [] for lab in labels}
        for _ in range(args.boot):
            counts = np.bincount(rng.integers(0, na, na), minlength=na).astype(np.float64)
            v = _cells_metric(ca, labels, counts, present, reducer)
            for lab in labels:
                boots[lab].append(v[lab])
        print(f"\nbootstrap over {na} (tile, seed) units, {args.boot} resamples "
              f"({time.time() - t0:.1f}s)")
        for lab in labels:
            b = np.asarray(boots[lab])
            lo, hi = np.nanpercentile(b, [2.5, 97.5])
            print(f"  {lab:>10s}  mean {means[lab]:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    if args.vs:
        fb = load_runs(args.vs, args.split)
        print(f"\n[ap] B = {len(fb)} cache(s): {', '.join(p.stem for p in fb)}")
        cb, _, nb = build(fb, args.bins, thr, args.scale_to, args.topk,
                          args.fp_bkg_verified, args.fp_bkg_recall,
                          args.fp_bkg_conf, args.match_mode, args.drop_crowded,
                          args.drop_overlap, vregions)
        mb = _cells_metric(cb, labels, None, present, reducer)
        paired = na == nb
        if not paired:
            print(f"  NOTE A has {na} units, B has {nb} — unpaired bootstrap.")
        print(f"\ndelta (B - A), {args.metric}")
        for lab in labels:
            print(f"  {lab:>10s}  A {means[lab]:.4f}  B {mb[lab]:.4f}  "
                  f"delta {mb[lab] - means[lab]:+.4f}", end="")
            if args.boot:
                d = []
                for _ in range(args.boot):
                    ia = rng.integers(0, na, na)
                    counts_a = np.bincount(ia, minlength=na).astype(np.float64)
                    counts_b = (counts_a if paired
                                else np.bincount(rng.integers(0, nb, nb),
                                                 minlength=nb).astype(np.float64))
                    va = _cells_metric(ca, [lab], counts_a, present, reducer)[lab]
                    vb = _cells_metric(cb, [lab], counts_b, present, reducer)[lab]
                    d.append(vb - va)
                d = np.asarray(d)
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
                print(f"  95% CI [{lo:+.4f}, {hi:+.4f}]  p={max(p, 1.0 / args.boot):.3f}")
            else:
                print()
            rows.append({"config": f"{args.vs} MINUS {args.runs}", "class": "MEAN",
                         "size_bin": lab, "n_gt": int(sum(ca[(c, lab)].n_gt for c in present)),
                         args.metric: round(mb[lab] - means[lab], 4)})

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
