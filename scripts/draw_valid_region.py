"""Draw the region where a detection is valid, one image at a time.

The annotations cover the inter-row area by design, so a correct detection in the
crop-row band is scored as a false positive. This tool records where the valid
region actually is, so evaluation can ignore everything outside it instead of
penalising it. Output feeds `ap_analysis.py --valid-regions`.

Run it LOCALLY, not on the login node — it opens a window. Copy the split's
images down first:

  rsync -a <cluster>:<project>/data/real/images/test/ ./test_tiles/
  python scripts/draw_valid_region.py --images ./test_tiles --out valid_regions.json

Controls
  left click     add a polygon vertex
  ENTER          close the polygon and start another one on the same image
  V or SPACE     whole image is valid (the common case — no crop row in view)
  X              whole image is invalid (skip it entirely)
  U              undo the last vertex
  R              reset this image
  N / RIGHT      save and go to the next image
  B / LEFT       go back one image
  Q              save and quit

Progress is written after every image, so stopping and restarting resumes where
you left off. Images already decided are skipped unless --redo is passed.

IMPORTANT: draw the region from the CROP ROWS visible in the photo. Ground-truth
boxes are deliberately not shown, because drawing the region around the existing
annotations would define every unannotated detection as out of scope and make
the correction meaningless. --show-gt exists for spot-checking only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common import classes, paths  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"}


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"regions": {}, "meta": {"tool": "draw_valid_region", "version": 1}}


def _save(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def _gt_boxes(img: Path, split: str):
    lp = paths.REAL / "labels" / split / f"{img.stem}.txt"
    if not lp.exists():
        return []
    from PIL import Image
    with Image.open(img) as im:
        W, H = im.size
    out = []
    for line in lp.read_text(encoding="utf-8").splitlines():
        q = line.split()
        if len(q) != 5:
            continue
        cid, cx, cy, bw, bh = int(q[0]), *(float(v) for v in q[1:])
        out.append((cid, (cx - bw / 2) * W, (cy - bh / 2) * H,
                    (cx + bw / 2) * W, (cy + bh / 2) * H))
    return out


def main() -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon, Rectangle

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=None,
                    help="directory of images. Default: data/real/images/<split>")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("valid_regions.json"))
    ap.add_argument("--limit", type=int, default=0,
                    help="only present the first N undecided images")
    ap.add_argument("--redo", action="store_true",
                    help="present images that already have a decision")
    ap.add_argument("--show-gt", action="store_true",
                    help="overlay ground-truth boxes. Spot-checking only — see the "
                         "warning in the module docstring.")
    args = ap.parse_args()

    img_dir = args.images or (paths.REAL / "images" / args.split)
    if not img_dir.exists():
        raise SystemExit(f"no such directory: {img_dir}")
    images = sorted(p for p in img_dir.iterdir() if p.suffix in IMG_EXT)
    if not images:
        raise SystemExit(f"no images in {img_dir}")

    data = _load(args.out)
    regions = data["regions"]
    todo = images if args.redo else [p for p in images if p.name not in regions]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print(f"[draw] all {len(images)} images already decided in {args.out}")
        _summary(regions, len(images))
        return

    print(f"[draw] {len(images)} images, {len(regions)} already decided, "
          f"{len(todo)} to go")
    if args.show_gt:
        print("[draw] WARNING ground truth is visible — do not trace the region "
              "around it, or the correction becomes circular")

    state = {"i": 0, "pts": [], "polys": [], "quit": False}
    fig, ax = plt.subplots(figsize=(11, 9))
    try:
        matplotlib.rcParams["keymap.quit"] = []
        matplotlib.rcParams["keymap.back"] = []
        matplotlib.rcParams["keymap.forward"] = []
        matplotlib.rcParams["keymap.save"] = []
    except Exception:
        pass

    def load_existing(name):
        rec = regions.get(name)
        if rec and rec.get("mode") == "poly":
            return [list(map(list, p)) for p in rec.get("polys", [])]
        return []

    def draw():
        from PIL import Image
        ax.clear()
        p = todo[state["i"]]
        with Image.open(p) as im:
            ax.imshow(np.asarray(im.convert("RGB")))
        if args.show_gt:
            for cid, x1, y1, x2, y2 in _gt_boxes(p, args.split):
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor="#3ce65a", lw=1.2))
                ax.text(x1, y1 - 3, classes.CLASS_NAMES[cid], color="#3ce65a", fontsize=7)
        for poly in state["polys"]:
            ax.add_patch(MplPolygon(poly, closed=True, fill=True, alpha=0.22,
                                    facecolor="#2b8cff", edgecolor="#2b8cff", lw=2))
        if state["pts"]:
            xs = [q[0] for q in state["pts"]]
            ys = [q[1] for q in state["pts"]]
            ax.plot(xs, ys, "-o", color="#ff9a2b", lw=2, ms=5)
        rec = regions.get(p.name, {})
        tag = {"all": "  [WHOLE IMAGE VALID]", "none": "  [SKIPPED]"}.get(rec.get("mode"), "")
        ax.set_title(f"{state['i']+1}/{len(todo)}  {p.name}{tag}\n"
                     f"click=vertex  ENTER=close polygon  V=all valid  X=skip  "
                     f"U=undo  R=reset  N=next  B=back  Q=quit", fontsize=9)
        ax.set_axis_off()
        fig.canvas.draw_idle()

    def commit(mode):
        name = todo[state["i"]].name
        if mode == "poly":
            polys = [p for p in state["polys"] if len(p) >= 3]
            if not polys:
                regions.pop(name, None)
                return
            regions[name] = {"mode": "poly", "polys": polys}
        else:
            regions[name] = {"mode": mode}
        _save(args.out, data)

    def advance(step):
        commit("poly") if state["polys"] or state["pts"] else None
        state["i"] += step
        if state["i"] >= len(todo):
            state["quit"] = True
            plt.close(fig)
            return
        state["i"] = max(0, state["i"])
        state["pts"] = []
        state["polys"] = load_existing(todo[state["i"]].name)
        draw()

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None or ev.button != 1:
            return
        state["pts"].append([float(ev.xdata), float(ev.ydata)])
        draw()

    def on_key(ev):
        k = (ev.key or "").lower()
        if k == "enter":
            if len(state["pts"]) >= 3:
                state["polys"].append(state["pts"])
                state["pts"] = []
                commit("poly")
            draw()
        elif k in (" ", "v"):
            state["pts"], state["polys"] = [], []
            commit("all")
            advance(1)
        elif k == "x":
            state["pts"], state["polys"] = [], []
            commit("none")
            advance(1)
        elif k == "u":
            if state["pts"]:
                state["pts"].pop()
            elif state["polys"]:
                state["pts"] = state["polys"].pop()
                state["pts"].pop()
            draw()
        elif k == "r":
            state["pts"], state["polys"] = [], []
            regions.pop(todo[state["i"]].name, None)
            _save(args.out, data)
            draw()
        elif k in ("n", "right"):
            advance(1)
        elif k in ("b", "left"):
            advance(-1)
        elif k == "q":
            if state["polys"] or state["pts"]:
                commit("poly")
            state["quit"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    state["polys"] = load_existing(todo[0].name)
    draw()
    plt.show()

    _save(args.out, data)
    print(f"\n[draw] -> {args.out}")
    _summary(regions, len(images))


def _summary(regions: dict, n_images: int) -> None:
    modes = {}
    for r in regions.values():
        modes[r.get("mode", "?")] = modes.get(r.get("mode", "?"), 0) + 1
    print(f"[draw] decided {len(regions)}/{n_images}: " +
          ", ".join(f"{k}={v}" for k, v in sorted(modes.items())))
    if len(regions) < n_images:
        print(f"[draw] {n_images - len(regions)} still undecided — images without a "
              f"decision are treated as fully valid by ap_analysis.")


if __name__ == "__main__":
    main()
