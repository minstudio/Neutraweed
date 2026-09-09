"""Map valid regions drawn on full frames onto the tiles used for evaluation.

Draw once per photo (27 for the test field) instead of once per tile (532). The
tiling in src/data_prep/tile.py is pure geometry, so every tile's origin is
recoverable from the frame size and the tiling config — the tile index in the
filename (`<stem>_t<ti>.jpg`) is the position in that same enumeration.

Each polygon is clipped to the tile rectangle and translated into tile
coordinates. A tile fully inside the region becomes "all", fully outside becomes
"none", and anything straddling keeps a clipped polygon.

  # locally, on the full frames
  python scripts/draw_valid_region.py --images data/real/images/test --out regions_frames.json
  python scripts/map_regions_to_tiles.py --in regions_frames.json --out regions_tiles.json \
      --images data/real/images/test
  # then on the cluster
  python scripts/ap_analysis.py --runs '...' --valid-regions regions_tiles.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import paths  # noqa: E402
from src.data_prep.tile import tile_origins  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"}


def clip_poly(poly, x0, y0, x1, y1):
    """Sutherland-Hodgman clip of a polygon against an axis-aligned rectangle."""
    def inside(p, edge):
        if edge == 0:
            return p[0] >= x0
        if edge == 1:
            return p[0] <= x1
        if edge == 2:
            return p[1] >= y0
        return p[1] <= y1

    def cut(a, b, edge):
        ax, ay = a
        bx, by = b
        if edge in (0, 1):
            xe = x0 if edge == 0 else x1
            t = (xe - ax) / (bx - ax) if bx != ax else 0.0
            return [xe, ay + t * (by - ay)]
        ye = y0 if edge == 2 else y1
        t = (ye - ay) / (by - ay) if by != ay else 0.0
        return [ax + t * (bx - ax), ye]

    out = [list(p) for p in poly]
    for edge in range(4):
        if not out:
            return []
        buf = []
        prev = out[-1]
        for cur in out:
            ci, pi = inside(cur, edge), inside(prev, edge)
            if ci:
                if not pi:
                    buf.append(cut(prev, cur, edge))
                buf.append(cur)
            elif pi:
                buf.append(cut(prev, cur, edge))
            prev = cur
        out = buf
    return out


def _area(poly) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def main() -> None:
    from PIL import Image
    from src.common.config import load_config

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, required=True,
                    help="regions JSON drawn on the FULL frames")
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    ap.add_argument("--images", type=Path, default=None,
                    help="directory of the full frames (for their sizes). "
                         "Default: data/real/images/<split>")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tile", type=int, default=None, help="default: tiling.size")
    ap.add_argument("--stride", type=int, default=None, help="default: tiling.stride")
    ap.add_argument("--ext", default=".jpg", help="tile filename extension")
    ap.add_argument("--full-frac", type=float, default=0.995,
                    help="a clipped polygon covering at least this fraction of the "
                         "tile marks it fully valid")
    args = ap.parse_args()

    t = load_config("base.yaml").get("tiling", {}) or {}
    tile = args.tile or int(t.get("size", 1536))
    stride = args.stride or int(t.get("stride", 1024))

    img_dir = args.images or (paths.REAL / "images" / args.split)
    frames = {p.stem: p for p in img_dir.iterdir() if p.suffix in IMG_EXT} \
        if img_dir.exists() else {}

    src = json.loads(args.src.read_text(encoding="utf-8"))
    regions = src.get("regions", {})
    out: dict = {}
    stats = {"all": 0, "none": 0, "poly": 0}
    missing = []

    for fname, rec in regions.items():
        stem = Path(fname).stem
        fp = frames.get(stem)
        if fp is None:
            missing.append(fname)
            continue
        with Image.open(fp) as im:
            W, H = im.size
        mode = rec.get("mode", "all")
        polys = rec.get("polys", [])

        ti = -1
        for ty in tile_origins(H, tile, stride):
            th = min(tile, H - ty)
            for tx in tile_origins(W, tile, stride):
                tw = min(tile, W - tx)
                ti += 1
                name = f"{stem}_t{ti}{args.ext}"
                if mode in ("all", "none"):
                    out[name] = {"mode": mode}
                    stats[mode] += 1
                    continue
                clipped = []
                for poly in polys:
                    c = clip_poly(poly, tx, ty, tx + tw, ty + th)
                    if _area(c) > 1.0:
                        clipped.append([[p[0] - tx, p[1] - ty] for p in c])
                cov = sum(_area(c) for c in clipped) / max(tw * th, 1)
                if not clipped:
                    out[name] = {"mode": "none"}
                    stats["none"] += 1
                elif cov >= args.full_frac:
                    out[name] = {"mode": "all"}
                    stats["all"] += 1
                else:
                    out[name] = {"mode": "poly", "polys": clipped}
                    stats["poly"] += 1

    args.dst.write_text(json.dumps(
        {"regions": out,
         "meta": {"tool": "map_regions_to_tiles", "tile": tile, "stride": stride,
                  "source": str(args.src), "n_frames": len(regions) - len(missing)}},
        indent=1), encoding="utf-8")

    print(f"[map] {len(regions) - len(missing)} frames -> {len(out)} tiles")
    print(f"[map] " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    if missing:
        print(f"[map] WARNING {len(missing)} frames in the JSON had no image in "
              f"{img_dir}: {missing[:4]}{'...' if len(missing) > 4 else ''}")
    print(f"[map] -> {args.dst}")
    print("[map] tiles with no entry are treated as fully valid by ap_analysis")


if __name__ == "__main__":
    main()
