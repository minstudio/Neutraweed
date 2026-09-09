"""Cut-and-composite generator (Stage C, Option B).

Labels are perfect by construction: we cut REAL weeds out of the real images
(using the SAM2 instance masks we already have) and paste them onto weed-free
soil backgrounds. The box IS the cutout, so pixels and label can never disagree
— the failure mode that sank the diffusion-ControlNet route.

Two phases (mirrors the SD3.5 generator so Stage D treats it the same):
  prep()      : extract per-class RGBA cutouts + soil background tiles (one-time, CPU).
  generate()  : sample a background + K cutouts, scale, blend, write image + YOLO label
                to data/synthetic/composite/{images,labels} (CPU, no GPU).

Output plugs straight into Stage D as generator 'composite'.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from ..common import classes, paths
from ..annotate.composite import Cutout, Photoreal, paste_cutouts

POOL = paths.SYNTHETIC / "composite"
CUTOUTS = POOL / "cutouts"
BACKGROUNDS = POOL / "backgrounds"


def _pool_dirs(name: str = "composite"):
    """Resolve (pool, cutouts, backgrounds) dirs for a named pool so we can build
    domain variants (e.g. 'composite2022') without clobbering the default pool."""
    pool = paths.SYNTHETIC / name
    return pool, pool / "cutouts", pool / "backgrounds"


def _matches_sources(fname: str, sources) -> bool:
    """True if fname starts with any allowed source prefix (None = accept all).
    Prefixes are the tiled filename tags: 'TOMATO_1__' = 2021, 'TOMATO_2__' = 2022."""
    return sources is None or any(fname.startswith(s) for s in sources)


# ---------------------------------------------------------------------------
# Scale prior — reproduce the real apparent box-size distribution
# ---------------------------------------------------------------------------
SCALE_PRIOR = paths.REAL / "scale_prior.json"


def load_scale_prior(path=None) -> dict | None:
    p = Path(path) if path else SCALE_PRIOR
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _prior_sampler(prior: dict, cls_name: str):
    """Inverse-CDF sampler over the class's apparent-longest-side quantiles.

    Falls back to the pooled distribution for classes the prior never saw."""
    levels = np.asarray(prior["quantile_levels"], dtype=np.float64)
    entry = prior.get("classes", {}).get(cls_name) or {}
    vals = entry.get("longest_side_q") or prior.get("all", {}).get("longest_side_q")
    if not vals:
        return None
    vals = np.asarray(vals, dtype=np.float64)

    def sample(rng) -> float:
        u = rng.random()
        return float(np.interp(u, levels, vals))

    return sample


def _cutout_index(cutouts_dir: Path, cls_name: str, paths_list) -> np.ndarray:
    """Native longest side of every cutout, cached next to the pool.

    Lets the compositor pick a cutout whose native size is already near the
    sampled target, so we hit the real size distribution by SELECTION instead of
    by up-scaling small renders into mush."""
    from PIL import Image

    cache = cutouts_dir / f".sizes_{cls_name}.json"
    known = {}
    if cache.exists():
        try:
            known = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            known = {}
    out, dirty = [], False
    for p in paths_list:
        k = p.name
        if k not in known:
            with Image.open(p) as im:
                w, h = im.size
            known[k] = max(w, h)
            dirty = True
        out.append(known[k])
    if dirty:
        try:
            cache.write_text(json.dumps(known), encoding="utf-8")
        except Exception:
            pass
    return np.asarray(out, dtype=np.float64)


def _matte_coverage(cutouts_dir: Path, cls_name: str, paths_list) -> np.ndarray:
    """Fraction of each cutout's bounding box that the matte calls plant.

    A coverage near 1.0 means the matte failed and selected the whole crop, so
    the "plant" is a rectangle of foliage-coloured pixels. Pasted with the old
    feather blend those merely looked like soft squares; with a cast shadow they
    throw a hard rectangular shadow and become the most obviously fake thing in
    the frame. Cheap to measure once and cache."""
    from PIL import Image

    cache = cutouts_dir / f".matte_{cls_name}.json"
    known = {}
    if cache.exists():
        try:
            known = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            known = {}
    out, dirty = [], False
    for p in paths_list:
        if p.name not in known:
            with Image.open(p) as im:
                a = np.asarray(im.convert("RGBA").resize((96, 96)))[:, :, 3]
            known[p.name] = float((a > 127).mean())
            dirty = True
        out.append(known[p.name])
    if dirty:
        try:
            cache.write_text(json.dumps(known), encoding="utf-8")
        except Exception:
            pass
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Phase 1 — extract real weed cutouts (RGBA) from SAM2 masks
# ---------------------------------------------------------------------------
def extract_cutouts(split: str = "train", max_per_class: int = 600, min_area: int = 600,
                    pad: int = 4, name: str = "composite", sources=None) -> dict[str, int]:
    from PIL import Image

    _, cutouts_dir, _ = _pool_dirs(name)
    masks_dir = paths.MASKS / "sam2" / split
    img_dir = paths.REAL / "images" / split
    counts = {c: 0 for c in classes.CLASS_NAMES}
    for c in classes.CLASS_NAMES:
        (cutouts_dir / c).mkdir(parents=True, exist_ok=True)

    for js in sorted(masks_dir.glob("*.json")):
        if all(counts[c] >= max_per_class for c in classes.CLASS_NAMES):
            break
        if not _matches_sources(js.stem, sources):
            continue                              # restrict cutout domain (e.g. 2021-only)
        meta = json.loads(js.read_text(encoding="utf-8"))
        uid = js.stem
        src = next(iter(img_dir.glob(f"{uid}.*")), None)
        png = js.with_suffix(".png")
        if src is None or not png.exists():
            continue
        rgb = np.asarray(Image.open(src).convert("RGB"))
        label_map = np.asarray(Image.open(png))
        H, W = label_map.shape[:2]
        for inst in meta.get("instances", []):
            c = inst["class"]
            if c not in counts or counts[c] >= max_per_class:
                continue
            box = inst.get("mask_box")
            if not box:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
            sub_lab = label_map[y1:y2, x1:x2]
            alpha = (sub_lab == inst["id"]).astype(np.uint8) * 255
            if int((alpha > 0).sum()) < min_area:
                continue
            rgba = np.dstack([rgb[y1:y2, x1:x2], alpha])
            Image.fromarray(rgba, mode="RGBA").save(cutouts_dir / c / f"{uid}_{inst['id']}.png")
            counts[c] += 1
    print(f"[{name}] cutouts extracted: {counts} -> {cutouts_dir}")
    return counts


def copy_cutouts(name: str, from_pool: str = "composite", sources=None) -> dict[str, int]:
    """Populate a pool's cutouts by copying from an already-built pool instead of
    re-extracting from SAM2 masks. The cutouts are the same real weed silhouettes
    regardless of background domain, so a 2022-background pool can reuse the
    working 'composite' pool's cutouts — no mask files needed on the cluster.

    `sources` filters by the cutout filename prefix (the source image uid), e.g.
    ['TOMATO_1__'] keeps only 2021-derived cutouts for the strict zero-cost arm."""
    import shutil

    _, dst_dir, _ = _pool_dirs(name)
    _, src_dir, _ = _pool_dirs(from_pool)
    if not src_dir.exists():
        raise SystemExit(
            f"Source pool cutouts {src_dir} not found — build '{from_pool}' first "
            f"(scripts/gen_composite.py prep) or sync it to the cluster."
        )
    if dst_dir.resolve() == src_dir.resolve():
        n = sum(len(list((src_dir / c).glob("*.png"))) for c in classes.CLASS_NAMES
                if (src_dir / c).exists())
        print(f"[{name}] cutouts already resolve to '{from_pool}' ({n} files) — "
              f"nothing to copy")
        return {c: len(list((src_dir / c).glob("*.png"))) if (src_dir / c).exists() else 0
                for c in classes.CLASS_NAMES}
    counts = {c: 0 for c in classes.CLASS_NAMES}
    for c in classes.CLASS_NAMES:
        (dst_dir / c).mkdir(parents=True, exist_ok=True)
        for png in (src_dir / c).glob("*.png"):
            if not _matches_sources(png.name, sources):
                continue
            shutil.copy2(png, dst_dir / c / png.name)
            counts[c] += 1
    print(f"[{name}] cutouts copied from '{from_pool}' "
          f"(sources={sources or 'all'}): {counts} -> {dst_dir}")
    return counts


# ---------------------------------------------------------------------------
# Phase 1 — extract weed-free soil background tiles
# ---------------------------------------------------------------------------
def _boxes_px(label_path: Path, w: int, h: int):
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 5:
            _, cx, cy, bw, bh = p
            cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
            out.append((( cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def _mean_exg(img) -> float:
    """Mean Excess-Green of a PIL crop (downscaled). Soil ~<=0; foliage >0.1."""
    a = np.asarray(img.resize((64, 64))).astype(np.float32)
    s = a.sum(axis=2) + 1e-6
    return float(((2 * a[..., 1] - a[..., 0] - a[..., 2]) / s).mean())


def extract_backgrounds(split: str = "train", size: int = 1024, per_image: int = 2,
                        max_images: int = 400, seed: int = 0, max_green: float = 0.05,
                        name: str = "composite", sources=None,
                        match_luminance: bool = False, oversample: int = 4) -> int:
    """Crop square tiles that are bare SOIL, so the only weeds in a composite are
    the ones we paste (perfect labels). A tile must (a) contain NO ground-truth
    weed box AND (b) not be too green — the crop (LYPES) was dropped from the
    labels, so tomato-foliage regions otherwise pass the no-box test and leak in
    as backgrounds.

    `sources` restricts which source images may donate backgrounds (None = all).
    Pass ['TOMATO_2__'] to harvest weed-free 2022-Finca soil (zero annotation
    cost — no boxes needed) so the composite pool carries the 2022 target domain
    instead of being 2021-only. Selection is now shuffled, not prefix-sorted, so
    the old TOMATO_1-first truncation no longer silently biases the pool to 2021.

    KNOWN BIAS, corrected by `match_luminance`. The two acceptance tests — no
    weed box AND low ExG — are not neutral with respect to brightness. In a dense
    field the regions that satisfy both are disproportionately shadowed soil,
    because shade suppresses ExG and weeds are sparser there. Measured on this
    dataset the resulting bank is 57% darker in shadow depth than a random real
    tile (0.515 vs 0.328), so every composite is built on ground darker than the
    photographs it will be trained beside, and the pasted plants read as lit
    while the ground reads as shaded.

    With `match_luminance`, candidates are collected without being written,
    scored by mean HSV value, and then subsampled so their brightness
    distribution follows that of *unconstrained* random tiles from the same
    images. `oversample` sets how many candidates are gathered per tile finally
    kept — the correction can only discard, so it needs surplus to choose from."""
    from PIL import Image

    _, _, backgrounds_dir = _pool_dirs(name)
    img_dir = paths.REAL / "images" / split
    lbl_dir = paths.REAL / "labels" / split
    # A pool built with SRC_POOL has backgrounds/ as a SYMLINK into another pool.
    # Clearing stale tiles through that link would delete the donor pool's
    # backgrounds and silently break every experiment that depends on it. Break
    # the link and start a real directory instead.
    rng = random.Random(seed)
    candidates = [p for p in sorted(img_dir.iterdir())
                  if p.is_file() and _matches_sources(p.name, sources)]
    if not candidates:
        raise SystemExit(f"[{name}] no source images in {img_dir} "
                         f"(sources={sources or 'all'})")
    random.Random(seed).shuffle(candidates)       # de-bias source-year selection
    imgs = candidates[:max_images]

    # Probe sizes BEFORE touching backgrounds/. A pool built with SRC_POOL has
    # backgrounds/ as a symlink into the donor pool, and the old code broke that
    # link first and harvested second — so a run that could never produce a tile
    # still left an empty real directory behind and took the donor's tiles out of
    # reach. The common way to get zero tiles is asking for a window larger than
    # the source images: on the cluster data/real/images/<split> is already
    # tiled, so --bg-size 1536 matches nothing. Fail here, before any damage.
    probe = []
    for p in imgs[:40]:
        try:
            with Image.open(p) as im:
                probe.append(min(im.size))
        except Exception:
            continue
    if probe and max(probe) < size:
        raise SystemExit(
            f"[{name}] every source image is smaller than --bg-size {size} "
            f"(largest short side seen: {max(probe)} px in {img_dir}).\n"
            f"  data/real/images/{split} is tiled, so a {size} px window cannot "
            f"be cut from it.\n"
            f"  Use --bg-size {max(probe)} or point --split at untiled frames. "
            f"backgrounds/ has NOT been modified.")
    if probe and min(probe) <= size:
        # x and y are drawn from randint(0, W - size), so when the window is the
        # whole image there is exactly ONE candidate per source and the retry
        # loop cannot help: the tile is either weed-free or the image is wasted.
        # probe_backgrounds already established that this field has no weed-free
        # full tiles, so this configuration harvests nothing however long it runs.
        print(f"[{name}] WARNING: --bg-size {size} equals the source tile size, "
              f"so there is one candidate window per image and no room to "
              f"reject-sample around weeds. If this yields few or no tiles, "
              f"harvest smaller (e.g. --bg-size {int(size * 2 / 3)}) and let "
              f"--bg-mode tilematch mosaic them back to tile scale.")

    # Harvest into a staging directory and swap only on success, so a failed or
    # empty run cannot destroy an existing bank or symlink.
    stage = backgrounds_dir.parent / f".{backgrounds_dir.name}.staging"
    if stage.exists():
        for f in stage.glob("*.png"):
            f.unlink()
    else:
        stage.mkdir(parents=True, exist_ok=True)
    want_per_image = per_image * (oversample if match_luminance else 1)
    tries = 30 * (oversample if match_luminance else 1)

    found: list[tuple[Path, int, int, float]] = []
    target_luma: list[float] = []
    for ip in imgs:
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        if W < size or H < size:
            continue
        boxes = _boxes_px(lbl_dir / f"{ip.stem}.txt", W, H)
        if match_luminance:
            # The reference distribution: tiles drawn with no acceptance test at
            # all, which is what a real training tile is.
            for _ in range(per_image * 2):
                rx, ry = rng.randint(0, W - size), rng.randint(0, H - size)
                target_luma.append(_mean_value(im.crop((rx, ry, rx + size, ry + size))))
        got = 0
        for _ in range(tries):
            if got >= want_per_image:
                break
            x = rng.randint(0, W - size)
            y = rng.randint(0, H - size)
            if any(not (bx2 <= x or bx1 >= x + size or by2 <= y or by1 >= y + size)
                   for bx1, by1, bx2, by2 in boxes):
                continue                          # overlaps a weed -> skip
            tile = im.crop((x, y, x + size, y + size))
            if _mean_exg(tile) > max_green:       # too green -> crop foliage, not soil
                continue
            if match_luminance:
                found.append((ip, x, y, _mean_value(tile)))
            else:
                tile.save(stage / f"{ip.stem}_{got}.png")
            got += 1

    if not match_luminance:
        n = len(list(stage.glob("*.png")))
        _commit_backgrounds(stage, backgrounds_dir, n, name, max_green, size)
        print(f"[{name}] {n} soil background tiles (max_green={max_green}, "
              f"sources={sources or 'all'}) -> {backgrounds_dir}")
        return n

    keep = _match_luma(found, target_luma, max(1, len(found) // max(oversample, 1)), rng)
    n = 0
    by_img: dict[str, int] = {}
    for ip, x, y, _v in keep:
        with Image.open(ip) as im:
            k = by_img.get(ip.stem, 0)
            im.convert("RGB").crop((x, y, x + size, y + size)).save(
                stage / f"{ip.stem}_{k}.png")
            by_img[ip.stem] = k + 1
        n += 1

    _commit_backgrounds(stage, backgrounds_dir, n, name, max_green, size)

    if found and target_luma:
        import numpy as np
        before = np.asarray([v for *_r, v in found], dtype=np.float64)
        after = np.asarray([v for *_r, v in keep], dtype=np.float64)
        tgt = np.asarray(target_luma, dtype=np.float64)
        print(f"[{name}] luminance match: bank mean {before.mean():.1f} -> "
              f"{after.mean():.1f}, real tiles {tgt.mean():.1f} "
              f"(bias {before.mean() - tgt.mean():+.1f} -> "
              f"{after.mean() - tgt.mean():+.1f})")
    print(f"[{name}] {n} soil background tiles (max_green={max_green}, "
          f"sources={sources or 'all'}, luminance-matched) -> {backgrounds_dir}")
    return n


def _commit_backgrounds(stage: Path, dest: Path, n: int, name: str,
                        max_green: float, size: int) -> None:
    """Swap a staged harvest into place, or refuse and leave the pool alone.

    Zero tiles is always a configuration error, never a valid outcome, so it must
    not be allowed to replace a working bank — or, when dest is a symlink into a
    donor pool, to put an empty directory where that pool's tiles were.
    """
    import shutil

    if n == 0:
        for f in stage.glob("*.png"):
            f.unlink()
        stage.rmdir()
        raise SystemExit(
            f"[{name}] harvested 0 soil tiles at --bg-size {size} with "
            f"--max-green {max_green}.\n"
            f"  Nothing was written; {dest} is untouched.\n"
            f"  Usual causes: the greenness gate is too strict for a dense "
            f"field (raise --max-green, e.g. 0.10), or every candidate window "
            f"overlaps a weed box.")

    if dest.is_symlink():
        target = dest.resolve()
        dest.unlink()
        print(f"[{name}] backgrounds/ was a symlink to {target}; replacing with "
              f"a real directory of {n} tiles. The donor pool is untouched, but "
              f"this pool no longer shares its bank — a comparison against it is "
              f"no longer compositor-only.")
    elif dest.exists():
        shutil.rmtree(dest)
    stage.replace(dest)


def _mean_value(tile) -> float:
    import cv2
    import numpy as np

    a = np.asarray(tile.resize((96, 96)), dtype=np.uint8)
    return float(cv2.cvtColor(a, cv2.COLOR_RGB2HSV)[..., 2].mean())


def _match_luma(found, target, k, rng):
    """Pick k of `found` so their brightness follows `target`'s distribution.

    Inverse-transform sampling without replacement: for each of k evenly spaced
    quantiles of the target, take the nearest unused candidate. Degrades
    gracefully — if the bank simply has no bright tiles, the closest available
    are chosen and the printed bias shows how far short it fell, rather than
    silently returning a shadowed bank as before.
    """
    import numpy as np

    if not found or not target or k <= 0:
        return found
    k = min(k, len(found))
    tq = np.percentile(np.asarray(target, dtype=np.float64),
                       np.linspace(2.5, 97.5, k))
    vals = np.asarray([v for *_r, v in found], dtype=np.float64)
    used = np.zeros(len(found), dtype=bool)
    out = []
    for q in tq:
        d = np.abs(vals - q)
        d[used] = np.inf
        i = int(np.argmin(d))
        if not np.isfinite(d[i]):
            break
        used[i] = True
        out.append(found[i])
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Phase 2 — composite generation
# ---------------------------------------------------------------------------
class CompositeGenerator:
    """Compose labelled scenes from cutouts + soil backgrounds.

    Two scale-matching knobs decide whether the pool is statistically usable
    next to the real tiles it will be trained beside:

    scale_mode
      'prior'   (default) sample each instance's apparent longest side from the
                measured real distribution (scripts/fit_scale_prior.py), then
                SELECT a cutout whose native size is already close to it.
      'uniform' the legacy randint(*scale_px) behaviour, kept for A/B.

    bg_mode
      'tilematch' (default) a real training tile is `tiling.size` native pixels
                shown at imgsz, i.e. a 0.667x downscale. A background crop pasted
                1:1 is therefore 1.5x too finely grained. Mosaic four crops at
                the tile's own scale so soil texture matches the real half of the
                training set.
      'native'  the legacy single-crop, no-rescale behaviour.
    """

    name = "composite"

    def __init__(self, image_size: int = 1024, n_instances=(3, 12),
                 scale_px=(60, 260), blend: str = "poisson", name: str = "composite",
                 max_upscale: float | None = None, scale_mode: str = "prior",
                 scale_prior=None, scale_gain: float = 1.0, bg_mode: str = "tilematch",
                 tile_scale: float | None = None, class_weights: str = "uniform",
                 select_tol: float = 1.6, img_format: str = "png", jpg_quality: int = 92,
                 prefer_downscale: bool = False, bg_qc: bool = False,
                 bg_max_blown: float = 0.02, photoreal: "Photoreal | None" = None,
                 cutout_qc: bool = False, min_matte: float = 0.03,
                 max_matte: float = 0.85, bg_match: bool = False):
        self.size = image_size
        self.n_instances = tuple(n_instances)
        self.scale_px = scale_px
        self.blend = blend
        self.name = name
        self.max_upscale = (2.0 if scale_mode == "prior" else 1.2) \
            if max_upscale is None else float(max_upscale)
        self.scale_mode = scale_mode
        self.scale_gain = scale_gain
        self.bg_mode = bg_mode
        self.class_weights = class_weights
        self.select_tol = select_tol
        self.img_format = img_format.lower().lstrip(".")
        self.jpg_quality = jpg_quality
        self.prefer_downscale = bool(prefer_downscale)
        self.bg_qc = bool(bg_qc)
        self.bg_max_blown = float(bg_max_blown)
        self.photoreal = photoreal
        self.cutout_qc = bool(cutout_qc)
        self.min_matte = float(min_matte)
        self.max_matte = float(max_matte)
        self.bg_match = bool(bg_match)
        self.pool_dir, self.cutouts_dir, self.backgrounds_dir = _pool_dirs(name)

        self.prior = load_scale_prior(scale_prior) if scale_mode == "prior" else None
        if scale_mode == "prior" and self.prior is None:
            raise SystemExit(
                "scale_mode='prior' needs data/real/scale_prior.json — run "
                "`python scripts/fit_scale_prior.py` first (login-safe), or pass "
                "scale_mode='uniform'.")

        if tile_scale is not None:
            self.tile_scale = float(tile_scale)
        else:
            self.tile_scale = self._tile_scale_from_config()

        self._canvas_k = self.size / (self.prior["imgsz"] if self.prior else self.size)

    @staticmethod
    def _tile_scale_from_config() -> float:
        try:
            from ..common.config import load_config
            cfg = load_config("base.yaml")
            t = cfg.get("tiling", {}) or {}
            if not t.get("enabled", False):
                print("[composite] tiling disabled in base.yaml -> tile_scale=1.0")
                return 1.0
            return float(cfg["detector"]["imgsz"]) / float(t["size"])
        except Exception as exc:
            print(f"[composite] WARNING could not read tiling/imgsz from base.yaml "
                  f"({exc}); falling back to tile_scale=1.0, so --bg-mode tilematch "
                  f"is a NO-OP. Pass tile_scale= explicitly if this is wrong.")
            return 1.0

    def _load_banks(self):
        cutouts = {c: sorted((self.cutouts_dir / c).glob("*.png")) for c in classes.CLASS_NAMES}
        cutouts = {c: v for c, v in cutouts.items() if v}
        if not cutouts:
            raise SystemExit(f"No cutouts in {self.cutouts_dir} — run prep (extract_cutouts) first.")
        if self.cutout_qc:
            kept, dropped = {}, 0
            for c, v in cutouts.items():
                cov = _matte_coverage(self.cutouts_dir, c, v)
                ok = (cov >= self.min_matte) & (cov <= self.max_matte)
                dropped += int((~ok).sum())
                if ok.any():
                    kept[c] = [v[i] for i in np.flatnonzero(ok)]
                else:
                    print(f"[{self.name}] matte QC would empty {c} — keeping it unfiltered")
                    kept[c] = v
            print(f"[{self.name}] matte QC: kept {sum(len(v) for v in kept.values())}, "
                  f"dropped {dropped} outside coverage "
                  f"[{self.min_matte:.2f}, {self.max_matte:.2f}]")
            cutouts = kept
        bgs = sorted(self.backgrounds_dir.glob("*.png"))
        if not bgs:
            raise SystemExit(f"No backgrounds in {self.backgrounds_dir} — run prep (extract_backgrounds) first.")
        if self.bg_qc:
            bgs = self._qc_backgrounds(bgs)
        return cutouts, bgs

    def _qc_backgrounds(self, bgs: list[Path]) -> list[Path]:
        """Drop soil tiles containing blown highlights.

        extract_backgrounds only tests for green, so a tile holding a white
        reference panel, a plastic marker or a specular blowout passes and then
        shows up as a bright band across one mosaic quadrant. Those bands are the
        most obviously non-field thing in the pool."""
        from PIL import Image

        import cv2

        # The old test was `min(RGB) > 245`, i.e. fully blown pixels only.
        # Measured on this bank, tiler padding sits at V 233-248 with at least
        # one channel below 245, so it scored 0.96% against a 2% threshold and
        # passed — then covered 6% of a composited frame in flat white bands at
        # the top and bottom edges. Photos are cut into 1536 px tiles at stride
        # 1024, so tiles on a photo's right and bottom edge run past the image
        # and are padded; those tiles carry no weed boxes and no green, which is
        # exactly what the harvest looks for.
        #
        # What distinguishes padding, a reference panel or a plastic marker from
        # bright soil is not brightness alone. It is bright AND colourless AND
        # textureless together — sunlit pale soil is bright but keeps both its
        # colour cast and its grain.
        cache = self.backgrounds_dir / f".qc_foreign_{self.bg_max_blown:.3f}.json"
        known = {}
        if cache.exists():
            try:
                known = json.loads(cache.read_text(encoding="utf-8"))
            except Exception:
                known = {}
        keep, dropped, dirty = [], 0, False
        for p in bgs:
            if p.name not in known:
                with Image.open(p) as im:
                    a = np.asarray(im.convert("RGB").resize((256, 256))).astype(np.uint8)
                hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
                V = hsv[:, :, 2].astype(np.float32)
                S = hsv[:, :, 1].astype(np.float32)
                g = a.astype(np.float32) @ np.asarray([0.299, 0.587, 0.114], np.float32)
                tex = cv2.blur(np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.2)), (9, 9))
                foreign = (V > 230) & (S < 30) & (tex < 3.0)
                blown = a.astype(np.float32).min(axis=2) > 245
                known[p.name] = float(np.maximum(foreign, blown).mean())
                dirty = True
            if known[p.name] <= self.bg_max_blown:
                keep.append(p)
            else:
                dropped += 1
        if dirty:
            try:
                cache.write_text(json.dumps(known), encoding="utf-8")
            except Exception:
                pass
        print(f"[{self.name}] background QC: kept {len(keep)}, dropped {dropped} "
              f"with >{self.bg_max_blown:.1%} pale/colourless/textureless pixels "
              f"(tiler padding, panels, blowouts)")
        if not keep:
            raise SystemExit("background QC rejected every tile — raise --bg-max-blown")
        return keep

    def _class_weights(self, usable: list[str]) -> list[float]:
        if self.class_weights == "uniform" or not self.prior:
            return [1.0] * len(usable)
        shares = []
        for c in usable:
            e = self.prior.get("classes", {}).get(c) or {}
            shares.append(max(float(e.get("share") or 0.0), 1e-3))
        if self.class_weights == "real":
            return shares
        if self.class_weights == "inverse":
            inv = [1.0 / s for s in shares]
            m = sum(inv)
            return [x / m for x in inv]
        raise SystemExit(f"unknown class_weights={self.class_weights!r}")

    def generate(self, n: int, seed: int = 0) -> int:
        from PIL import Image

        cutouts, bgs = self._load_banks()
        (self.pool_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.pool_dir / "labels").mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        usable = list(cutouts)
        weights = self._class_weights(usable)
        samplers = {c: (_prior_sampler(self.prior, c) if self.prior else None) for c in usable}
        sizes = {c: _cutout_index(self.cutouts_dir, c, cutouts[c]) for c in usable}

        drawn = {c: [] for c in usable}
        self._clamped = {c: [0, 0] for c in usable}
        self._upscale = {c: [] for c in usable}
        for i in range(n):
            bg = self._background(bgs, rng)
            k = rng.randint(*self.n_instances)
            cuts: list[Cutout] = []
            for _ in range(k):
                c = rng.choices(usable, weights=weights, k=1)[0]
                target = self._target(c, samplers, rng)
                path = self._pick(cutouts[c], sizes[c], target, rng)
                rgba = np.asarray(Image.open(path).convert("RGBA"))
                native = max(rgba.shape[0], rgba.shape[1])
                self._clamped[c][1] += 1
                if target > native * self.max_upscale:
                    self._clamped[c][0] += 1
                self._upscale[c].append(min(target, native * self.max_upscale) / max(native, 1))
                rgba = self._scale(rgba, rng, target)
                drawn[c].append(max(rgba.shape[0], rgba.shape[1]))
                cuts.append(Cutout(rgba=rgba, cls_name=c))
            stem = f"comp_{seed:02d}_{i:06d}"
            out_img = self.pool_dir / "images" / f"{stem}.{self.img_format}"
            out_lbl = self.pool_dir / "labels" / f"{stem}.txt"
            canvas, _ = paste_cutouts(bg, cuts, out_lbl, seed=seed * 100000 + i,
                                      blend=self.blend, photoreal=self.photoreal)
            if self.img_format in ("jpg", "jpeg"):
                Image.fromarray(canvas).save(out_img, quality=self.jpg_quality)
            else:
                Image.fromarray(canvas).save(out_img)

        self._report(drawn)
        from ..annotate.composite import opacity_report
        opacity_report(self.name)
        print(f"[{self.name}] generated {n} images -> {self.pool_dir / 'images'} "
              f"(scale={self.scale_mode}, bg={self.bg_mode}, cls={self.class_weights})")
        return n

    def _report(self, drawn: dict) -> None:
        print(f"[{self.name}] pasted longest side, canvas px "
              f"(target = real apparent size x {self._canvas_k:.3f})")
        print(f"{'class':8s}{'n':>7s}{'p10':>7s}{'p50':>7s}{'p90':>7s}"
              f"{'real p50':>10s}{'real p90':>10s}{'clamped':>9s}")
        worst = 0.0
        for c, v in drawn.items():
            if not v:
                continue
            a = np.asarray(v, dtype=np.float64)
            r50 = r90 = float("nan")
            if self.prior:
                e = self.prior.get("classes", {}).get(c) or {}
                q = e.get("longest_side_q")
                if q:
                    lv = np.asarray(self.prior["quantile_levels"], dtype=np.float64)
                    r50 = float(np.interp(0.5, lv, q)) * self._canvas_k
                    r90 = float(np.interp(0.9, lv, q)) * self._canvas_k
            hit, tot = getattr(self, "_clamped", {}).get(c, (0, 0))
            frac = (hit / tot) if tot else 0.0
            worst = max(worst, frac)
            print(f"{c:8s}{a.size:7d}" + "".join(f"{x:7.0f}" for x in np.percentile(a, [10, 50, 90]))
                  + f"{r50:10.0f}{r90:10.0f}{frac:8.1%}")
        if worst > 0.05:
            print(f"[{self.name}] NOTE {worst:.0%} of instances in the worst class hit "
                  f"max_upscale={self.max_upscale}: the cutout bank has no renders big "
                  f"enough for the top of the real size distribution, so the large tail "
                  f"is still under-covered. Use a higher-resolution cutout pool, or raise "
                  f"--max-upscale.")

        # Every factor above 1.0 is a cutout enlarged past its rendered size, i.e.
        # invented pixels. That is what makes a pasted plant look softer than the
        # soil around it, and it is invisible in the size table above because the
        # pasted size can be perfectly correct while the detail is not there.
        up = getattr(self, "_upscale", {})
        if any(up.values()):
            print(f"[{self.name}] resize factor applied to cutouts "
                  f"(>1 = upscaled = blurred)")
            print(f"{'class':8s}{'p50':>7s}{'p90':>7s}{'max':>7s}{'>1.0':>8s}{'>1.5':>8s}")
            for c, v in up.items():
                if not v:
                    continue
                a = np.asarray(v, dtype=np.float64)
                print(f"{c:8s}" + "".join(f"{x:7.2f}" for x in np.percentile(a, [50, 90]))
                      + f"{a.max():7.2f}{(a > 1.0).mean():8.1%}{(a > 1.5).mean():8.1%}")
            allv = np.concatenate([np.asarray(v) for v in up.values() if v])
            if (allv > 1.0).mean() > 0.25:
                print(f"[{self.name}] NOTE {(allv > 1.0).mean():.0%} of instances are "
                      f"upscaled. Pass --prefer-downscale to select bigger cutouts "
                      f"first, and/or render the bank at a larger --render-size.")

    def _target(self, cls_name: str, samplers, rng) -> float:
        if self.scale_mode == "uniform" or samplers.get(cls_name) is None:
            return float(rng.randint(*self.scale_px))
        t = samplers[cls_name](rng) * self._canvas_k * self.scale_gain
        return float(min(max(t, 8.0), self.size * 0.95))

    def _pick(self, paths_list, native_sizes: np.ndarray, target: float, rng):
        if self.scale_mode == "uniform" or native_sizes.size == 0:
            return rng.choice(paths_list)
        lo, hi = target / self.select_tol, target * self.select_tol
        if self.prefer_downscale:
            # Downscaling throws detail away; upscaling invents it. Given a choice
            # of cutouts that land in the tolerance window, take one already at
            # least as large as the target, so the resize is a downsample.
            idx = np.flatnonzero((native_sizes >= target * 0.98) & (native_sizes <= hi))
            if idx.size == 0:
                idx = np.flatnonzero(native_sizes >= target * 0.98)
            if idx.size:
                return paths_list[int(idx[rng.randrange(len(idx))])]
        idx = np.flatnonzero((native_sizes >= lo) & (native_sizes <= hi))
        if idx.size == 0:
            d = np.abs(native_sizes - target)
            idx = np.argsort(d)[: min(8, native_sizes.size)]
        return paths_list[int(idx[rng.randrange(len(idx))])]

    def _scale(self, rgba: np.ndarray, rng, target: float) -> np.ndarray:
        from PIL import Image
        h, w = rgba.shape[:2]
        target = min(target, max(h, w) * self.max_upscale)
        s = target / max(h, w)
        nh, nw = max(8, int(round(h * s))), max(8, int(round(w * s)))
        return np.asarray(Image.fromarray(rgba).resize((nw, nh), Image.LANCZOS))

    def _background(self, bgs, rng) -> np.ndarray:
        from PIL import Image
        if self.bg_mode == "crop":
            return self._crop_background(bgs, rng)
        if self.bg_mode != "tilematch":
            return np.asarray(
                Image.open(rng.choice(bgs)).convert("RGB").resize((self.size, self.size)))
        return _mosaic_background(bgs, rng, self.size, self.tile_scale,
                                  match=self.bg_match)

    def _crop_background(self, bgs, rng) -> np.ndarray:
        """One native crop of `size / tile_scale` px, downscaled to `size`.

        This is byte-for-byte the pipeline a real training tile goes through
        (1536 px window, LANCZOS to 1024), so the soil arrives with the right
        grain AND without the 2x2 structure the mosaic imposes. Measured
        blockiness — the share of a frame's brightness variation explained by a
        piecewise-constant 2x2 step — is 0.40 for the mosaic against 0.20 for
        real tiles; a single crop scores 0.16. That axis-aligned blockiness is
        the straight seam you can see running through the mosaic pools.

        Needs background tiles at least `size / tile_scale` px, i.e. re-run prep
        with --bg-size 1536. Falls back to the mosaic, loudly, if they are
        smaller."""
        from PIL import Image

        src = int(round(self.size / max(self.tile_scale, 1e-6)))
        for _ in range(12):
            p = rng.choice(bgs)
            with Image.open(p) as im:
                im = im.convert("RGB")
                W, H = im.size
                if min(W, H) < src:
                    continue
                x, y = rng.randint(0, W - src), rng.randint(0, H - src)
                return np.asarray(im.crop((x, y, x + src, y + src))
                                  .resize((self.size, self.size), Image.LANCZOS))
        if not getattr(self, "_warned_crop", False):
            self._warned_crop = True
            print(f"[{self.name}] WARNING --bg-mode crop needs background tiles of at "
                  f"least {src}px but this bank is smaller; falling back to the "
                  f"mosaic. Re-run prep with --bg-size {src}.")
        return _mosaic_background(bgs, rng, self.size, self.tile_scale)


def add_scene_args(p) -> None:
    """Scene-statistics flags shared by gen_composite.py and gen_sd35_cutout.py."""
    p.add_argument("--scale-mode", default="prior", choices=["prior", "uniform"],
                   help="'prior' matches the measured real box-size distribution "
                        "(needs data/real/scale_prior.json); 'uniform' = legacy "
                        "randint(60,260).")
    p.add_argument("--scale-prior", default=None,
                   help="override path to scale_prior.json")
    p.add_argument("--scale-gain", type=float, default=1.0,
                   help="multiply every sampled target size (A/B knob, default 1.0)")
    p.add_argument("--bg-mode", default="tilematch",
                   choices=["tilematch", "native", "crop"],
                   help="'crop' takes ONE imgsz/tile_scale window and downscales "
                        "it, exactly as a real tile is made — no 2x2 seam. Needs "
                        "background tiles >= that size (prep --bg-size 1536). "
                        "'tilematch' mosaics four crops (legacy, seams). "
                        "'native' = legacy 1:1 crop.")
    p.add_argument("--class-weights", default="uniform",
                   choices=["uniform", "real", "inverse"],
                   help="per-instance class sampling: uniform (legacy), real "
                        "(match train frequencies), inverse (boost rare species).")
    p.add_argument("--n-instances", type=int, nargs=2, default=[3, 12],
                   metavar=("LO", "HI"), help="instances per scene (default 3 12; "
                                              "real tiles average ~3.2).")
    p.add_argument("--max-upscale", type=float, default=None,
                   help="cap on enlarging a cutout past its native size "
                        "(default 2.0 in prior mode, 1.2 in uniform mode).")
    p.add_argument("--tile-scale", type=float, default=None,
                   help="soil scale = training imgsz / tiling.size. Read from "
                        "base.yaml by default (1024/1536=0.667). Pass 1.0 when "
                        "building a pool for training AT the tile's native size, "
                        "so the soil is not upscaled.")
    p.add_argument("--select-tol", type=float, default=1.6,
                   help="pick a cutout whose native size is within this factor of "
                        "the sampled target before resampling.")
    p.add_argument("--img-format", default="png", choices=["png", "jpg"],
                   help="pool image format. Default png matches every existing "
                        "pool. Real tiles are saved as jpg q92, so 'jpg' removes a "
                        "real-vs-synthetic compression cue — but only compare a jpg "
                        "pool against another jpg pool.")
    p.add_argument("--jpg-quality", type=int, default=92,
                   help="JPEG quality when --img-format jpg (92 matches the tiler).")
    p.add_argument("--prefer-downscale", action="store_true",
                   help="pick a cutout at least as large as the sampled target so "
                        "the resize is a downsample. Upscaled cutouts are the main "
                        "reason pasted plants look softer than the soil.")
    p.add_argument("--bg-qc", action="store_true",
                   help="drop background tiles containing blown highlights (white "
                        "panels, specular blowouts) that show as bright bands.")
    p.add_argument("--bg-max-blown", type=float, default=0.02,
                   help="max fraction of near-white pixels a background tile may "
                        "have under --bg-qc.")
    p.add_argument("--bg-match", action="store_true",
                   help="normalise each mosaic quadrant to a common exposure. "
                        "Without it the four soil crops come from four different "
                        "photographs, and the brightness step between quadrants "
                        "registers as a large phantom cast shadow.")
    p.add_argument("--cutout-qc", action="store_true",
                   help="drop cutouts whose matte covers almost none or almost "
                        "all of the crop. A matte that grabbed the whole crop is "
                        "a rectangle, and with cast shadows on it becomes the "
                        "most obviously fake object in the frame.")
    p.add_argument("--min-matte", type=float, default=0.03,
                   help="min plant coverage of the crop under --cutout-qc.")
    p.add_argument("--max-matte", type=float, default=0.85,
                   help="max plant coverage of the crop under --cutout-qc.")
    p.add_argument("--shadow-strength", type=float,
                   default=Photoreal.shadow_strength,
                   help="peak darkening of the cast shadow (blend=photoreal). "
                        "Default tracks the calibrated value on Photoreal; it "
                        "used to be hardcoded at 0.45 here, which silently "
                        "overrode the calibration for every pool built.")
    p.add_argument("--no-shadow", action="store_true",
                   help="disable cast shadows even with blend=photoreal (ablation).")
    p.add_argument("--no-noise-match", action="store_true",
                   help="disable soil-matched grain on the plant (ablation).")
    p.add_argument("--max-overlap", type=float, default=0.15,
                   help="max box overlap as a fraction of the smaller box when "
                        "placing instances; 1.0 restores the legacy free-for-all.")
    p.add_argument("--feather-px", type=float, default=0.8,
                   help="edge feather in canvas pixels (blend=photoreal). Fixed, "
                        "unlike the legacy 4%%-of-size feather that hollowed out "
                        "thin leaves.")
    p.add_argument("--no-relight", action="store_true",
                   help="disable plant relighting (ablation). With relighting on, "
                        "each plant is shaded by the same sun vector that throws "
                        "its shadow, instead of keeping the flat frontal light "
                        "the diffusion model rendered it under.")
    p.add_argument("--relight-prior", type=Path,
                   default=paths.REAL / "appearance_prior.json",
                   help="targets measured by scripts/fit_appearance_prior.py. "
                        "Supplies plant self-shading contrast and the plant-to-"
                        "soil brightness ratio. Missing file = relight with the "
                        "built-in strength and no ratio target.")
    p.add_argument("--sun-elev", type=float, default=0.55,
                   help="sun height for relighting, 0 grazing .. 1 overhead.")


def _load_relight_prior(path):
    """Read fol_contrast / rel_v / rel_s medians, if they have been measured."""
    if path is None or not Path(path).exists():
        return None, None, None
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None, None, None
    real = d.get("real", d)

    def _p50(key):
        e = real.get(key)
        if isinstance(e, dict) and "p50" in e:
            return float(e["p50"])
        return None

    return _p50("fol_contrast"), _p50("rel_v"), _p50("rel_s")


def photoreal_kwargs(args):
    """Build the Photoreal config, or None when the blend does not use it."""
    from ..annotate.composite import Photoreal

    if getattr(args, "blend", None) != "photoreal":
        return None
    contrast, rel_v, rel_s = _load_relight_prior(
        getattr(args, "relight_prior", None))
    if contrast is None and not getattr(args, "no_relight", False):
        print("[photoreal] no appearance prior found; relighting with the "
              "built-in strength and no plant/soil ratio target. Run "
              "scripts/fit_appearance_prior.py --split train to measure them.")
    else:
        print(f"[photoreal] relight targets: contrast={contrast}, "
              f"rel_v={rel_v}, rel_s={rel_s}")
    return Photoreal(
        feather_px=args.feather_px,
        shadow=not args.no_shadow,
        shadow_strength=args.shadow_strength,
        noise_match=not args.no_noise_match,
        max_overlap=args.max_overlap,
        relight=not getattr(args, "no_relight", False),
        sun_elev=getattr(args, "sun_elev", 0.55),
        target_contrast=contrast,
        target_rel_v=rel_v,
        target_rel_s=rel_s,
    )


def scene_kwargs(args) -> dict:
    return {
        "scale_mode": args.scale_mode,
        "scale_prior": args.scale_prior,
        "scale_gain": args.scale_gain,
        "bg_mode": args.bg_mode,
        "class_weights": args.class_weights,
        "n_instances": tuple(args.n_instances),
        "max_upscale": args.max_upscale,
        "select_tol": args.select_tol,
        "img_format": args.img_format,
        "jpg_quality": args.jpg_quality,
        "tile_scale": args.tile_scale,
        "prefer_downscale": args.prefer_downscale,
        "bg_qc": args.bg_qc,
        "bg_max_blown": args.bg_max_blown,
        "photoreal": photoreal_kwargs(args),
        "cutout_qc": args.cutout_qc,
        "min_matte": args.min_matte,
        "max_matte": args.max_matte,
        "bg_match": args.bg_match,
    }


def _grain(patch: np.ndarray) -> float:
    """High-frequency energy — how pebbly a soil crop is."""
    import cv2

    g = patch @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    return float(np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.2)).mean())


def _pick_matched_grain(cand, k: int):
    """Keep the k candidates whose grain agrees most closely.

    Sorting by grain and taking the tightest window of k is the cheapest way to
    get a consistent set: any window that mixes a flat crop with a pebbly one has
    a worse max/min ratio than one that does not, so the minimum-ratio window
    never contains an outlier unless every candidate is an outlier.
    """
    if len(cand) <= k:
        return cand
    order = sorted(range(len(cand)), key=lambda i: _grain(cand[i]))
    gs = [_grain(cand[i]) for i in order]
    best, lo = None, 0
    for s in range(len(order) - k + 1):
        ratio = gs[s + k - 1] / max(gs[s], 1e-6)
        if best is None or ratio < best:
            best, lo = ratio, s
    return [cand[i] for i in order[lo:lo + k]]


def _match_quadrants(patches, sd_clip: tuple[float, float] = (0.9, 1.1)):
    """Remove the exposure step between mosaic quadrants, and nothing else.

    The step is a difference in MEAN — four crops from four photographs at four
    exposures. Matching the standard deviation too was overreach: it rescales
    real contrast, and on a low-variance crop the gain is large enough to push
    shadowed soil below zero, where it clips to black.

    So: shift each quadrant onto the common mean, and allow only a token spread
    correction inside `sd_clip`. The anchor is the MEDIAN across quadrants
    rather than whichever was drawn first, so the result does not depend on
    draw order and no single odd crop can drag the other three.
    """
    mus = np.stack([p.reshape(-1, 3).mean(0) for p in patches])
    sds = np.stack([p.reshape(-1, 3).std(0) for p in patches]) + 1e-6
    t_mu = np.median(mus, axis=0)
    t_sd = np.median(sds, axis=0)

    out = []
    for p, mu, sd in zip(patches, mus, sds):
        g = np.clip(t_sd / sd, sd_clip[0], sd_clip[1])
        out.append(np.clip((p - mu) * g + t_mu, 0, 255))
    return out


def _mosaic_background(bgs, rng, size: int, tile_scale: float, overlap: int = 128,
                       match: bool = False, same_source: bool = True) -> np.ndarray:
    """A `size`-px canvas whose soil is rendered at the real tiles' scale.

    Each quadrant shows `out_px / tile_scale` native pixels downscaled to out_px,
    so the grain matches a tiling.size window shown at imgsz. Quadrants overlap
    and cross-fade, which also buys extra background diversity per image."""
    from PIL import Image, ImageOps

    out_px = size // 2 + overlap // 2
    src_px = int(round(out_px / tile_scale))
    ov = 2 * out_px - size
    if ov < 8:
        ov = 8

    ramp = np.linspace(0.0, 1.0, ov, dtype=np.float32)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    wsum = np.zeros((size, size, 1), dtype=np.float32)

    corners = [(y0, x0) for y0 in (0, size - out_px) for x0 in (0, size - out_px)]

    # All four quadrants from ONE source photograph.
    #
    # KNOWN DEFECT, fixed here. The quadrants used to be drawn independently, so
    # a frame combined four different photographs taken at four different camera
    # distances. Measured on a v7 frame, the characteristic grain size of the
    # soil was 3.0 px in one quadrant and 10.0 px in another — a 3.3x change in
    # ground scale inside a single image, which cannot happen in a real
    # photograph and is the reason the soil reads as different ground either
    # side of the frame. Unlike an exposure step, scale cannot be corrected
    # afterwards.
    #
    # The bank names tiles `{source_photo}_{k}.png`, so grouping by stem gives
    # the photograph each tile came from. Crop positions are still random and
    # each quadrant is still flipped independently, so the quadrants differ —
    # they just agree about how far away the camera was.
    pool = bgs
    if same_source:
        by_src: dict[str, list] = {}
        for b in bgs:
            by_src.setdefault(Path(b).stem.rsplit("_", 1)[0], []).append(b)
        pool = by_src[rng.choice(sorted(by_src))]

    # Draw more candidates than needed, then keep the four whose GRAIN agrees.
    #
    # KNOWN DEFECT, fixed here. The harvest keeps tiles with no weed box and low
    # greenness, and in a dense field the regions passing both are
    # disproportionately smooth, featureless soil. Drawing quadrants blind
    # therefore lands a flat crop next to a grainy one often enough to be the
    # dominant visual artefact: measured on a v9 frame, a 455x477 region at
    # (452, 490) — 9.5% of the frame, one whole quadrant — had texture 4.27
    # against a frame median of 10.28. It reads as a pale flat panel with hard
    # edges, because the eye interprets loss of grain as a highlight.
    #
    # A flat crop cannot be repaired after the fact, so it has to not be picked.
    # Candidates are scored by high-frequency energy and the tightest-agreeing
    # group of four is kept, which enforces consistency within a frame without
    # needing a global threshold or a clean bank.
    cand = []
    for _ in range(max(4, 3 * len(corners))):
        with Image.open(rng.choice(pool)) as im:
            im = im.convert("RGB")
            W, H = im.size
            s = min(src_px, W, H)
            x = rng.randint(0, W - s)
            y = rng.randint(0, H - s)
            patch = im.crop((x, y, x + s, y + s)).resize((out_px, out_px), Image.LANCZOS)
        if rng.random() < 0.5:
            patch = ImageOps.mirror(patch)
        if rng.random() < 0.5:
            patch = ImageOps.flip(patch)
        cand.append(np.asarray(patch, dtype=np.float32))

    patches = _pick_matched_grain(cand, len(corners))

    if match:
        patches = _match_quadrants(patches)
    # One exposure for the whole frame. Four soil crops from four different
    # photographs arrive at four different exposures; the cross-fade hides the
    # edge but not the step, so a ring of soil spanning two quadrants contains a
    # brightness jump that reads as a large cast shadow. Measured on the v7 pool
    # with cast shadows switched OFF: shadow depth still +70% and shadow area
    # +81% against real. Normalising each quadrant to a common mean and spread
    # removes the step, and with it both that phantom shadow and the 2x2
    # blockiness. Only relevant when a single native-size crop is impossible,
    # which in this field it is: there are no weed-free 1536 px windows.
    #
    # KNOWN DEFECT, fixed in _match_quadrants. The first version matched mean AND
    # standard deviation, anchored on whichever quadrant was drawn first:
    #     a = clip((a - mu) / sd * target_sd + target_mu, 0, 255)
    # `target_sd / sd` is unbounded, so a flat soil crop matched to a
    # high-contrast anchor had its contrast multiplied by up to ~7x. Real soil
    # carries shadowed patches; amplified, those went below zero and clipped to
    # PURE BLACK. Measured on this pool: 1.096% of every frame was near-black
    # against 0.067% for the same scenes composited without it, and cast shadows
    # accounted for only 0.014% of that. It also drove 2x2 blockiness to 0.020,
    # below the 0.027 floor of any real tile, by erasing the genuine variation
    # between quadrants. One flag, both anomalies.

    # Choose between quadrants, never average them.
    #
    # KNOWN DEFECT, fixed here. The quadrants used to be cross-faded with a
    # linear ramp, i.e. the overlap was the MEAN of two soil crops. Averaging two
    # uncorrelated textures divides their variance by two, so the overlap came
    # out ~30% smoother than the rest of the frame — visible as pale, flat bands
    # through the middle of every image. Measured on a v9 frame: texture 7.27 in
    # the band against a 10.35 median, at y 501-552 and x 519-542, which is the
    # centre of the fade region the ramp defines. Same brightness, no texture.
    #
    # A cross-fade is the right idea for hiding an exposure step and the wrong
    # tool for soil. Instead every output pixel takes its value from exactly ONE
    # quadrant, chosen at random with probability equal to what its blend weight
    # used to be. Full texture amplitude survives everywhere. The choice is
    # driven by a blurred noise field rather than per-pixel white noise, so the
    # transition is an irregular organic boundary rather than static — which is
    # also closer to how soil actually varies than a straight line would be.
    W = np.zeros((4, size, size), dtype=np.float32)
    for j, (y0, x0) in enumerate(corners):
        wx = np.ones(out_px, dtype=np.float32)
        wy = np.ones(out_px, dtype=np.float32)
        if x0 > 0:
            wx[:ov] = ramp
        else:
            wx[out_px - ov:] = ramp[::-1]
        if y0 > 0:
            wy[:ov] = ramp
        else:
            wy[out_px - ov:] = ramp[::-1]
        W[j, y0:y0 + out_px, x0:x0 + out_px] = wy[:, None] * wx[None, :]

    import cv2
    P = W / np.maximum(W.sum(0, keepdims=True), 1e-6)
    u = np.random.default_rng(rng.randrange(1 << 31)).random((size, size)).astype(np.float32)
    u = cv2.GaussianBlur(u, (0, 0), 6.0)
    u = (u - u.min()) / max(float(np.ptp(u)), 1e-6)

    pick = np.zeros((size, size), dtype=np.int8)
    acc = np.zeros((size, size), dtype=np.float32)
    assigned = np.zeros((size, size), dtype=bool)
    for j in range(4):
        acc += P[j]
        take = (~assigned) & (u <= acc)
        pick[take] = j
        assigned |= take
    pick[~assigned] = 3

    out = np.zeros((size, size, 3), dtype=np.float32)
    for j, (y0, x0) in enumerate(corners):
        m = pick == j
        full = np.zeros((size, size, 3), dtype=np.float32)
        full[y0:y0 + out_px, x0:x0 + out_px] = patches[j]
        out[m] = full[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def prep(split: str = "train", name: str = "composite",
         bg_sources=None, cut_sources=None, cutouts_from=None,
         max_green: float = 0.05, per_image: int = 2, max_images: int = 400,
         bg_size: int = 1024) -> None:
    if cutouts_from:
        copy_cutouts(name=name, from_pool=cutouts_from, sources=cut_sources)
    else:
        extract_cutouts(split, name=name, sources=cut_sources)
    extract_backgrounds(split, name=name, sources=bg_sources, size=bg_size,
                        max_green=max_green, per_image=per_image, max_images=max_images)
