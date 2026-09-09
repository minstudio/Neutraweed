"""Cut-and-composite route (Stage C).

Segment REAL weeds (not generated ones — segmenters are reliable on real images),
cut them out, and paste onto varied backgrounds. Labels are PERFECT because we
place each cutout ourselves.

Blend modes, oldest to newest:

  alpha      straight alpha composite. Visible paste seam.
  poisson    cv2.seamlessClone. Preserves gradients but drags the plant's
             absolute colour toward the soil — small weeds go brown.
  feather    alpha composite with a feathered edge + clipped exposure match.
             Keeps hue/saturation. This is what pools v2..v6 used.
  photoreal  feather, plus the four things that make a composited frame read as
             fake even when every individual cutout is good:

             1. SEMI-TRANSPARENT PLANTS. `feather` blurs the alpha by 4% of the
                cutout's own size — for a 250 px cutout that is a 21x21 kernel.
                A grass blade 6 px across is thinner than the kernel, so its
                peak alpha collapses to ~0.3 and you can see soil straight
                through it. photoreal hardens the matte first and then feathers
                by a fixed 1-2 px, which is anti-aliasing, not translucency.
                This is the single most visible defect in the v2-v6 pools and it
                hits exactly the classes made of thin structures: ECHCG, SETVE,
                CYPRO.
             2. NO CAST SHADOW. Every real plant in this field throws a hard
                shadow; pasted ones throw none, so they read as stickers.
                photoreal draws a shadow per instance from ONE sun vector
                sampled per scene, so the whole frame agrees on where the sun
                is. The shadow multiplies the soil rather than painting over it,
                so soil texture survives, and it removes more red than blue
                because shadow is lit by sky.
             3. TOO CLEAN. The soil carries sensor noise and the 1536->1024
                tile downscale; an SD render pasted on top does not. photoreal
                measures the high-frequency energy of the surrounding soil and
                adds matched noise to the plant.
             4. IMPOSSIBLE OVERLAPS. Positions were drawn uniformly with no
                rejection, so cutouts stack on top of each other while both keep
                a full box. photoreal rejection-samples positions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import masks


@dataclass
class Cutout:
    rgba: np.ndarray          # HxWx4, alpha = matte of a REAL weed
    cls_name: str


@dataclass
class Photoreal:
    """Knobs for blend='photoreal'. Defaults are the ones used to build v7."""

    # Plants are opaque. The matte decides WHERE the plant is; it must not also
    # decide how solid it is. matte_thresh binarises it, holes are filled, and
    # the only fractional alpha in the result is a feather_px-wide rim on the
    # silhouette — anti-aliasing, which is the one place a real edge really is
    # partly transparent. Nothing inside the outline can be see-through at any
    # threshold setting, so there is nothing here to tune per pool.
    matte_thresh: float = 0.35    # fraction of the cutout's own max alpha
    fill_holes: bool = True
    # Ceiling on what hole-filling may add, as a fraction of the crop's area.
    # Real leaf interiors are a few percent; a fill that adds more than this has
    # sealed the background, not a hole, and is discarded. See _opaque.
    fill_max_growth: float = 0.15
    # Repaint the transparent region with the nearest plant colour before
    # blending, so the feather rim never mixes in a cutout's background —
    # black for rembg-style mattes, soil for ExG ones.
    edge_extend: bool = True
    min_blob_px: int = 12         # drop specks the matte picked up off the soil
    alpha_lo: float = 0.35        # legacy ramp, used only if fill_holes=False
    alpha_hi: float = 0.70
    # 0.8 px, not 1.2. The feather is a rim, so its cost scales with how thin the
    # structure is: on a 140 px leaf both settings retain 100% of the plant's
    # colour, but on a 6 px grass blade 1.2 px retains 96% and 0.8 px retains
    # 99.6%. ECHCG, SETVE and CYPRO are made of 6 px blades.
    feather_px: float = 0.8       # fixed, in canvas pixels — NOT a fraction of size
    shadow: bool = True
    # Calibrated, not guessed. scripts/fit_appearance_prior.py measures shadow
    # depth and area around real annotated plants; the first version of this
    # code used strength 0.45 with offsets up to 0.45x the plant size and
    # overshot real depth by 34% and real shadow area by 78%, which is why the
    # first photoreal pool looked worse than the thing it replaced. The soil
    # crops are real photographs and already carry real shadow, so what is
    # added here has to be small.
    shadow_strength: float = 0.25     # peak darkening under the plant
    # Contact shadow, not a displaced replica.
    #
    # The offset used to be 0.08-0.22 of the plant's LATERAL size, applied as a
    # rigid translation of the whole silhouette. Two things are wrong with that.
    # Displacement should track height above the ground, and these are prostrate
    # seedlings photographed from above — a purslane rosette is a couple of
    # centimetres tall and 20 cm across, so its shadow sits essentially under it.
    # And translating a long thin blade sideways produces a parallel twin, which
    # is what reads as the plant levitating.
    #
    # Judgement, not calibration: measuring real plant-to-shadow displacement on
    # tiles proved unreliable (v2, which draws no synthetic shadow at all, scored
    # 0.50 because the estimator locks onto real shadows already in the soil).
    # The direction is supported though — the background crops are real
    # photographs and already carry the field's real shadow budget, so what is
    # added on top must be small and must hug the plant.
    shadow_offset: tuple[float, float] = (0.02, 0.08)   # as a fraction of plant size
    shadow_soft: tuple[float, float] = (0.020, 0.045)   # blur radius, same units
    shadow_tint: tuple[float, float, float] = (1.0, 0.95, 0.80)  # per-channel, R..B
    noise_match: bool = True
    noise_cap: float = 6.0        # max std of added noise, 0-255

    # ---- relighting -------------------------------------------------------
    # A diffusion model renders a plant under soft, near-frontal light. The soil
    # it is pasted onto is a photograph taken in hard sun. Measured on real
    # tiles, a real plant spans a factor of ~4 between its lit and self-shaded
    # parts; a rendered one spans ~2.5. That is a difference in CONTRAST, and no
    # global gain can add it — which is why `lum_match` alone never made these
    # look lit. relight builds a shading field from the plant's own silhouette
    # and the scene's sun vector, so the plant is shaded by the same light that
    # casts its shadow.
    relight: bool = True
    sun_elev: float = 0.55        # sun height; 0 = grazing, 1 = overhead
    ambient: float = 0.45         # sky fill, so the unlit side never goes black
    relight_strength: float = 0.35    # used when target_contrast is None
    # Targets from scripts/fit_appearance_prior.py on real tiles. None = skip
    # that correction rather than guess a value.
    target_contrast: float | None = None   # plant V p90/p10  (prior: fol_contrast)
    target_rel_v: float | None = None      # plant V / local soil V (prior: rel_v)
    target_rel_s: float | None = None      # plant S / local soil S (prior: rel_s)
    # Shaded foliage is lit by sky rather than sun, so it is cooler as well as
    # darker. Per-channel R..B, applied in proportion to how shaded a pixel is.
    shade_tint: tuple[float, float, float] = (0.96, 1.00, 1.08)

    lum_match: bool = True
    # Was hardcoded (0.9, 1.1). Measured on v2 and v7, the requested gain is
    # 1.2-1.4 in essentially every scene, so the old clip saturated every time
    # and lum_match degenerated into a constant +10% brightening that adapted to
    # nothing. Widened, and given a meaningful target via target_rel_v.
    lum_clip: tuple[float, float] = (0.70, 1.40)
    max_overlap: float = 0.15     # fraction of the smaller box; 1.0 disables
    place_tries: int = 40
    exposure_jitter: float = 0.06     # per-scene global gain, +-this
    wb_jitter: float = 0.03           # per-scene R/B gain, +-this
    label_alpha: float = 0.5      # matte threshold the YOLO box is drawn around
    # Which matte the LABEL is drawn around — a separate question from which one
    # is rendered, and they were wrongly tied together.
    #
    #   'source'   the delivered cutout's own alpha, thresholded at
    #              label_alpha_src. This is what the legacy blends used
    #              (alpha > 0) and it is the convention v2 was built with.
    #   'hardened' the post-_opaque matte at label_alpha. What v7 and v8 used.
    #
    # `_opaque` thresholds at 0.35 x peak, which throws away the soft outer halo
    # of an ExG matte. Rendering wants that — the halo is where translucency
    # came from. Labelling does not: measured against the real training labels,
    # median box side is 117.0 px real, 120.0 px on v2 ('source'), and 95.5 px
    # on v8 ('hardened'), i.e. 18% tight. A box convention that disagrees with
    # the real annotations costs more the more synthetic data is mixed in, which
    # matches the one thing that separates v8 from v2: v2 improves with ratio
    # (+0.0270 from r0.75 to r1.25) and v8 declines (-0.0059).
    label_mode: str = "source"
    label_alpha_src: float = 0.0


def paste_cutouts(
    background: np.ndarray,
    cutouts: list[Cutout],
    out_label: Path,
    seed: int = 0,
    blend: str = "alpha",
    photoreal: Photoreal | None = None,
) -> tuple[np.ndarray, int]:
    """Paste cutouts and emit perfect labels.

    Returns (composited_image, n_boxes). 'poisson' needs OpenCV seamlessClone;
    'photoreal' needs OpenCV too.
    """
    rng = random.Random(seed)
    canvas = background.copy()
    H, W = canvas.shape[:2]
    instances: list[tuple[str, np.ndarray]] = []

    if blend != "photoreal":
        return _paste_legacy(canvas, cutouts, out_label, rng, blend)

    pr = photoreal or Photoreal()

    # One sun for the whole scene. Inconsistent shadow directions inside a single
    # frame are the giveaway that a composite is machine-made, so this is sampled
    # once here and shared by every instance.
    theta = rng.uniform(0.0, 2.0 * math.pi)
    ox_f = pr.shadow_offset[0] + rng.random() * (pr.shadow_offset[1] - pr.shadow_offset[0])
    sun = (math.cos(theta) * ox_f, math.sin(theta) * ox_f)
    soft_f = pr.shadow_soft[0] + rng.random() * (pr.shadow_soft[1] - pr.shadow_soft[0])

    placed: list[tuple[int, int, int, int]] = []
    to_draw: list[tuple[Cutout, np.ndarray, int, int]] = []

    for cut in cutouts:
        ch, cw = cut.rgba.shape[:2]
        if ch >= H or cw >= W:
            continue
        a0 = cut.rgba[:, :, 3].astype(np.float32) / 255.0
        a = _opaque(a0, pr) if pr.fill_holes else _harden(a0, pr.alpha_lo, pr.alpha_hi)
        _AUDIT.check(a, pr)
        pos = _place(rng, W, H, cw, ch, placed, pr.max_overlap, pr.place_tries)
        if pos is None:
            continue
        x, y = pos
        placed.append((x, y, x + cw, y + ch))
        to_draw.append((cut, a, x, y))

    # Pass 1: every shadow, accumulated with max() so overlapping shadows do not
    # compound into black. Drawn before any plant so plants sit on top of them.
    if pr.shadow and to_draw:
        acc = np.zeros((H, W), dtype=np.float32)
        for _cut, a, x, y in to_draw:
            _accumulate_shadow(acc, a, x, y, sun, soft_f)
        canvas = _apply_shadow(canvas, acc, pr.shadow_strength, pr.shadow_tint)

    # Pass 2: the plants, shaded by the same sun that threw the shadows above.
    for cut, a, x, y in to_draw:
        canvas = _photoreal_blend(canvas, cut.rgba[:, :, :3], a, (x, y), pr, rng,
                                  sun=sun)
        ch, cw = a.shape
        if pr.label_mode == "source":
            src = cut.rgba[:, :, 3].astype(np.float32) / 255.0
            lab = src > pr.label_alpha_src
        else:
            lab = a >= pr.label_alpha
        full = np.zeros((H, W), dtype=np.uint8)
        full[y:y + ch, x:x + cw] = lab.astype(np.uint8)
        instances.append((cut.cls_name, full))

    canvas = _scene_illuminant(canvas, rng, pr)

    n = masks.masks_to_yolo(instances, (W, H), out_label)
    return canvas, n


# ---------------------------------------------------------------------------
# photoreal internals
# ---------------------------------------------------------------------------
def _harden(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Legacy ramp. Rescales [lo, hi] to [0, 1].

    SUPERSEDED by _opaque. A ramp cannot guarantee opacity: a matte pixel at
    0.45 comes out at 0.29 under lo=0.35/hi=0.70, so leaf interiors stay
    see-through and the thresholds have to be retuned for every matte. Kept only
    for the fill_holes=False ablation."""
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


class _OpacityAudit:
    """Count instances pasted with a see-through interior.

    Every previous transparency defect — the 4%-of-size feather, the `_harden`
    ramp, and the all-or-nothing hole-fill cap — shipped into a trained pool
    because nothing in the build looked at the alpha it was about to paste. The
    matte is supposed to decide WHERE a plant is, never how solid it is, so any
    pixel well inside the silhouette that is not fully opaque is a defect, and
    the build should say so while it runs.
    """

    def __init__(self):
        self.n = 0
        self.bad = 0
        self.worst = 1.0

    def check(self, a: np.ndarray, pr: "Photoreal") -> None:
        import cv2

        m = (a > 0.5).astype(np.uint8)
        if m.sum() < 64:
            return
        d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        deep = d > max(2.0, pr.feather_px + 1.0)
        if deep.sum() < 32:
            return
        self.n += 1
        lo = float(a[deep].min())
        self.worst = min(self.worst, lo)
        if lo < 0.99:
            self.bad += 1

    def report(self, name: str = "") -> None:
        if self.n == 0:
            return
        tag = f"[{name}] " if name else ""
        if self.bad:
            print(f"{tag}OPACITY WARNING: {self.bad}/{self.n} instances "
                  f"({self.bad / self.n * 100:.1f}%) pasted with a see-through "
                  f"interior; worst interior alpha {self.worst:.3f}. Plants are "
                  f"meant to be opaque — the matte decides where, not how solid.")
        else:
            print(f"{tag}opacity: {self.n} instances, all interiors fully opaque")


_AUDIT = _OpacityAudit()


def opacity_report(name: str = "") -> None:
    """Print the audit and reset it. Called once per pool by the generator."""
    _AUDIT.report(name)
    _AUDIT.__init__()


def _fill_holes(m: np.ndarray) -> np.ndarray:
    """Fill only true interior holes.

    KNOWN DEFECT, fixed here: the first version flood-filled the inverse mask
    from pixel (0, 0) and called every unreached zero a hole. That is only valid
    when the background of the crop is connected AND touches (0, 0). A plant
    whose leaves reach the edge of its own crop — which is the normal case,
    because `matte_exg` crops to the plant's bbox with 6 px of padding — cuts the
    background into several corner wedges, and every wedge that does not contain
    (0, 0) is declared a hole and filled. The result is a fully opaque rectangle:
    the black squares in the v8 preview. Measured on a four-lobed rosette that
    touches all four edges, the old code returned 0.81 of the crop opaque against
    a true plant coverage of 0.24, and it did so silently, because the *input*
    matte coverage that `--cutout-qc` inspects was a perfectly healthy 0.24.

    Padding by one transparent pixel makes the background provably connected and
    provably border-touching, so a single fill from the padded corner reaches all
    of it and only real enclosed holes survive.
    """
    import cv2

    h, w = m.shape
    inv = np.ones((h + 2, w + 2), np.uint8)
    inv[1:-1, 1:-1] = 1 - m
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(inv, ff, (0, 0), 2)
    holes = (inv[1:-1, 1:-1] == 1).astype(np.uint8)
    if holes.sum() == 0:
        return m

    # Cap each hole separately, not the operation as a whole.
    #
    # KNOWN DEFECT, fixed here. The first version of the cap was all-or-nothing:
    # if sealing the matte grew it past a fraction of the crop, the WHOLE fill
    # was discarded and every hole stayed open. A pale or specular leaf, whose
    # greenness matte leaves one large interior gap, blew the cap and was pasted
    # hollow — soil visible straight through the middle of the leaf. That is the
    # exact defect `_opaque` exists to prevent, reintroduced by its own guard.
    #
    # A hole is filled when it is enclosed and smaller than the plant around it.
    # A hole larger than the silhouette is not a hole, it is background that the
    # topology sealed off, and that one is left alone.
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes, 4)
    keep = np.zeros_like(holes)
    limit = max(1.0, float(m.sum())) * 1.0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] <= limit:
            keep |= (lab == i).astype(np.uint8)
    return np.where(keep.astype(bool), 1, m).astype(np.uint8)


def _extend_edge_colour(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Push plant colour outward under the feather rim.

    A cutout carries no useful RGB where the matte said "not plant". Depending on
    which matte produced it that region is either soil or, for `matte_rembg` and
    anything else that writes a transparent-black background, literally
    (0, 0, 0). The rim alpha from the distance transform is fractional, so those
    pixels get mixed into the frame at up to ~50%, drawing a dark outline around
    every plant. Replacing them with the nearest plant colour before blending
    makes the rim a plant-to-soil transition instead of a plant-to-black one, and
    is a no-op wherever alpha is already 1.
    """
    import cv2

    if m.sum() == 0 or m.all():
        return rgb
    holes = ((1 - m) * 255).astype(np.uint8)
    return cv2.inpaint(np.clip(rgb, 0, 255).astype(np.uint8), holes, 3,
                       cv2.INPAINT_TELEA).astype(np.float32)


def _opaque(a: np.ndarray, pr: "Photoreal") -> np.ndarray:
    """Binarise the matte, seal it, and put the ONLY soft alpha on the rim.

    Construction, not calibration:
      1. threshold at `matte_thresh` x the cutout's own peak alpha, so a matte
         that never reaches 1.0 is handled the same as one that does;
      2. close, then fill interior holes — the gaps a greenness matte leaves
         where a leaf is specular or shaded (see `_fill_holes`);
      3. drop blobs under `min_blob_px` (soil speckle the matte grabbed);
      4. alpha = clip(distance_inside / feather_px), which is exactly 1.0
         everywhere except a feather_px band inside the silhouette.

    Every pixel more than feather_px inside the outline is therefore fully
    opaque by construction, whatever the matte looked like.

    Step 2 is also bounded: if sealing the mask grows it past `fill_max_growth`
    of the crop area, the fill is discarded and the closed mask is used instead.
    Hole-filling should recover leaf interiors, which are a small fraction of a
    silhouette; anything that doubles the area is a topology accident, not a
    hole, and an unsealed leaf is a far cheaper defect than an opaque rectangle.
    """
    import cv2

    peak = float(a.max())
    if peak <= 0:
        return a
    m = (a >= pr.matte_thresh * peak).astype(np.uint8)
    if m.sum() == 0:
        return a

    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    if pr.fill_holes:
        m = _fill_holes(m)

    if pr.min_blob_px > 0:
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            keep = np.flatnonzero(st[1:, cv2.CC_STAT_AREA] >= pr.min_blob_px) + 1
            m = np.isin(lab, keep).astype(np.uint8)
            if m.sum() == 0:
                m = (lab > 0).astype(np.uint8)

    f = max(float(pr.feather_px), 1e-3)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return np.clip(d / f, 0.0, 1.0).astype(np.float32)


def _shading_field(a: np.ndarray, sun: tuple[float, float],
                   pr: "Photoreal") -> np.ndarray:
    """A Lambertian shading map for the plant, from its own silhouette.

    There is no geometry here to light, so the silhouette stands in for it: a
    heavily blurred matte is a height field that is high in the middle of a leaf
    mass and falls off at the edges, which is roughly true of a rosette or a
    tuft. Its gradient gives a surface normal, and `sun` — the same vector the
    scene already uses to throw shadows — gives the light direction. The result
    is that the bright side of every plant in a frame agrees with the direction
    every shadow points, which is the part the eye checks first.

    Normalised to mean 1.0 over the plant, so this redistributes light without
    changing exposure. Overall brightness stays the job of the ratio gain.
    """
    import cv2

    m = (a > 0.5).astype(np.float32)
    if m.sum() < 16:
        return np.ones_like(a)

    L = float(max(a.shape))
    h = cv2.GaussianBlur(m, (0, 0), max(1.5, 0.10 * L))
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)

    g = np.sqrt(gx * gx + gy * gy)
    inside = m > 0.5
    s = float(np.percentile(g[inside], 90)) if inside.sum() else 0.0
    if s <= 1e-6:
        return np.ones_like(a)
    gx, gy = gx / s, gy / s

    nz = np.ones_like(gx)
    nrm = np.sqrt(gx * gx + gy * gy + 1.0)
    nx, ny, nz = -gx / nrm, -gy / nrm, nz / nrm

    # `sun` is the shadow OFFSET vector, so its length encodes how far shadows
    # are thrown (0.08-0.22 of plant size), not how strong the light is. Only its
    # direction is wanted here; using it unnormalised put the light almost
    # straight overhead and produced a shading field that was nearly flat, which
    # is the defect this whole function exists to remove.
    sx, sy = -sun[0], -sun[1]
    sl = math.hypot(sx, sy)
    if sl < 1e-6:
        sx, sy = 1.0, 0.0
    else:
        sx, sy = sx / sl, sy / sl
    hz = max(0.05, float(pr.sun_elev))
    ln = math.sqrt(sx * sx + sy * sy + hz * hz)
    lx, ly, lz = sx / ln, sy / ln, hz / ln

    lam = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    sh = float(pr.ambient) + (1.0 - float(pr.ambient)) * lam

    mean = float(sh[inside].mean()) if inside.sum() else 1.0
    return (sh / max(mean, 1e-6)).astype(np.float32)


def _apply_relight(rgb: np.ndarray, a: np.ndarray, sun, pr: "Photoreal") -> np.ndarray:
    """Shade the plant with `sun`, scaled to hit the measured self-shading."""
    import cv2

    inside = a > 0.5
    if inside.sum() < 64:
        return rgb

    base = _shading_field(a, sun, pr)

    def _contrast(strength: float) -> float:
        f = 1.0 + strength * (base - 1.0)
        v = (cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8),
                          cv2.COLOR_RGB2HSV)[..., 2].astype(np.float32) * f)[inside]
        lo = float(np.percentile(v, 10))
        return float(np.percentile(v, 90)) / max(lo, 1.0)

    if pr.target_contrast is None:
        k = float(pr.relight_strength)
    else:
        # Contrast rises monotonically with strength, so bisect. Capped at 2.0:
        # if a render is so flat that it cannot reach the target, exaggerating
        # the shading further would look worse than falling short.
        lo, hi = 0.0, 2.0
        if _contrast(hi) <= pr.target_contrast:
            k = hi
        else:
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                if _contrast(mid) < pr.target_contrast:
                    lo = mid
                else:
                    hi = mid
            k = 0.5 * (lo + hi)

    f = 1.0 + k * (base - 1.0)
    out = rgb * f[:, :, None]

    shade = np.clip(1.0 - f, 0.0, 1.0)[:, :, None]
    tint = np.asarray(pr.shade_tint, dtype=np.float32)[None, None, :]
    out = out * (1.0 + shade * (tint - 1.0))
    return np.clip(out, 0, 255)


def _ratio_gain(rgb, region, pm, bm, pr: "Photoreal"):
    """Set the plant's brightness relative to the soil it stands on.

    The old code matched the plant's mean luminance to the soil's, i.e. aimed at
    a ratio of 1.0. Foliage is darker than dry pale soil under the same sun, so
    that target is wrong in principle, and in practice the +-10% clip meant the
    requested gain (measured at 1.2-1.4) saturated in every scene and the same
    fixed brightening was applied regardless of the background. With
    target_rel_v set from real tiles this aims at the measured ratio instead.
    """
    import cv2

    if pm.sum() <= 20 or bm.sum() <= 20:
        return rgb

    hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8),
                       cv2.COLOR_RGB2HSV).astype(np.float32)
    pv = hsv[..., 2][pm]
    sv = cv2.cvtColor(np.clip(region, 0, 255).astype(np.uint8),
                      cv2.COLOR_RGB2HSV)[..., 2][bm].astype(np.float32)
    if pv.size == 0 or sv.size == 0 or pv.mean() < 1.0:
        return rgb
    soil = float(np.median(sv))

    if pr.target_rel_v is not None:
        want = (pr.target_rel_v * soil) / float(pv.mean())
        clip = pr.lum_clip
    else:
        # No measured target, so the only thing available is the old
        # match-the-means behaviour, whose target of 1.0 is wrong in the first
        # place. Keep it on the narrow legacy clip: widening a gain that aims at
        # the wrong number just gets there faster. The wide clip is earned by
        # having measured rel_v, not by turning relighting on.
        def _lum(im, m):
            return float((0.299 * im[..., 0] + 0.587 * im[..., 1]
                          + 0.114 * im[..., 2])[m].mean())
        lp = _lum(rgb, pm)
        if lp <= 1.0:
            return rgb
        want = _lum(region, bm) / lp
        clip = (0.9, 1.1)

    gain = float(np.clip(want, clip[0], clip[1]))
    hsv[..., 2] = np.clip(hsv[..., 2] * gain, 0, 255)

    if pr.target_rel_s is not None:
        ss = cv2.cvtColor(np.clip(region, 0, 255).astype(np.uint8),
                          cv2.COLOR_RGB2HSV)[..., 1][bm].astype(np.float32)
        ps = float(hsv[..., 1][pm].mean())
        ms = float(np.median(ss))
        if ps > 1.0 and ms > 1.0:
            g = float(np.clip((pr.target_rel_s * ms) / ps, 0.7, 1.4))
            hsv[..., 1] = np.clip(hsv[..., 1] * g, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)


def _box_overlap(a, b) -> float:
    """Intersection as a fraction of the SMALLER box. More intuitive than IoU
    when one plant is much larger than the other."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1, min(sa, sb))


def _place(rng, W, H, cw, ch, placed, max_overlap, tries):
    if max_overlap >= 1.0:
        return rng.randint(0, W - cw), rng.randint(0, H - ch)
    for _ in range(tries):
        x, y = rng.randint(0, W - cw), rng.randint(0, H - ch)
        box = (x, y, x + cw, y + ch)
        if all(_box_overlap(box, p) <= max_overlap for p in placed):
            return x, y
    return None


def _blit_max(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    H, W = dst.shape[:2]
    sh, sw = src.shape[:2]
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(W, x + sw), min(H, y + sh)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    np.maximum(dst[dy0:dy1, dx0:dx1], src[dy0 - y:dy1 - y, dx0 - x:dx1 - x],
               out=dst[dy0:dy1, dx0:dx1])


def _accumulate_shadow(acc, a, x, y, sun, soft_f) -> None:
    """Project one plant's matte onto the ground and add it to the scene mask.

    Bigger plants stand taller, so both the offset and the penumbra scale with
    the plant's size — that is what makes a field of shadows read as one
    lighting condition rather than a rubber-stamped drop shadow."""
    import cv2

    ch, cw = a.shape
    L = float(max(ch, cw))
    r = max(1, int(round(soft_f * L)))
    pad = 2 * r
    s = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0.0)
    s = cv2.GaussianBlur(s, (2 * r + 1, 2 * r + 1), 0)
    ox, oy = int(round(sun[0] * L)), int(round(sun[1] * L))
    _blit_max(acc, s, x + ox - pad, y + oy - pad)


def _apply_shadow(canvas, acc, strength, tint) -> np.ndarray:
    """Multiply the canvas down. Multiplying preserves soil texture inside the
    shadow; painting a grey overlay would flatten it, which is the other way
    composites give themselves away."""
    m = np.clip(acc, 0.0, 1.0)[:, :, None] * float(strength)
    t = np.asarray(tint, dtype=np.float32)[None, None, :]
    out = canvas.astype(np.float32) * (1.0 - m * t)
    return np.clip(out, 0, 255).astype(canvas.dtype)


def _photoreal_blend(canvas, rgb_u8, a, top_left, pr: Photoreal, rng,
                     sun: tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
    import cv2

    x, y = top_left
    ch, cw = a.shape
    region = canvas[y:y + ch, x:x + cw, :].astype(np.float32)
    rgb = rgb_u8.astype(np.float32)

    pm = a > 0.5
    bm = a < 0.1

    if pr.edge_extend:
        rgb = _extend_edge_colour(rgb, pm.astype(np.uint8))

    # Order matters: shape the light across the plant first, then set its overall
    # level against the soil. Reversing them lets the shading undo the match.
    if pr.relight:
        rgb = _apply_relight(rgb, a, sun, pr)

    if pr.lum_match:
        rgb = _ratio_gain(rgb, region, pm, bm, pr)

    if pr.noise_match and pm.sum() > 50 and bm.sum() > 50:
        rgb = _match_noise(rgb, region, pm, bm, rng, pr.noise_cap)

    # Fixed-radius feather: enough to kill the staircase on the silhouette,
    # far too small to hollow out a grass blade.
    k = max(1, int(round(pr.feather_px)) * 2 + 1)
    a_soft = cv2.GaussianBlur(a, (k, k), pr.feather_px / 2.0)

    a3 = a_soft[:, :, None]
    out = a3 * rgb + (1.0 - a3) * region
    canvas[y:y + ch, x:x + cw, :] = np.clip(out, 0, 255).astype(canvas.dtype)
    return canvas


def _match_noise(rgb, region, pm, bm, rng, cap) -> np.ndarray:
    """Give the plant the same grain as the soil it sits on.

    A diffusion render is smoother than a photograph at the pixel level. Trained
    beside real tiles, that difference is a free real-vs-synthetic cue.
    """
    import cv2

    def _hf_std(im, m):
        lp = im - cv2.GaussianBlur(im, (0, 0), 1.0)
        return float(lp[m].std())

    sb, sp = _hf_std(region, bm), _hf_std(rgb, pm)
    if sb <= sp:
        return rgb
    sigma = min(float(cap), math.sqrt(max(sb * sb - sp * sp, 0.0)))
    if sigma < 0.5:
        return rgb
    g = np.random.default_rng(rng.randrange(1 << 31))
    return np.clip(rgb + g.normal(0.0, sigma, rgb.shape).astype(np.float32), 0, 255)


def _scene_illuminant(canvas, rng, pr: Photoreal) -> np.ndarray:
    """One exposure and one white balance for the whole frame.

    Applied last, so plants and soil are lit by the same thing. Sub-6% gains —
    the point is that plant and soil move together, not that the frame changes
    much."""
    if pr.exposure_jitter <= 0 and pr.wb_jitter <= 0:
        return canvas
    e = 1.0 + rng.uniform(-pr.exposure_jitter, pr.exposure_jitter)
    wr = 1.0 + rng.uniform(-pr.wb_jitter, pr.wb_jitter)
    wb = 1.0 - (wr - 1.0)
    g = np.asarray([e * wr, e, e * wb], dtype=np.float32)[None, None, :]
    return np.clip(canvas.astype(np.float32) * g, 0, 255).astype(canvas.dtype)


# ---------------------------------------------------------------------------
# legacy paths — unchanged, so v2..v6 stay byte-reproducible
# ---------------------------------------------------------------------------
def _paste_legacy(canvas, cutouts, out_label, rng, blend):
    H, W = canvas.shape[:2]
    instances: list[tuple[str, np.ndarray]] = []
    for cut in cutouts:
        ch, cw = cut.rgba.shape[:2]
        if ch >= H or cw >= W:
            continue
        x = rng.randint(0, W - cw)
        y = rng.randint(0, H - ch)
        alpha = cut.rgba[:, :, 3:4].astype(np.float32) / 255.0
        rgb = cut.rgba[:, :, :3].astype(np.float32)

        if blend == "poisson":
            canvas = _poisson_blend(canvas, cut.rgba, (x, y))
        elif blend == "feather":
            canvas = _feather_blend(canvas, cut.rgba, (x, y))
        else:
            region = canvas[y:y + ch, x:x + cw, :].astype(np.float32)
            canvas[y:y + ch, x:x + cw, :] = (
                alpha * rgb + (1 - alpha) * region).astype(canvas.dtype)

        full = np.zeros((H, W), dtype=np.uint8)
        full[y:y + ch, x:x + cw] = (cut.rgba[:, :, 3] > 0).astype(np.uint8)
        instances.append((cut.cls_name, full))

    n = masks.masks_to_yolo(instances, (W, H), out_label)
    return canvas, n


def _feather_blend(
    canvas: np.ndarray,
    rgba: np.ndarray,
    top_left: tuple[int, int],
    feather_frac: float = 0.04,
    lum_match: bool = True,
) -> np.ndarray:
    """Colour-preserving composite: alpha blend with a feathered edge, plus an
    optional brightness-only (HSV value) match to the surrounding soil.

    Unlike Poisson/seamlessClone this never recolours the plant. seamlessClone
    only preserves gradients and pins the patch boundary to the destination, so a
    small green weed on brown soil gets its absolute colour dragged toward the
    soil (goes soil-coloured), worst for small plants. Here the plant keeps its
    own RGB; we only (a) soften the cut edge so there is no paste seam and (b)
    nudge exposure via a tightly clipped value gain, so hue and saturation are
    untouched. The YOLO label is unchanged (it uses the hard alpha upstream).

    KNOWN DEFECT, fixed in blend='photoreal': feather_frac is a fraction of the
    cutout's size, so large cutouts get a large blur kernel, and any structure
    thinner than that kernel loses most of its alpha and turns translucent.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenCV required for feather blending.") from exc
    x, y = top_left
    ch, cw = rgba.shape[:2]
    region = canvas[y : y + ch, x : x + cw, :].astype(np.float32)
    rgb = rgba[:, :, :3].astype(np.float32)
    a = rgba[:, :, 3].astype(np.float32) / 255.0

    r = max(1, int(round(min(ch, cw) * feather_frac)))
    a_soft = cv2.GaussianBlur(a, (2 * r + 1, 2 * r + 1), 0)

    if lum_match:
        pm = a > 0.5            # plant pixels
        bm = a < 0.1            # surrounding soil inside the paste box
        if pm.sum() > 20 and bm.sum() > 20:
            def _lum(im, m):
                return float((0.299 * im[..., 0] + 0.587 * im[..., 1]
                              + 0.114 * im[..., 2])[m].mean())
            lp, lb = _lum(rgb, pm), _lum(region, bm)
            if lp > 1.0:
                gain = float(np.clip(lb / lp, 0.9, 1.1))   # exposure only, clipped
                hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8),
                                   cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 2] = np.clip(hsv[..., 2] * gain, 0, 255)
                rgb = cv2.cvtColor(hsv.astype(np.uint8),
                                   cv2.COLOR_HSV2RGB).astype(np.float32)

    a3 = a_soft[:, :, None]
    out = a3 * rgb + (1.0 - a3) * region
    canvas[y : y + ch, x : x + cw, :] = np.clip(out, 0, 255).astype(canvas.dtype)
    return canvas


def _poisson_blend(canvas: np.ndarray, rgba: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenCV required for Poisson blending.") from exc
    x, y = top_left
    ch, cw = rgba.shape[:2]
    mask = (rgba[:, :, 3] > 0).astype(np.uint8) * 255
    center = (x + cw // 2, y + ch // 2)
    return cv2.seamlessClone(rgba[:, :, :3], canvas, mask, center, cv2.NORMAL_CLONE)
