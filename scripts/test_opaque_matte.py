"""Regression tests for the photoreal matte (src/annotate/composite.py).

Each test is a defect that shipped in a pool and was found by eye, turned into a
number so it cannot ship again.

    python scripts/test_opaque_matte.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotate.composite import (  # noqa: E402
    Photoreal, _opaque, _photoreal_blend,
)

SOIL = np.array([216, 200, 165], np.float32)
GREEN = np.array([92, 128, 70], np.float32)
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def rosette(n: int = 120, peak: float = 0.62) -> np.ndarray:
    """Four lobes from the centre to all four edges, like a real POROL crop
    bbox-cropped with 6 px of padding."""
    m = np.zeros((n, n), np.float32)
    c = n // 2
    for ang in (0, 90, 180, 270):
        t = np.deg2rad(ang)
        for r in range(c):
            x, y = int(c + r * np.cos(t)), int(c + r * np.sin(t))
            cv2.circle(m, (x, y), int(round((1 - r / c) * 14)) + 1, 1.0, -1)
    return cv2.GaussianBlur(m, (5, 5), 2) / 1.0 * peak


def donut(n: int = 120, peak: float = 0.62) -> np.ndarray:
    """A genuine enclosed hole, well away from the border. Hole-filling must
    still close this one."""
    m = np.zeros((n, n), np.float32)
    cv2.circle(m, (n // 2, n // 2), 40, 1.0, -1)
    cv2.circle(m, (n // 2, n // 2), 16, 0.0, -1)
    return m * peak


def blades(n: int = 200, width: int = 7, peak: float = 0.62) -> np.ndarray:
    m = np.zeros((n, n), np.float32)
    cv2.line(m, (15, 180), (185, 20), 1.0, width)
    cv2.line(m, (15, 100), (185, 80), 1.0, width)
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    return m / m.max() * peak


def as_rgba(matte: np.ndarray, bg: np.ndarray) -> np.ndarray:
    rgb = np.empty(matte.shape + (3,), np.float32)
    rgb[:] = bg
    rgb[matte > 0.3 * matte.max()] = GREEN
    return np.dstack([rgb, matte * 255]).astype(np.uint8)


def paste(rgba: np.ndarray, pr: Photoreal):
    a = _opaque(rgba[:, :, 3].astype(np.float32) / 255.0, pr)
    h, w = a.shape
    canvas = np.empty((h + 40, w + 40, 3), np.uint8)
    canvas[:] = SOIL.astype(np.uint8)
    rng = __import__("random").Random(0)
    out = _photoreal_blend(canvas, rgba[:, :, :3], a, (20, 20), pr, rng)
    return a, out


def main() -> None:
    # Relighting is off for the matte tests below: they check what the matte
    # does to the plant's colour, and relighting changes that colour on purpose.
    # Section 7 tests relighting on its own.
    pr = Photoreal(relight=False)

    print("\n1. Plant touching the crop border must not seal into a rectangle")
    for name, matte in (("rosette (4 edges)", rosette()), ("blades", blades())):
        a = _opaque(matte, pr)
        true_cov = float((matte >= pr.matte_thresh * matte.max()).mean())
        got_cov = float((a > 0.9).mean())
        check(name, got_cov <= true_cov + pr.fill_max_growth + 0.02,
              f"opaque {got_cov:.2f} of crop vs matte {true_cov:.2f}")

    print("\n2. A real enclosed hole is still filled")
    m = donut()
    a = _opaque(m, pr)
    hole = np.zeros_like(m, np.uint8)
    cv2.circle(hole, (60, 60), 12, 1, -1)
    check("donut interior", float(a[hole == 1].min()) > 0.9,
          f"min alpha inside the hole {float(a[hole == 1].min()):.2f}")

    print("\n3. Plant interiors are fully opaque whatever the matte peaked at")
    for peak in (0.45, 0.62, 1.00):
        a = _opaque(rosette(peak=peak), pr)
        core = a > 0
        d = cv2.distanceTransform((a > 0).astype(np.uint8), cv2.DIST_L2, 3)
        deep = d > pr.feather_px + 1
        check(f"peak={peak:.2f}", bool(deep.sum()) and float(a[deep].min()) > 0.99,
              f"min alpha more than {pr.feather_px} px inside "
              f"{float(a[deep].min()) if deep.sum() else float('nan'):.3f}")

    print("\n4. A 7 px blade keeps its colour when pasted")
    print("   Photometry off, so this measures only what the matte lets through.")
    pr_pure = Photoreal(relight=False, lum_match=False, noise_match=False)
    rgba = as_rgba(blades(), bg=np.zeros(3, np.float32))
    a, out = paste(rgba, pr_pure)
    core = a[20:-20, 20:-20] if False else None
    aa = np.zeros(out.shape[:2], np.float32)
    aa[20:20 + a.shape[0], 20:20 + a.shape[1]] = a
    solid = aa > 0.99
    got = out[solid].astype(np.float32).mean(0)
    retained = 1.0 - np.abs(got - GREEN).sum() / np.abs(SOIL - GREEN).sum()
    check("blade colour", retained > 0.90,
          f"{retained * 100:.1f}% of pure plant colour retained "
          f"(RGB {got.round(0)} vs {GREEN})")

    print("\n5. What lies under the matte must not reach the frame")
    print("   Same silhouette, same plant colour, different discarded background.")
    print("   A cutout is defined by its matte, so the two must composite alike.")
    outs = {}
    for label, bg in (("rembg/black", np.zeros(3, np.float32)),
                      ("ExG/soil", SOIL)):
        _, outs[label] = paste(as_rgba(rosette(), bg=bg), pr_pure)
    d = np.abs(outs["rembg/black"].astype(np.float32)
               - outs["ExG/soil"].astype(np.float32))
    check("black vs soil backing", float(d.max()) <= 8.0,
          f"worst pixel differs by {float(d.max()):.0f}/255, "
          f"mean {float(d.mean()):.2f}")

    pr_off = Photoreal(edge_extend=False, relight=False, lum_match=False,
                       noise_match=False)
    outs_off = {}
    for label, bg in (("rembg/black", np.zeros(3, np.float32)),
                      ("ExG/soil", SOIL)):
        _, outs_off[label] = paste(as_rgba(rosette(), bg=bg), pr_off)
    d_off = np.abs(outs_off["rembg/black"].astype(np.float32)
                   - outs_off["ExG/soil"].astype(np.float32))
    print(f"  [INFO] with edge_extend=False the same pair differs by "
          f"{float(d_off.max()):.0f}/255 at worst, mean {float(d_off.mean()):.2f}")

    print("\n7. Relighting")
    print("   7a. The lit side of the plant follows the scene's sun vector.")
    from src.annotate.composite import _shading_field, _apply_relight, _ratio_gain
    m = rosette()
    hard = _opaque(m, pr)
    ok_dir = True
    for sun, want in (((1.0, 0.0), (-1, 0)), ((0.0, 1.0), (0, -1)),
                      ((-1.0, 0.0), (1, 0)), ((0.0, -1.0), (0, 1))):
        f = _shading_field(hard, sun, pr)
        ys, xs = np.where(hard > 0.5)
        w = f[ys, xs]
        dx = ((xs - xs.mean()) * w).sum() / w.sum()
        dy = ((ys - ys.mean()) * w).sum() / w.sum()
        if np.sign(dx or want[0]) != np.sign(want[0] or dx) and want[0]:
            ok_dir = False
        if want[1] and np.sign(dy) != np.sign(want[1]):
            ok_dir = False
    check("bright side opposes the shadow", ok_dir,
          "brightest region sits on the side the light comes from, "
          "for all four cardinal sun vectors")

    f = _shading_field(hard, (1.0, 0.0), pr)
    check("shading is exposure-neutral", abs(float(f[hard > 0.5].mean()) - 1.0) < 0.01,
          f"mean shading over the plant {float(f[hard > 0.5].mean()):.4f} "
          f"(must be 1.0 so it redistributes light, not adds it)")

    print("   7b. Self-shading is driven to the measured target.")
    rgba = as_rgba(rosette(), bg=SOIL)
    flat = rgba[:, :, :3].astype(np.float32)
    a = _opaque(rgba[:, :, 3].astype(np.float32) / 255.0, pr)
    inside = a > 0.5

    def contrast(img):
        v = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                         cv2.COLOR_RGB2HSV)[..., 2].astype(np.float32)[inside]
        return float(np.percentile(v, 90)) / max(float(np.percentile(v, 10)), 1.0)

    for target in (2.5, 4.0):
        prl = Photoreal(target_contrast=target)
        got = contrast(_apply_relight(flat.copy(), a, (0.6, 0.8), prl))
        check(f"target {target:.1f}", abs(got - target) / target < 0.12,
              f"reached {got:.2f} from a flat render at {contrast(flat):.2f}")

    print("   7c. The plant/soil brightness ratio hits the measured target.")
    soil = np.empty(flat.shape, np.float32)
    soil[:] = SOIL
    bm = a < 0.1
    for target in (0.60, 0.75):
        prl = Photoreal(target_rel_v=target)
        out = _ratio_gain(flat.copy(), soil, inside, bm, prl)
        hv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8),
                          cv2.COLOR_RGB2HSV)[..., 2].astype(np.float32)
        sv = cv2.cvtColor(SOIL.astype(np.uint8).reshape(1, 1, 3),
                          cv2.COLOR_RGB2HSV)[0, 0, 2]
        got = float(hv[inside].mean()) / float(sv)
        check(f"ratio {target:.2f}", abs(got - target) < 0.05,
              f"reached {got:.3f}")

    print("   7d. The old clip really did saturate.")
    prl = Photoreal()
    v_plant = cv2.cvtColor(np.clip(flat, 0, 255).astype(np.uint8),
                           cv2.COLOR_RGB2HSV)[..., 2].astype(np.float32)[inside].mean()
    v_soil = cv2.cvtColor(SOIL.astype(np.uint8).reshape(1, 1, 3),
                          cv2.COLOR_RGB2HSV)[0, 0, 2]
    print(f"  [INFO] mean-matching would request gain {v_soil / v_plant:.2f}; "
          f"the old clip allowed 1.10, the new one {prl.lum_clip[1]:.2f}")

    print("\n8. Mosaic quadrant matching must not manufacture black")
    from src.generators.composite_gen import _match_quadrants
    g = np.random.default_rng(1)

    def soil(mean, sd):
        a = np.clip(g.normal(mean, sd, (300, 300, 3)), 0, 255)
        a[:70, :70] = np.clip(g.normal(mean * 0.42, 10, (70, 70, 3)), 0, 255)
        return a

    # A high-contrast crop plus three flat ones is the case that used to blow up:
    # matching sd to the high-contrast anchor multiplied the flat crops' contrast
    # until their shadowed corners clipped to zero.
    P = [soil(180, 46), soil(175, 12), soil(190, 10), soil(170, 14)]
    out = _match_quadrants([p.copy() for p in P])
    d_in = float(np.mean([(p.astype(np.int32).sum(2) < 60).mean() for p in P]))
    d_out = float(np.mean([(o.astype(np.int32).sum(2) < 60).mean() for o in out]))
    check("no black created", d_out <= d_in + 0.0005,
          f"near-black {d_in * 100:.3f}% in -> {d_out * 100:.3f}% out")

    mus = [float(o.mean()) for o in out]
    check("exposure step still removed", max(mus) - min(mus) < 1.0,
          f"quadrant means {[round(m, 1) for m in mus]} "
          f"(spread {max(mus) - min(mus):.2f}, was "
          f"{max(float(p.mean()) for p in P) - min(float(p.mean()) for p in P):.1f})")

    sd_in = [float(p.std()) for p in P]
    sd_out = [float(o.std()) for o in out]
    ratio = max(o / i for o, i in zip(sd_out, sd_in))
    check("contrast is not rescaled", ratio < 1.15,
          f"largest spread change {ratio:.2f}x (the old code reached ~7x)")

    def old_match(patches):
        t_mu = t_sd = None
        res = []
        for a in patches:
            mu, sd = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0) + 1e-6
            if t_mu is None:
                t_mu, t_sd = mu, sd
                res.append(a)
            else:
                res.append(np.clip((a - mu) / sd * t_sd + t_mu, 0, 255))
        return res

    o = old_match([p.copy() for p in P])
    print(f"  [INFO] the old mean+sd match on the same input: "
          f"{np.mean([(x.astype(np.int32).sum(2) < 60).mean() for x in o]) * 100:.3f}% near-black")

    print("\n6. The old code really did fail test 1")
    m = rosette()
    b = (m >= pr.matte_thresh * m.max()).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    inv = (1 - b).copy()
    cv2.floodFill(inv, np.zeros((b.shape[0] + 2, b.shape[1] + 2), np.uint8), (0, 0), 2)
    old = np.where(inv == 1, 1, b)
    print(f"  [INFO] corner-seeded fill: {float(old.mean()):.2f} of the crop "
          f"opaque vs a true plant coverage of {float(b.mean()):.2f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all matte regression tests passed")


if __name__ == "__main__":
    main()
