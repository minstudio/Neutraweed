"""SD3.5 cut-and-composite generator (CLAUDE.md 3/4 Stage C).

The salvageable SD route. Instead of asking SD3.5 to render a fully labelled
multi-instance field tile (unusable: instances drift off the boxes, wrong
species, loose boxes), we copy the CSIC move and generate ONE isolated weed per
image on a plain background — trivially labelable — matte it to an RGBA cutout,
then feed those cutouts into the existing composite paste pipeline. Labels stay
perfect by construction (the box is the pasted cutout), and the only thing that
changed vs the `composite` arm is that the cutouts are SD-generated instead of
cut from real images. That makes a clean generator comparison:
    composite (real cutouts)  vs  sd35cut (SD cutouts)  vs  flux cutouts ...

Two phases:
  cutouts()  : GPU. SD3.5 txt2img per class -> matte -> RGBA cutouts in the pool.
  generate() : CPU. reuse composite_gen.CompositeGenerator on this pool.

Backgrounds are reused from an existing composite pool (real soil) so this arm
differs from `composite` ONLY in cutout provenance.
"""

from __future__ import annotations

import math
import random
import shutil
from pathlib import Path

import numpy as np

from ..common import classes, paths
from .composite_gen import _pool_dirs

# Caption used for BOTH the LoRA training crops (export_weed_crops.py) and the
# generation prompt, so the domain LoRA actually fires at render time — the
# fine-tune associates this exact species phrasing with the real appearance,
# fixing base SD3.5's habit of drawing generic flowers/leaves for these rare
# EPPO weeds. Keep the two in sync via caption_for().
def caption_for(cls: str, stage: str | None = None) -> str:
    disp = classes.CLASS_DISPLAY.get(cls, cls)
    return f"a top-down close-up photo of {disp}, {stage or 'a weed seedling'} on bare soil"


# Morphology group per species -> which growth-stage phrases are plausible.
_MORPH = {
    "SOLNI": "broadleaf", "POROL": "succulent",
    "SETVE": "grass", "CYPRO": "grass", "ECHCG": "grass",
}
# Early-postemergence stages ONLY (the detection target window, < ~10 days).
# Staying inside the range the species LoRA actually saw keeps renders
# species-correct; prompting mature/flowering plants would be off-task and
# unreliable. This adds phenological COVERAGE, the lever that moved the numbers.
_STAGES = {
    "broadleaf": [
        "at the cotyledon stage with a single pair of seed leaves",
        "a young seedling with its first pair of true leaves",
        "at the two to four true-leaf stage",
        "an early vegetative seedling with several small leaves",
    ],
    "succulent": [
        "at the cotyledon stage with two small fleshy seed leaves",
        "a young seedling with a few small fleshy leaves",
        "an early vegetative plant forming a low spreading rosette of fleshy leaves",
        "a spreading mat of thick fleshy leaves on branching reddish stems",
    ],
    "grass": [
        "a grass seedling at the one-leaf stage",
        "a young grass seedling with two to three narrow leaves",
        "at the early tillering stage with several thin blades",
    ],
}


def stages_for(cls: str) -> list[str]:
    return _STAGES[_MORPH.get(cls, "broadleaf")]


# Generation adds only isolation/framing so the plant is the sole green object
# and mattes cleanly; the species-bearing core is caption_for(), matching training.
def _prompt(cls: str, stage: str | None = None) -> str:
    return caption_for(cls, stage) + ", single isolated plant, centered, plain uniform background, sharp focus"


_NEG_PROMPT = (
    "multiple plants, several weeds, dense vegetation, grass field, crop rows, "
    "tomato plant, flower, blossom, single leaf, close-up of one leaf, blurry, "
    "cropped, out of frame, watermark, text, hand, pot, flowerpot, indoor"
)


# ---------------------------------------------------------------------------
# Matte: isolate the green plant from the plain background -> RGBA
# ---------------------------------------------------------------------------
def matte_exg(rgb: np.ndarray, thresh: float = 0.06, min_area: int = 400,
              feather: int = 2, pad: int = 6) -> np.ndarray | None:
    """ExG greenness matte for a single-plant-on-soil image. Keeps the largest
    green connected component, fills holes, feathers the edge, crops to bbox.
    Returns an RGBA array, or None if no plausible plant was found."""
    from PIL import Image, ImageFilter
    from scipy import ndimage

    a = rgb.astype(np.float32)
    s = a.sum(axis=2) + 1e-6
    exg = (2 * a[..., 1] - a[..., 0] - a[..., 2]) / s
    mask = exg > thresh
    if mask.sum() < min_area:
        return None

    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    mask = ndimage.binary_fill_holes(lab == keep)
    if mask.sum() < min_area:
        return None

    ys, xs = np.where(mask)
    y1, y2 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
    x1, x2 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
    sub_rgb = rgb[y1:y2, x1:x2]
    sub_a = (mask[y1:y2, x1:x2].astype(np.uint8) * 255)
    if feather > 0:
        sub_a = np.asarray(
            Image.fromarray(sub_a).filter(ImageFilter.GaussianBlur(feather)))
    return np.dstack([sub_rgb, sub_a]).astype(np.uint8)


def matte_rembg(rgb: np.ndarray, min_area: int = 400, pad: int = 6):
    """Optional foreground matte via rembg/BiRefNet if installed. Falls back to
    None so the caller can use matte_exg."""
    try:
        from rembg import remove
    except ImportError:
        return None
    from PIL import Image

    out = np.asarray(remove(Image.fromarray(rgb)).convert("RGBA"))
    alpha = out[..., 3]
    if (alpha > 10).sum() < min_area:
        return None
    ys, xs = np.where(alpha > 10)
    y1, y2 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
    x1, x2 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
    return out[y1:y2, x1:x2].astype(np.uint8)


def _matte(rgb: np.ndarray, method: str, **kw):
    if method == "rembg":
        r = matte_rembg(rgb, min_area=kw.get("min_area", 400))
        if r is not None:
            return r
    return matte_exg(rgb, **{k: v for k, v in kw.items()
                             if k in ("thresh", "min_area", "feather", "pad")})


def _sharpness(rgba: np.ndarray) -> float:
    """Variance of the Laplacian over the plant (alpha>0) region — a standard
    focus measure. Higher = sharper; soft/bokeh SD renders score low. Computed on
    the masked region only, so it doesn't depend on how much bare crop surrounds
    the plant."""
    from scipy import ndimage

    a = rgba[..., 3] > 10
    if int(a.sum()) < 50:
        return 0.0
    gray = rgba[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    lap = ndimage.laplace(gray)
    return float(lap[a].var())


# ---------------------------------------------------------------------------
# Phase 1 — SD3.5 single-weed generation + matte -> cutouts
# ---------------------------------------------------------------------------
class SD35CutoutGenerator:
    name = "sd35cut"

    DEFAULT_BASE = "stabilityai/stable-diffusion-3.5-medium"

    def __init__(self, pool: str = "sd35cut", base: str | None = None,
                 lora: str | None = None, device: str = "cuda",
                 steps: int = 28, guidance: float = 4.5,
                 render_size: int = 768, matte: str = "exg",
                 min_sharpness: float = 0.0, buffer_frac: float = 0.35,
                 min_clipiqa: float = 0.0, neg_prompt: bool = True,
                 stages: bool = False):
        self.pool = pool
        self.stages = stages   # vary growth-stage prompt per render (phenology coverage)
        self.base = base or self.DEFAULT_BASE
        self.lora = lora
        self.device = device
        self.steps = steps
        self.guidance = guidance
        self.render_size = render_size
        self.matte = matte
        self.min_sharpness = min_sharpness
        self.buffer_frac = buffer_frac
        self.min_clipiqa = min_clipiqa   # perceptual-quality gate on cutouts (0=off)
        self.neg_prompt = neg_prompt   # SD uses a negative prompt; FLUX does not
        self.pool_dir, self.cutouts_dir, self.backgrounds_dir = _pool_dirs(pool)
        self._pipe = None

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import StableDiffusion3Pipeline
        except ImportError as exc:
            raise ImportError(
                "pip install 'diffusers>=0.31' transformers accelerate peft "
                "sentencepiece protobuf") from exc
        pipe = StableDiffusion3Pipeline.from_pretrained(
            self.base, torch_dtype=torch.float16)
        if self.lora and Path(self.lora).exists():
            pipe.load_lora_weights(self.lora)
            print(f"[sd35cut] loaded domain LoRA from {self.lora}")
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def cutouts(self, per_class: int, seed: int = 0, batch: int = 4,
                keep_rejects: bool = False, only=None) -> dict[str, int]:
        """Render single-weed images per class, matte each, then keep the SHARPEST
        `per_class` cutouts. We over-render by buffer_frac and rank by variance of
        Laplacian so soft/bokeh SD renders are dropped instead of pasted (they blur
        further when the compositor upscales them). min_sharpness adds an absolute
        floor. Returns per-class kept counts."""
        from PIL import Image
        import torch

        pipe = self._load_pipe()
        gen = torch.Generator(device=self.device)
        counts = {c: 0 for c in classes.CLASS_NAMES}
        rej_dir = self.pool_dir / "rejects"
        for c in classes.CLASS_NAMES:
            (self.cutouts_dir / c).mkdir(parents=True, exist_ok=True)
        if keep_rejects:
            rej_dir.mkdir(parents=True, exist_ok=True)

        target_cand = int(math.ceil(per_class * (1.0 + self.buffer_frac)))
        for c in classes.CLASS_NAMES:
            if only and c not in only:      # render a subset of classes only
                continue
            stage_pool = stages_for(c) if self.stages else None
            srng = random.Random(seed * 7919 + hash(c) % 9973)
            cands: list[tuple[float, np.ndarray]] = []   # (sharpness, rgba)
            matte_rej = 0
            attempt = 0
            while len(cands) < target_cand:
                gen.manual_seed(seed * 100000 + hash(c) % 1000 + attempt)
                if stage_pool:
                    prompts = [_prompt(c, srng.choice(stage_pool)) for _ in range(batch)]
                else:
                    prompts = [_prompt(c)] * batch
                call_kw = dict(
                    prompt=prompts,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance,
                    height=self.render_size,
                    width=self.render_size,
                    generator=gen,
                )
                if self.neg_prompt:
                    call_kw["negative_prompt"] = [_NEG_PROMPT] * batch
                imgs = pipe(**call_kw).images
                for im in imgs:
                    rgba = _matte(np.asarray(im.convert("RGB")), self.matte)
                    if rgba is None:
                        matte_rej += 1
                        if keep_rejects:
                            im.save(rej_dir / f"{c}_matte_{matte_rej:04d}.png")
                        continue
                    cands.append((_sharpness(rgba), rgba))
                attempt += 1
                if attempt > per_class * 4:
                    print(f"[sd35cut] {c}: stopped collecting at {len(cands)} "
                          f"candidates (matte reject rate high)")
                    break

            if self.min_clipiqa > 0 and cands:
                try:
                    from src.qc.clip_iqa import score_arrays
                    rgbs = []
                    for _, rgba in cands:
                        bg = np.full(rgba.shape[:2] + (3,), 235, np.uint8)  # white matte
                        a = (rgba[..., 3:4].astype(np.float32) / 255.0)
                        rgbs.append((a * rgba[..., :3] + (1 - a) * bg).astype(np.uint8))
                    iqa = score_arrays(rgbs, device=self.device)
                    before = len(cands)
                    cands = [cd for cd, q in zip(cands, iqa) if q >= self.min_clipiqa]
                    print(f"[sd35cut] {c}: CLIP-IQA gate kept {len(cands)}/{before} "
                          f"(>= {self.min_clipiqa})")
                except Exception as exc:   # never let a QC dep crash the whole render
                    print(f"[sd35cut] {c}: CLIP-IQA gate SKIPPED ({exc}); "
                          f"using sharpness ranking only")

            cands.sort(key=lambda t: t[0], reverse=True)
            if cands:
                sv = np.array([s for s, _ in cands])
                print(f"[sd35cut] {c}: sharpness p10/p50/p90 = "
                      f"{np.percentile(sv,10):.0f}/{np.percentile(sv,50):.0f}/"
                      f"{np.percentile(sv,90):.0f}  (matte rejects={matte_rej})")
            kept = [(s, r) for s, r in cands if s >= self.min_sharpness][:per_class]
            for i, (s, rgba) in enumerate(kept):
                Image.fromarray(rgba, "RGBA").save(
                    self.cutouts_dir / c / f"sd_{c}_{seed:02d}_{i:05d}.png")
                counts[c] += 1
            if keep_rejects:                          # the softest, not-kept ones
                for j, (s, rgba) in enumerate(cands[len(kept):]):
                    Image.fromarray(rgba, "RGBA").save(
                        rej_dir / f"{c}_soft_{j:04d}_s{int(s)}.png")
            print(f"[sd35cut] {c}: kept {counts[c]} sharpest of {len(cands)} candidates")
        print(f"[sd35cut] total cutouts {counts} -> {self.cutouts_dir}")
        return counts


# ---------------------------------------------------------------------------
# Phase 1b — backgrounds: reuse a real-soil composite pool (domain-matched)
# ---------------------------------------------------------------------------
def copy_backgrounds(pool: str, from_pool: str = "composite") -> int:
    _, _, dst = _pool_dirs(pool)
    _, _, src = _pool_dirs(from_pool)
    if not src.exists() or not any(src.glob("*.png")):
        raise SystemExit(
            f"No backgrounds in {src} — build '{from_pool}' first "
            f"(scripts/gen_composite.py prep --pool {from_pool}).")
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.glob("*.png"):
        shutil.copy2(p, dst / p.name)
        n += 1
    print(f"[sd35cut] copied {n} backgrounds from '{from_pool}' -> {dst}")
    return n



class FluxCutoutGenerator(SD35CutoutGenerator):
    """FLUX.1-dev variant of the cut-and-composite generator. Reuses the entire
    pipeline (matte, sharpness cull, compositing, the LoRA crop set) — only the
    image-generation backend changes. FLUX takes no negative prompt and prefers
    bf16 + lower guidance. FLUX.1-dev is gated on HF (accept the license + token).
    """

    name = "fluxcut"
    DEFAULT_BASE = "black-forest-labs/FLUX.1-dev"

    def __init__(self, pool: str = "fluxcut", base: str | None = None,
                 lora: str | None = None, device: str = "cuda",
                 steps: int = 28, guidance: float = 3.5, render_size: int = 768,
                 matte: str = "exg", min_sharpness: float = 0.0,
                 buffer_frac: float = 0.35, cpu_offload: bool = False,
                 stages: bool = False):
        super().__init__(pool=pool, base=base, lora=lora, device=device,
                         steps=steps, guidance=guidance, render_size=render_size,
                         matte=matte, min_sharpness=min_sharpness,
                         buffer_frac=buffer_frac, neg_prompt=False, stages=stages)
        self.cpu_offload = cpu_offload

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import FluxPipeline
        except ImportError as exc:
            raise ImportError(
                "pip install 'diffusers>=0.31' transformers accelerate peft "
                "sentencepiece protobuf (FLUX needs a recent diffusers)") from exc
        pipe = FluxPipeline.from_pretrained(self.base, torch_dtype=torch.bfloat16)
        if self.lora and Path(self.lora).exists():
            pipe.load_lora_weights(self.lora)
            print(f"[fluxcut] loaded LoRA from {self.lora}")
        if self.cpu_offload:
            pipe.enable_model_cpu_offload()   # for GPUs that can't hold FLUX fully
        else:
            pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe
