"""Conditioning + layout sampling for ControlNet generation (CLAUDE.md §4 B/C).

The label is known by construction: we *build* the conditioning layout, so every
synthetic image ships with exact YOLO boxes — no segmentation of generated pixels.

Pieces:
  * PALETTE                 : fixed RGB colour per class for the semantic map.
  * InstanceBank            : real weed silhouettes harvested from the SAM2 masks
                              (Phase 0), grouped by class — the shape vocabulary.
  * LayoutSampler           : compose random scenes from the bank -> (semantic
                              map, instance boxes). This is what gives synthetic
                              *layout diversity* (vs just restyling real layouts).
  * control_image()         : turn a semantic map into the ControlNet input
                              (seg map as-is, or its Canny edges).
  * export_controlnet_pairs : dump (conditioning, target, prompt) from real images
                              to train a seg ControlNet with the diffusers example.

SAM2 masks being imperfect is fine here: the mask is a spatial prior, the label
comes from the layout we author.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..common import classes, paths

# Distinct, well-separated colours (RGB) — one per frozen class id. Background=black.
PALETTE: dict[str, tuple[int, int, int]] = {
    "SOLNI": (220, 40, 40),
    "POROL": (40, 200, 60),
    "SETVE": (40, 120, 220),
    "CYPRO": (230, 200, 40),
    "ECHCG": (180, 60, 220),
}
BACKGROUND = (0, 0, 0)


def palette_array() -> np.ndarray:
    """(num_classes, 3) RGB palette in frozen class-id order."""
    return np.array([PALETTE[c] for c in classes.CLASS_NAMES], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Instance bank: real weed silhouettes per class, harvested from SAM2 masks.
# ---------------------------------------------------------------------------
@dataclass
class InstanceBank:
    shapes: dict[str, list[np.ndarray]] = field(default_factory=dict)  # class -> [binary crops]

    @classmethod
    def build(cls, masks_split_dir: Path, max_per_class: int = 400, min_area: int = 64) -> "InstanceBank":
        """Harvest per-instance binary silhouettes from a SAM2 masks directory
        (data/real/masks/sam2/<split>/), using the label-map PNG + sidecar JSON."""
        from PIL import Image

        shapes: dict[str, list[np.ndarray]] = {c: [] for c in classes.CLASS_NAMES}
        for js in sorted(masks_split_dir.glob("*.json")):
            meta = json.loads(js.read_text(encoding="utf-8"))
            png = js.with_suffix(".png")
            if not png.exists():
                continue
            label_map = np.array(Image.open(png))
            for inst in meta.get("instances", []):
                c = inst["class"]
                if c not in shapes or len(shapes[c]) >= max_per_class:
                    continue
                # Crop to the precomputed mask_box first, then compare inside that
                # small window — avoids a full 20 MP `== id` scan per instance.
                box = inst.get("mask_box")
                if box:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    crop = (label_map[y1:y2, x1:x2] == inst["id"]).astype(np.uint8)
                else:                                   # fallback (no mask_box)
                    m = (label_map == inst["id"])
                    ys, xs = np.where(m)
                    if xs.size == 0:
                        continue
                    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
                if int(crop.sum()) < min_area:
                    continue
                shapes[c].append(crop)
        return cls(shapes=shapes)

    def save(self, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        flat, index = [], []
        for c, lst in self.shapes.items():
            for arr in lst:
                index.append((c, arr.shape[0], arr.shape[1]))
                flat.append(arr.ravel())
        np.savez_compressed(out, index=np.array(index, dtype=object), data=np.array(flat, dtype=object))

    @classmethod
    def load(cls, path: Path) -> "InstanceBank":
        z = np.load(path, allow_pickle=True)
        shapes: dict[str, list[np.ndarray]] = {c: [] for c in classes.CLASS_NAMES}
        for (c, h, w), flat in zip(z["index"], z["data"]):
            shapes[c].append(np.asarray(flat, dtype=np.uint8).reshape(int(h), int(w)))
        return cls(shapes=shapes)

    def counts(self) -> dict[str, int]:
        return {c: len(v) for c, v in self.shapes.items()}


# ---------------------------------------------------------------------------
# Layout sampler: compose a scene -> semantic map + boxes (the free labels).
# ---------------------------------------------------------------------------
@dataclass
class SampledLayout:
    semantic: np.ndarray             # HxWx3 uint8 colour map
    instances: list[tuple[str, tuple[int, int, int, int]]]  # (class, xyxy)


class LayoutSampler:
    def __init__(
        self,
        bank: InstanceBank,
        image_size: tuple[int, int] = (1024, 1024),
        n_instances: tuple[int, int] = (3, 12),
        scale_range: tuple[float, float] = (0.5, 1.4),
        class_weights: dict[str, float] | None = None,
        seed: int = 0,
    ):
        self.bank = bank
        self.W, self.H = image_size
        self.n_instances = n_instances
        self.scale_range = scale_range
        self.rng = random.Random(seed)
        usable = [c for c in classes.CLASS_NAMES if bank.shapes.get(c)]
        if not usable:
            raise SystemExit("InstanceBank is empty — run Phase 0 SAM2 masks first.")
        self.usable = usable
        self.class_weights = class_weights or {c: 1.0 for c in usable}

    def _place_one(self, semantic: np.ndarray) -> tuple[str, tuple[int, int, int, int]] | None:
        import cv2

        c = self.rng.choices(self.usable, weights=[self.class_weights.get(k, 1.0) for k in self.usable])[0]
        shape = self.rng.choice(self.bank.shapes[c])
        s = self.rng.uniform(*self.scale_range)
        sh, sw = max(8, int(shape.shape[0] * s)), max(8, int(shape.shape[1] * s))
        if sh >= self.H or sw >= self.W:
            return None
        shape = cv2.resize(shape, (sw, sh), interpolation=cv2.INTER_NEAREST)
        if self.rng.random() < 0.5:
            shape = shape[:, ::-1]
        x = self.rng.randint(0, self.W - sw)
        y = self.rng.randint(0, self.H - sh)
        colour = np.array(PALETTE[c], dtype=np.uint8)
        region = semantic[y:y + sh, x:x + sw]
        region[shape > 0] = colour
        ys, xs = np.where(shape > 0)
        return c, (x + int(xs.min()), y + int(ys.min()), x + int(xs.max()) + 1, y + int(ys.max()) + 1)

    def sample(self) -> SampledLayout:
        semantic = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        k = self.rng.randint(*self.n_instances)
        instances: list[tuple[str, tuple[int, int, int, int]]] = []
        for _ in range(k):
            placed = self._place_one(semantic)
            if placed:
                instances.append(placed)
        return SampledLayout(semantic=semantic, instances=instances)


# ---------------------------------------------------------------------------
# Control-image + label helpers
# ---------------------------------------------------------------------------
def control_image(semantic: np.ndarray, kind: str = "seg") -> np.ndarray:
    """Build the ControlNet conditioning image from a semantic colour map.

    'seg'   -> the colour map itself (needs a seg-trained ControlNet).
    'canny' -> Canny edges of the map (works with a pretrained SD3.5 Canny CN).
    """
    if kind == "seg":
        return semantic
    if kind == "canny":
        import cv2

        gray = cv2.cvtColor(semantic, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.repeat(edges[:, :, None], 3, axis=2)
    raise ValueError(f"unknown control kind {kind!r} (seg|canny)")


def instances_to_yolo(instances, w: int, h: int) -> list[str]:
    lines = []
    for cls_name, (x1, y1, x2, y2) in instances:
        cid = classes.class_id(cls_name)
        lines.append(
            f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
        )
    return lines


def label_map_to_semantic(label_map: np.ndarray, instances: list[dict]) -> np.ndarray:
    """Real SAM2 label-map -> semantic colour map (for ControlNet training pairs).

    Uses an instance-id -> colour lookup table applied in a single indexed pass,
    instead of one full-frame `== id` comparison per instance. On 20 MP frames
    with up to ~160 instances that's the difference between ~minutes and ~ms per
    image (see prep timing in docs)."""
    h, w = label_map.shape[:2]
    if not instances:
        return np.zeros((h, w, 3), dtype=np.uint8)
    max_id = max(int(i["id"]) for i in instances)
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for inst in instances:
        lut[int(inst["id"])] = PALETTE[inst["class"]]
    lm = label_map
    if int(lm.max()) > max_id:              # guard stray ids beyond the table
        lm = np.minimum(lm, max_id)
    return lut[lm]


def export_controlnet_pairs(split: str = "train", control_kind: str = "seg",
                            cond_size: int = 1024) -> Path:
    """Dump (conditioning_image, image, text) triples to train a seg ControlNet
    with diffusers' train_controlnet_sd3 example. Output: a folder with images/,
    conditioning/, and metadata.jsonl.

    The conditioning map is built at `cond_size`x`cond_size` (the training
    resolution), NOT the 20 MP source: the trainer resizes both image and
    conditioning to `--resolution` anyway, so generating full-res conditioning is
    pure waste (it dominated prep time). The square resize matches how the trainer
    squishes the 4:3 photo, so the two stay spatially aligned."""
    import cv2
    from PIL import Image

    masks_dir = paths.MASKS / "sam2" / split
    img_dir = paths.REAL / "images" / split
    out = paths.SYNTHETIC / "sd35" / "controlnet_pairs" / split
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "conditioning").mkdir(parents=True, exist_ok=True)

    rows = []
    for js in sorted(masks_dir.glob("*.json")):
        meta = json.loads(js.read_text(encoding="utf-8"))
        uid = js.stem
        src_img = next(iter(img_dir.glob(f"{uid}.*")), None)
        png = js.with_suffix(".png")
        if src_img is None or not png.exists():
            continue
        label_map = np.array(Image.open(png))
        if cond_size:                       # downscale to training res (id-preserving)
            label_map = cv2.resize(label_map, (cond_size, cond_size),
                                   interpolation=cv2.INTER_NEAREST)
        semantic = label_map_to_semantic(label_map, meta.get("instances", []))
        cond = control_image(semantic, control_kind)

        # Symlink the target photo (don't re-encode 1000+ multi-MP images to PNG —
        # that's GBs of duplicate data and minutes of CPU). Conditioning maps MUST
        # be PNG: lossless, so the discrete palette colours survive exactly.
        img_link = out / "images" / f"{uid}{src_img.suffix}"
        if not img_link.exists():
            try:
                os.symlink(src_img.resolve(), img_link)
            except OSError:
                shutil.copy2(src_img, img_link)   # fallback if symlinks unavailable
        Image.fromarray(cond).save(out / "conditioning" / f"{uid}.png")
        present = sorted({i["class"] for i in meta.get("instances", [])})
        # HF `imagefolder` requires *_file_name keys to load each path as an Image
        # column (verified: these yield columns `image`/`conditioning_image`, which
        # is what train_controlnet_sd3 reads). Plain `image`/`conditioning_image`
        # keys fail with "`file_name` or `*_file_name` must be present".
        rows.append({
            "image_file_name": f"images/{uid}{src_img.suffix}",
            "conditioning_image_file_name": f"conditioning/{uid}.png",
            "text": _prompt_for(present),
        })

    (out / "metadata.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(rows)} ControlNet pairs -> {out}")
    return out


def _prompt_for(class_names: list[str]) -> str:
    names = ", ".join(classes.CLASS_DISPLAY.get(c, c) for c in class_names) or "weeds"
    return (
        f"a top-down aerial photograph of bare agricultural soil in a tomato field, "
        f"with {names} weed seedlings, natural daylight, high detail"
    )
