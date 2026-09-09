"""Harvest single-crop soil backgrounds from the ORIGINAL photographs.

Why this exists. A real training tile is `tiling.size` (1536) native pixels shown
at `imgsz` (1024), so a background must supply 1536 px of continuous weed-free
ground to sit at the right magnification. `extract_backgrounds` searches
`data/real/images/<split>`, which holds 1536 px TILES — a 1536 window is the
whole tile, so there is exactly one candidate per image and it must contain no
weed at all. In a dense field that finds nothing, which is why the compositor
falls back to mosaicking four smaller crops and why every background carries a
visible join.

The original photographs are 3372x3703 and larger. Searching those instead,
33% of random 1536 px windows in TOMATO_1 are both free of every annotated weed
and bare soil. The constraint was never the field, it was where we looked.

  python scripts/harvest_full_backgrounds.py --pool sd35cut_v9 --size 1536

Then composite with a single crop instead of a mosaic:

  sbatch --export=ALL,POOL=sd35cut_v9,PHOTOREAL=1,BG_MODE_OVERRIDE=crop,\\
PREBUILD=0 scripts/slurm/composite_pool.slurm

SPLIT SAFETY. Backgrounds must never come from the validation or test field.
Source photographs are derived from the filenames of the tiles actually present
in `data/real/images/<split>`, so only photographs that contributed to that
split can donate soil. Nothing is inferred from directory names.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import paths  # noqa: E402

TILE_SUFFIX = re.compile(r"_t\d+$")


def photos_from_datasets(datasets, repo: Path, allow=None, verbose: bool = True):
    """Enumerate photographs directly from <repo>/<dataset>/Images.

    For when the tiles are on another machine. Only safe for a dataset that lies
    ENTIRELY inside the training split — TOMATO_1 is 2021 and is all training
    data, so it qualifies. TOMATO_2 is 2022 and holds Finca Santa Amalia (train)
    alongside Parcelas B and C (validation and test), so it must be restricted
    with `allow`, a set of photograph stems known to be training photos.
    """
    out = []
    for d in datasets:
        idir = repo / d / "Images"
        if not idir.exists():
            print(f"[harvest] {idir} does not exist")
            continue
        n = 0
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if allow is not None and img.stem not in allow:
                continue
            xml = repo / d / "Annotations" / f"{img.stem}.xml"
            out.append((img, xml if xml.exists() else None))
            n += 1
        if verbose:
            print(f"[harvest] {d}: {n} photographs"
                  + ("" if allow is None else " (restricted to the supplied list)"))
    return out


def source_photos(split: str, repo: Path, verbose: bool = True):
    """Photographs that contributed tiles to `split`, as (image, annotation)."""
    tiles = paths.REAL / "images" / split
    if not tiles.exists():
        raise SystemExit(f"[harvest] no tile directory at {tiles}")

    names = [p.stem for p in tiles.iterdir() if p.is_file()]
    stems = set()
    for s in names:
        if "__" not in s:
            continue
        dataset, rest = s.split("__", 1)
        stems.add((dataset, TILE_SUFFIX.sub("", rest)))

    if verbose:
        print(f"[harvest] {len(names)} tiles in {tiles}")
        print(f"[harvest] example tile names: {names[:3]}")
        print(f"[harvest] -> {len(stems)} distinct source photographs, "
              f"datasets: {sorted({d for d, _ in stems}) or 'NONE'}")

    out, missing = [], {}
    for dataset, stem in sorted(stems):
        found = None
        for ext in (".JPG", ".jpg", ".png", ".PNG", ".jpeg", ".JPEG"):
            img = repo / dataset / "Images" / f"{stem}{ext}"
            if img.exists():
                found = img
                break
        if found is None:
            missing[dataset] = missing.get(dataset, 0) + 1
            continue
        xml = repo / dataset / "Annotations" / f"{stem}.xml"
        out.append((found, xml if xml.exists() else None))

    if verbose and missing:
        print(f"[harvest] could NOT find the original photograph for: "
              + ", ".join(f"{k} x{v}" for k, v in sorted(missing.items())))
        for d in sorted(missing):
            p = repo / d / "Images"
            print(f"[harvest]   looked in {p}  (exists: {p.exists()})")
        print(f"[harvest] --repo is {repo}; point it at wherever TOMATO_1/ and "
              f"TOMATO_2/ live if they are not there")
    return out


def voc_boxes(xml: Path):
    if xml is None or not xml.exists():
        return None                      # unknown annotation -> refuse the photo
    root = ET.parse(xml).getroot()
    out = []
    for o in root.findall("object"):
        b = o.find("bndbox")
        try:
            out.append([int(float(b.find(t).text))
                        for t in ("xmin", "ymin", "xmax", "ymax")])
        except Exception:
            continue
    return out


def _mean_exg(im) -> float:
    a = np.asarray(im.resize((96, 96)), dtype=np.float32)
    return float(((2 * a[..., 1] - a[..., 0] - a[..., 2]) / (a.sum(2) + 1e-6)).mean())


def _local_green(im) -> float:
    """Greenest local patch, not the average.

    A seedling a few centimetres across barely moves the mean greenness of a
    1536 px window, so the mean gate passes windows with real weeds in them.
    Measured on Parcela B, 96% of windows that passed the mean gate contained
    visible unlabelled plants. Those matter more than ordinary noise: pasting a
    labelled synthetic weed onto soil that already holds an unlabelled real one
    teaches the detector that weeds are background.

    The annotations do not save you either — they cover the inter-row band only,
    so a window with no weed BOX can still hold a real plant near a crop row.
    """
    import cv2

    a = np.asarray(im.resize((384, 384)), dtype=np.float32)
    exg = (2 * a[..., 1] - a[..., 0] - a[..., 2]) / (a.sum(2) + 1e-6)
    return float(cv2.blur(exg, (24, 24)).max())


def _dark_blob(im) -> float:
    """Largest dark region, as a fraction of the window.

    Catches the photographer and the camera rig, whose shadow falls across a
    large part of many frames. It is a real feature of the field, but it is a
    big irregular near-black shape that would repeat across the pool.
    """
    import cv2

    a = np.asarray(im.resize((384, 384)), dtype=np.float32)
    g = a @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    d = (g < np.median(g) * 0.62).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, 8)
    return max([st[i, cv2.CC_STAT_AREA] for i in range(1, n)], default=0) / d.size


def _is_foreign(im) -> float:
    """Fraction of pale, colourless, textureless pixels — tiler padding, panels,
    reflective markers. Same test the compositor's background QC applies."""
    import cv2

    a = np.asarray(im.resize((256, 256)), dtype=np.uint8)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    V = hsv[:, :, 2].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    g = a.astype(np.float32) @ np.asarray([0.299, 0.587, 0.114], np.float32)
    tex = cv2.blur(np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.2)), (9, 9))
    foreign = (V > 230) & (S < 30) & (tex < 3.0)
    blown = a.astype(np.float32).min(axis=2) > 245
    return float(np.maximum(foreign, blown).mean())


def _sheets(review: Path, per: int = 40, cell: int = 210) -> None:
    """Contact sheets with the index printed on every crop, so bad ones can be
    named and deleted."""
    from PIL import Image, ImageDraw, ImageFont

    files = sorted(review.glob("bg_*.png"))
    if not files:
        return
    for old in review.glob("contact_*.png"):
        old.unlink()
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    cols = 8
    for s in range(0, len(files), per):
        chunk = files[s:s + per]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows * cell), (250, 250, 250))
        d = ImageDraw.Draw(sheet)
        for i, f in enumerate(chunk):
            with Image.open(f) as im:
                t = im.convert("RGB").resize((cell - 4, cell - 4), Image.LANCZOS)
            x, y = (i % cols) * cell, (i // cols) * cell
            sheet.paste(t, (x + 2, y + 2))
            n = int(f.stem.split("_")[1])
            d.rectangle([x + 2, y + 2, x + 54, y + 30], fill=(0, 0, 0))
            d.text((x + 8, y + 5), str(n), fill=(255, 255, 0), font=font)
        out = review / f"contact_{s // per:02d}.png"
        sheet.save(out)
        print(f"[harvest] {out}")


def _install(files, dest: Path) -> None:
    import shutil

    stage = dest.parent / f".{dest.name}.staging"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, stage / f.name)
    if dest.is_symlink():
        target = dest.resolve()
        dest.unlink()
        print(f"[harvest] backgrounds/ was a symlink to {target}; replaced with "
              f"{len(files)} single crops. The donor pool is untouched.")
    elif dest.exists():
        shutil.rmtree(dest)
    stage.replace(dest)
    print(f"[harvest] installed {len(files)} single-crop backgrounds -> {dest}")
    print("[harvest] composite with BG_MODE_OVERRIDE=crop — one real crop per "
          "scene, no mosaic")


def main() -> None:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--size", type=int, default=1536,
                    help="native window side; must equal tiling.size so the soil "
                         "is shown at the same magnification as a real tile")
    ap.add_argument("--per-photo", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=80)
    ap.add_argument("--max-photos", type=int, default=400)
    ap.add_argument("--max-green", type=float, default=0.05)
    ap.add_argument("--max-green-local", type=float, default=0.06,
                    help="reject a window whose greenest 24px patch exceeds this. "
                         "Catches unlabelled real weeds that the mean gate misses.")
    ap.add_argument("--max-dark-blob", type=float, default=0.03,
                    help="reject a window whose largest dark region exceeds this "
                         "fraction — the photographer and the camera rig.")
    ap.add_argument("--max-foreign", type=float, default=0.02)
    ap.add_argument("--margin", type=int, default=8,
                    help="keep this many px clear of every annotated weed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="enumerate photographs straight from <repo>/<name>/Images "
                         "instead of deriving them from tile filenames. Use when "
                         "the tiles live on another machine. ONLY safe for a "
                         "dataset wholly inside the training split (TOMATO_1); "
                         "pair with --photo-list for anything else.")
    ap.add_argument("--photo-list", default=None,
                    help="file of photograph stems, one per line, that are known "
                         "training photos. Required to use TOMATO_2 safely.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write the review set here instead of under the pool "
                         "(for harvesting on a machine that has the photographs "
                         "but not the pool)")
    ap.add_argument("--no-box-test", action="store_true",
                    help="select soil by greenness ALONE, never consulting the "
                         "annotations. Required when harvesting from the "
                         "validation or test field: using their labels to choose "
                         "training backgrounds is leakage, and it also destroys "
                         "the zero-annotation-cost claim that makes a target-domain "
                         "pool worth building. The cost is that a real unlabelled "
                         "weed can slip into a background, so review the contact "
                         "sheets more carefully and keep --max-green strict.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--review", action="store_true",
                    help="harvest into <pool>/bg_review/ and write contact sheets "
                         "instead of installing. Field equipment — the white pole "
                         "and the calibration board — cannot be detected reliably "
                         "(three automated tests failed), so the crops get looked "
                         "at before they are used.")
    ap.add_argument("--drop", default="",
                    help="comma/space separated indices to delete from the review "
                         "set, as numbered on the contact sheets")
    ap.add_argument("--commit", action="store_true",
                    help="install whatever survives in bg_review/ as the pool's "
                         "backgrounds")
    args = ap.parse_args()

    review = args.out_dir or (paths.SYNTHETIC / args.pool / "bg_review")

    if args.drop:
        idx = {int(t) for t in re.split(r"[,\s]+", args.drop.strip()) if t}
        gone = 0
        for i in sorted(idx):
            f = review / f"bg_{i:04d}.png"
            if f.exists():
                f.unlink()
                gone += 1
        left = len(list(review.glob("bg_*.png")))
        print(f"[harvest] deleted {gone}, {left} crops remain in {review}")
        _sheets(review)
        return

    if args.commit:
        files = sorted(review.glob("bg_*.png"))
        if not files:
            raise SystemExit(f"[harvest] nothing in {review}; run --review first")
        _install(files, paths.SYNTHETIC / args.pool / "backgrounds")
        return

    if args.datasets:
        allow = None
        if args.photo_list:
            allow = {ln.strip() for ln in Path(args.photo_list).read_text().split()
                     if ln.strip()}
            print(f"[harvest] restricting to {len(allow)} photographs from "
                  f"{args.photo_list}")
        photos = photos_from_datasets(args.datasets, args.repo, allow)
    else:
        photos = source_photos(args.split, args.repo)
    if not photos:
        raise SystemExit(f"no source photographs found for split={args.split}")
    random.Random(args.seed).shuffle(photos)
    photos = photos[:args.max_photos]
    n_annot = sum(1 for _, x in photos if x is not None)
    print(f"[harvest] {len(photos)} source photographs for split={args.split}, "
          f"{n_annot} with annotations")
    if n_annot < len(photos):
        print(f"[harvest] {len(photos) - n_annot} photographs have no annotation "
              f"file and will be SKIPPED — an unannotated photo cannot be proven "
              f"weed-free")

    out_dir = review if args.review else (
        paths.SYNTHETIC / args.pool / ".backgrounds.staging")
    if not args.dry_run:
        if out_dir.exists():
            for f in out_dir.glob("*.png"):
                try:
                    f.unlink()
                except OSError as exc:
                    print(f"[harvest] could not remove {f.name} ({exc}); it will "
                          f"be overwritten if the new harvest reaches that index, "
                          f"otherwise delete the directory by hand")
        out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    n = tried = rej_box = rej_green = rej_foreign = rej_local = rej_dark = 0
    if args.no_box_test:
        print("[harvest] --no-box-test: annotations will NOT be read. Soil is "
              "selected by greenness alone, so this is safe to run on the "
              "validation or test field.")

    for img, xml in photos:
        bs = [] if args.no_box_test else voc_boxes(xml)
        if bs is None:
            continue
        try:
            im = Image.open(img).convert("RGB")
        except Exception:
            continue
        W, H = im.size
        if W < args.size or H < args.size:
            continue
        m = args.margin
        got = 0
        for _ in range(args.attempts):
            if got >= args.per_photo:
                break
            tried += 1
            x = rng.randint(0, W - args.size)
            y = rng.randint(0, H - args.size)
            x2, y2 = x + args.size, y + args.size
            if any(not (bx2 + m <= x or bx1 - m >= x2 or by2 + m <= y or by1 - m >= y2)
                   for bx1, by1, bx2, by2 in bs):
                rej_box += 1
                continue
            crop = im.crop((x, y, x2, y2))
            if _mean_exg(crop) > args.max_green:
                rej_green += 1
                continue
            if _is_foreign(crop) > args.max_foreign:
                rej_foreign += 1
                continue
            if _local_green(crop) > args.max_green_local:
                rej_local += 1
                continue
            if _dark_blob(crop) > args.max_dark_blob:
                rej_dark += 1
                continue
            if not args.dry_run:
                crop.save(out_dir / f"bg_{n:04d}.png")
            got += 1
            n += 1
        im.close()

    print(f"[harvest] {tried} windows tried: {rej_box} hit a weed box, "
          f"{rej_green} too green overall, {rej_foreign} padding/panel, "
          f"{rej_local} unlabelled plant, {rej_dark} operator shadow")
    print(f"[harvest] kept {n} single-crop backgrounds at {args.size}px")

    if args.dry_run:
        print("[harvest] --dry-run, nothing written")
        return
    if n == 0:
        raise SystemExit("[harvest] harvested nothing; backgrounds/ untouched")

    if args.review:
        _sheets(review)
        print(f"\n[harvest] {n} crops in {review}")
        print("[harvest] look at the contact sheets, then delete the ones with the")
        print("[harvest] white pole or the calibration board in them:")
        print(f"[harvest]   python scripts/harvest_full_backgrounds.py "
              f"--pool {args.pool} --drop 3,17,22")
        print(f"[harvest] when the sheets look clean:")
        print(f"[harvest]   python scripts/harvest_full_backgrounds.py "
              f"--pool {args.pool} --commit")
        return

    _install(sorted(out_dir.glob("bg_*.png")),
             paths.SYNTHETIC / args.pool / "backgrounds")


if __name__ == "__main__":
    main()
