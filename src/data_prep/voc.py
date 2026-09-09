"""Pascal VOC parsing helpers shared by the scan / split / convert scripts.

The source annotations look like:

    <annotation>
      <folder>01</folder>                 # the plot id (our grouping key)
      <filename>01_IMG_4602.JPG</filename>
      <size><width>3372</width><height>3703</height></size>
      <object>
        <name>POROL</name>
        <bndbox><xmin>..</xmin><ymin>..</ymin><xmax>..</xmax><ymax>..</ymax></bndbox>
      </object>
      ...
    </annotation>
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BBox:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def clamp(self, width: int, height: int) -> "BBox":
        """Clamp to image bounds; some source boxes spill a few px over the edge."""
        return BBox(
            self.name,
            max(0.0, min(self.xmin, width)),
            max(0.0, min(self.ymin, height)),
            max(0.0, min(self.xmax, width)),
            max(0.0, min(self.ymax, height)),
        )

    @property
    def is_valid(self) -> bool:
        return self.xmax > self.xmin and self.ymax > self.ymin


@dataclass
class VocAnnotation:
    source: str          # "TOMATO_1" / "TOMATO_2"
    plot: str            # the <folder> value — grouping key for leakage-free splits
    filename: str
    width: int
    height: int
    objects: list[BBox] = field(default_factory=list)
    xml_path: Path | None = None
    image_path: Path | None = None

    @property
    def uid(self) -> str:
        """Globally unique stem, namespaced by source to avoid collisions."""
        return f"{self.source}__{Path(self.filename).stem}"

    @property
    def plot_key(self) -> str:
        """`<source>/<folder>` — the field key used by the holdout split."""
        return f"{self.source}/{self.plot}"

    @property
    def year(self) -> int:
        """Acquisition year. TOMATO_1 = 2021, TOMATO_2 = 2022."""
        return 2021 if self.source == "TOMATO_1" else 2022


def parse_voc(xml_path: Path, source: str) -> VocAnnotation:
    root = ET.parse(xml_path).getroot()

    def _text(node, tag, default=""):
        el = node.find(tag)
        return el.text if el is not None and el.text is not None else default

    size = root.find("size")
    width = int(float(_text(size, "width", "0"))) if size is not None else 0
    height = int(float(_text(size, "height", "0"))) if size is not None else 0

    objects: list[BBox] = []
    for obj in root.findall("object"):
        name = _text(obj, "name").strip()
        bb = obj.find("bndbox")
        if bb is None or not name:
            continue
        objects.append(
            BBox(
                name,
                float(_text(bb, "xmin", "0")),
                float(_text(bb, "ymin", "0")),
                float(_text(bb, "xmax", "0")),
                float(_text(bb, "ymax", "0")),
            )
        )

    return VocAnnotation(
        source=source,
        plot=_text(root, "folder").strip() or "UNKNOWN",
        filename=_text(root, "filename").strip() or (xml_path.stem + ".JPG"),
        width=width,
        height=height,
        objects=objects,
        xml_path=xml_path,
    )


def find_image(ann: VocAnnotation, images_dir: Path) -> Path | None:
    """Locate the image for an annotation, tolerating case/extension drift."""
    stem = Path(ann.filename).stem
    candidates = [ann.filename, f"{stem}.JPG", f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.png"]
    for c in candidates:
        p = images_dir / c
        if p.exists():
            return p
    # last resort: case-insensitive stem match
    for p in images_dir.glob("*"):
        if p.stem.lower() == stem.lower():
            return p
    return None


def iter_annotations(source: str, source_dir: Path):
    """Yield parsed annotations with image paths resolved, for one source dir."""
    ann_dir = source_dir / "Annotations"
    img_dir = source_dir / "Images"
    for xml_path in sorted(ann_dir.glob("*.xml")):
        ann = parse_voc(xml_path, source)
        ann.image_path = find_image(ann, img_dir)
        yield ann
