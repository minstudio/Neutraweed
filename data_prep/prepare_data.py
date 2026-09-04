"""Stage A end-to-end: scan -> frozen split -> VOC->YOLO materialisation.

This is the one command to run after dropping the raw VOC data in place.

Run:  python -m src.data_prep.prepare_data
      python -m src.data_prep.prepare_data --copy   (copy images, no hardlink)
"""

from __future__ import annotations

import argparse

from . import make_splits, scan_dataset, voc_to_yolo


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare real data (Stage A).")
    ap.add_argument("--copy", action="store_true", help="copy images instead of hardlinking")
    ap.add_argument("--symlink", action="store_true", help="symlink images")
    args = ap.parse_args()

    print("=" * 60, "\n[1/3] Scanning raw VOC sources\n", "=" * 60, sep="")
    scan_dataset.scan()

    print("\n", "=" * 60, "\n[2/3] Building frozen spatio-temporal field-holdout split\n", "=" * 60, sep="")
    make_splits.build_split()

    print("\n", "=" * 60, "\n[3/3] Converting VOC -> YOLO\n", "=" * 60, sep="")
    mode = "copy" if args.copy else "symlink" if args.symlink else "hardlink"
    voc_to_yolo.convert(mode)

    print("\nStage A complete. Real dataset ready at data/real/ (real.yaml).")


if __name__ == "__main__":
    main()
