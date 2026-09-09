"""FID between a synthetic pool and the real pool (Stage C).

Uses clean-fid for reproducible, resize-robust FID. Lower is better; report it
per generator and per ratio. This is a pool-level metric (not per image).

    pip install clean-fid
"""

from __future__ import annotations

from pathlib import Path


def compute_fid(real_dir: Path, synth_dir: Path, device: str = "cuda") -> float:
    try:
        from cleanfid import fid as cleanfid
    except ImportError as exc:
        raise ImportError("Install clean-fid: pip install clean-fid") from exc
    return float(
        cleanfid.compute_fid(str(real_dir), str(synth_dir), device=device, verbose=False)
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="FID(real_dir, synth_dir)")
    ap.add_argument("real_dir", type=Path)
    ap.add_argument("synth_dir", type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    print(f"FID = {compute_fid(args.real_dir, args.synth_dir, args.device):.3f}")
