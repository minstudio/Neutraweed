#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_v5scale
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# v5 = v4 cutouts (the 1024 px learned-matte renders — higher fidelity, and more
# native headroom for the large tail of the real size distribution) re-composited
# with MATCHED SCENE STATISTICS. v4 already uses the feather blend, so v5 vs v4
# is a clean one-variable A/B on scene statistics alone:
#   --scale-mode prior   instance sizes drawn from the measured real box-size
#                        distribution instead of uniform(60,260), which never
#                        produced an instance above 260 px (16% of real boxes,
#                        35% of POROL and ECHCG) nor below 60 px.
#   --bg-mode tilematch  soil rendered at imgsz/tiling.size, the same 0.667x
#                        downscale a real training tile gets. The old 1:1 crop
#                        made every synthetic image 1.5x finer-grained than the
#                        real half of the same training set.
# CPU only -> standard partition (never the login node: it OOM-kills).
#   sbatch scripts/slurm/gen_v5_scalefit.slurm
# then (3 ratios x 3 seeds = 9 hybrid cells -> array 0-8; 9-17 would be the
# matched-size controls, which v2 already has):
#   sbatch --array=0-8%3 --export=ALL,DETECTOR=yolo26,GEN=sd35cut_lora_v5,\
#       SWEEP_RATIOS=0.75:1.25:0.25,SEEDS="0 1 2" scripts/slurm/train_sweep_array.slurm
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

SRC=${SRC:-sd35cut_lora_v4}
DST=${DST:-sd35cut_lora_v5}
RATIOS=${RATIOS:-0.75:1.25:0.25}
DET=${DET:-yolo26}
CLSW=${CLSW:-uniform}
NLO=${NLO:-3}
NHI=${NHI:-12}
FMT=${FMT:-png}
SEEDS_LIST=${SEEDS_LIST:-0 1 2}

echo "[v5] fitting the real box-size prior (data/real is tiled here)"
python scripts/fit_scale_prior.py --split train

# v4's own cutouts/backgrounds are symlinks into v3; resolve to the real dirs so
# v5 does not stack a third link.
CUT=$(readlink -f data/synthetic/$SRC/cutouts)
BG=$(readlink -f data/synthetic/$SRC/backgrounds)
[ -d "$CUT" ] || { echo "no cutouts at data/synthetic/$SRC/cutouts (-> $CUT)"; exit 1; }
[ -d "$BG" ]  || { echo "no backgrounds at data/synthetic/$SRC/backgrounds (-> $BG)"; exit 1; }
echo "[v5] cutouts     <- $CUT"
echo "[v5] backgrounds <- $BG"
echo "[v5] cutout native longest side, per class:"
python3 - "$CUT" <<'PY'
import sys, numpy as np
from pathlib import Path
from PIL import Image
root = Path(sys.argv[1])
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    s = []
    for f in sorted(d.glob("*.png")):
        with Image.open(f) as im:
            s.append(max(im.size))
    if s:
        a = np.asarray(s)
        print("  %-8s n=%-5d p10 %-5.0f p50 %-5.0f p90 %-5.0f max %.0f"
              % (d.name, a.size, *np.percentile(a, [10, 50, 90]), a.max()))
PY

mkdir -p data/synthetic/$DST
ln -sfn "$CUT" data/synthetic/$DST/cutouts
ln -sfn "$BG"  data/synthetic/$DST/backgrounds
rm -rf data/synthetic/$DST/images data/synthetic/$DST/labels

# NIMG overrides the scene count. Unset it and the pool matches the source pool
# 1:1 (2500), which is what v2/v3/v4 have — the comparable setting. Set it to
# round(max_ratio * 13110) and every hybrid image is a DISTINCT scene instead of
# build_hybrid re-listing the same 2500 files ~6.6x at r125:
#   sbatch --export=ALL,DST=sd35cut_lora_v6,NIMG=16388 scripts/slurm/gen_v5_scalefit.slurm
# Budget ~1.3 MB/scene on BeeGFS (2500 -> ~3 GB, 16388 -> ~21 GB) and bump
# --time; compositing runs about 1.5 s/scene on 8 CPUs.
SRCN=$(find -L data/synthetic/$SRC/images -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)
N=${NIMG:-$SRCN}
echo "[v5] source pool $SRC has $SRCN scenes; generating $N into $DST (format=$FMT)"
if [ "$N" -gt "$SRCN" ]; then
  echo "[v5] NOTE $DST will hold MORE unique scenes than $SRC — the pools differ on"
  echo "[v5]      scene count as well as scene statistics, so treat this as its own"
  echo "[v5]      arm rather than a one-variable A/B against $SRC."
fi
python scripts/gen_sd35_cutout.py generate --pool $DST --n "$N" --seed 0 \
    --blend feather --imgsz 1024 \
    --scale-mode prior --bg-mode tilematch \
    --class-weights "$CLSW" --n-instances "$NLO" "$NHI" --img-format "$FMT"

echo "[v5] prebuilding hybrid datasets ($DET, ratios=$RATIOS, seeds $SEEDS_LIST)"
python scripts/run_sweep.py --generator $DST --detector $DET \
    --ratios "$RATIOS" --seeds $SEEDS_LIST --no-baseline --build-only

echo "[v5] ready. Train with train_sweep_array.slurm GEN=$DST"
