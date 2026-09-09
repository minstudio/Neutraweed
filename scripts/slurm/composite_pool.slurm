#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_composite
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Composite an existing cutout pool into labelled scenes, then prebuild the
# hybrid datasets. CPU only — this is roughly 1.5 s per scene, so 2500 scenes is
# about an hour and the login node will kill it (10-min CPU cap, and it
# OOM-kills). Always run compositing through sbatch.
#
#   sbatch --export=ALL,POOL=sd35cut_perspecies scripts/slurm/composite_pool.slurm
#
# Native-resolution pool (for training at imgsz 1536):
#   sbatch --export=ALL,POOL=sd35cut_v2_n1536,IMGSZ=1536,TILE_SCALE=1.0 \
#       scripts/slurm/composite_pool.slurm
#
# PHOTOREAL (v7). Recomposites an EXISTING cutout bank with the fixed matte,
# cast shadows, soil-matched grain, overlap rejection and the QC passes. The
# cutouts are unchanged, so this isolates scene realism from generator quality —
# which is the one thing the five previous nulls never varied:
#   sbatch --export=ALL,POOL=sd35cut_v7,SRC_POOL=sd35cut_lora_v2,PHOTOREAL=1 \
#       scripts/slurm/composite_pool.slurm
#
# Ablations (same cutouts, one component off) with NO_SHADOW=1 / NO_NOISE=1 /
# NO_RELIGHT=1. There is no EXTRA_ARGS passthrough — an unrecognised variable in
# --export is silently ignored, so add a named flag here instead of inventing one
# at the command line.
#
# Skip the dataset prebuild with PREBUILD=0.
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

POOL=${POOL:?set POOL=<pool name under data/synthetic/>}
N=${N:-2500}
SEED=${SEED:-0}
IMGSZ=${IMGSZ:-1024}
BLEND=${BLEND:-feather}
SCALE_MODE=${SCALE_MODE:-prior}
BG_MODE=${BG_MODE:-tilematch}
TILE_SCALE=${TILE_SCALE:-}
CLSW=${CLSW:-uniform}
NLO=${NLO:-3}
NHI=${NHI:-12}
FMT=${FMT:-png}
PREBUILD=${PREBUILD:-1}
DET=${DET:-yolo26}
RATIOS=${RATIOS:-0.75:1.25:0.25}
SEEDS_LIST=${SEEDS_LIST:-0 1 2}
PHOTOREAL=${PHOTOREAL:-0}
SRC_POOL=${SRC_POOL:-}
NO_SHADOW=${NO_SHADOW:-0}
NO_NOISE=${NO_NOISE:-0}
NO_RELIGHT=${NO_RELIGHT:-0}
# 0.25 is the calibrated value (fit_appearance_prior). This defaulted to 0.45
# and so overrode the calibration on every pool built through this script,
# including v7 — which then measured +51% shadow depth and +81% shadow area
# against real. Leave it unset unless you are deliberately ablating.
SHADOW_STRENGTH=${SHADOW_STRENGTH:-0.25}
MAX_OVERLAP=${MAX_OVERLAP:-0.15}

# SRC_POOL symlinks an existing bank into a new pool name, so a photoreal pool
# reuses the exact cutouts of the pool it is being compared against. Same
# pixels in, different compositor — otherwise the comparison confounds the two.
if [ -n "$SRC_POOL" ]; then
  [ -d "data/synthetic/$SRC_POOL/cutouts" ] || { echo "no cutouts in SRC_POOL=$SRC_POOL"; exit 1; }
  mkdir -p "data/synthetic/$POOL"
  for sub in cutouts backgrounds; do
    if [ ! -e "data/synthetic/$POOL/$sub" ]; then
      ln -s "$(readlink -f "data/synthetic/$SRC_POOL/$sub")" "data/synthetic/$POOL/$sub"
      echo "[comp] linked $sub <- $SRC_POOL"
    fi
  done
fi

[ -d "data/synthetic/$POOL/cutouts" ] || { echo "no cutouts at data/synthetic/$POOL/cutouts"; exit 1; }
[ -d "data/synthetic/$POOL/backgrounds" ] || { echo "no backgrounds at data/synthetic/$POOL/backgrounds"; exit 1; }

if [ "$PHOTOREAL" = "1" ]; then
  BLEND=photoreal
  # bg-mode crop needs 1536-px soil tiles. Re-harvest them if the bank is still
  # the old 1024 one, otherwise the compositor falls back to the mosaic.
  BG_MODE=${BG_MODE_OVERRIDE:-crop}
  if [ "$BG_MODE" = "crop" ]; then
    # NEED must be the exact number the compositor will ask for. tile_scale is
    # imgsz/tiling.size = 1024/1536, which is 0.6666..., NOT 0.667 — rounding the
    # literal gives 1535 and every tile then fails the >= test by one pixel and
    # silently falls back to the mosaic.
    NEED=$(python -c "
import yaml
c = yaml.safe_load(open('configs/base.yaml'))
ts = ${TILE_SCALE:-0} or (float(c['detector']['imgsz']) / float(c['tiling']['size']))
print(int(round($IMGSZ / ts)))")
    HAVE=$(python -c "
import glob
from PIL import Image
f = sorted(glob.glob('data/synthetic/$POOL/backgrounds/*.png'))
print(min(Image.open(x).size[0] for x in f[:20]) if f else 0)")
    echo "[comp] background tiles are ${HAVE}px, bg-mode crop needs ${NEED}px"
    if [ "$HAVE" -lt "$NEED" ]; then
      echo "[comp] re-harvesting ${NEED}px soil tiles into $POOL (cutouts untouched)"
      python scripts/gen_composite.py backgrounds --pool "$POOL" --bg-size "$NEED" \
        --per-image "${BG_PER_IMAGE:-3}" --max-images "${BG_MAX_IMAGES:-400}"
    fi
  fi
  PR_ARGS="--prefer-downscale --bg-qc --cutout-qc --bg-match --max-upscale 1.15"
  PR_ARGS="$PR_ARGS --shadow-strength $SHADOW_STRENGTH --max-overlap $MAX_OVERLAP"
  [ "$NO_SHADOW" = "1" ] && PR_ARGS="$PR_ARGS --no-shadow"
  [ "$NO_NOISE" = "1" ] && PR_ARGS="$PR_ARGS --no-noise-match"
  [ "$NO_RELIGHT" = "1" ] && PR_ARGS="$PR_ARGS --no-relight"
else
  PR_ARGS=""
fi

echo "[comp] pool=$POOL n=$N imgsz=$IMGSZ scale=$SCALE_MODE bg=$BG_MODE fmt=$FMT blend=$BLEND"
[ -n "$PR_ARGS" ] && echo "[comp] photoreal: $PR_ARGS"
echo "[comp] cutouts per class:"
for d in data/synthetic/$POOL/cutouts/*/; do
  echo "  $(basename "$d")  $(find -L "$d" -name '*.png' | wc -l)"
done

# The prior is fitted from the tiled real labels on this machine; it is what
# --scale-mode prior samples instance sizes from.
python scripts/fit_scale_prior.py --split train

EXTRA=""
[ -n "$TILE_SCALE" ] && EXTRA="--tile-scale $TILE_SCALE"

python scripts/gen_sd35_cutout.py generate --pool "$POOL" --n "$N" --seed "$SEED" \
    --blend "$BLEND" --imgsz "$IMGSZ" \
    --scale-mode "$SCALE_MODE" --bg-mode "$BG_MODE" \
    --class-weights "$CLSW" --n-instances "$NLO" "$NHI" \
    --img-format "$FMT" $EXTRA $PR_ARGS

if [ "$PREBUILD" = "1" ]; then
  echo "[comp] prebuilding hybrid datasets ($DET, ratios=$RATIOS, seeds $SEEDS_LIST)"
  python scripts/run_sweep.py --generator "$POOL" --detector "$DET" \
      --ratios "$RATIOS" --seeds $SEEDS_LIST --no-baseline --build-only
fi

echo "[comp] done. Train with:"
echo "  sbatch --array=0-8%3 --export=ALL,DETECTOR=$DET,GEN=$POOL,\\"
echo "SWEEP_RATIOS=$RATIOS,SEEDS=\"$SEEDS_LIST\",EPOCHS=25 scripts/slurm/train_sweep_array.slurm"
