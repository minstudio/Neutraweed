#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_v4feather
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# FAST v4: reuse the v3 SD cutouts + backgrounds, re-composite the whole-scene
# images with the colour-preserving FEATHER blend (fixes the Poisson soil-colour
# cast), then prebuild the hybrid datasets for a single-seed YOLO26 test.
# CPU only -> standard partition (never the login node: it OOM-kills).
#   sbatch scripts/slurm/gen_v4_feather.slurm
# then (or chain with --dependency=afterok:<jobid>):
#   sbatch --array=0-2 --export=ALL,DETECTOR=yolo26,GEN=sd35cut_lora_v4,\
#       SWEEP_RATIOS=0.75:1.25:0.25,SEEDS="0" scripts/slurm/train_sweep_array.slurm
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

SRC=${SRC:-sd35cut_lora_v3}
DST=${DST:-sd35cut_lora_v4}
RATIOS=${RATIOS:-0.75:1.25:0.25}
DET=${DET:-yolo26}

# v4 inherits v3's cutouts/backgrounds unchanged (symlink = instant, no copy);
# only the whole-scene composite changes (Poisson -> feather).
mkdir -p data/synthetic/$DST
ln -sfn ../$SRC/cutouts     data/synthetic/$DST/cutouts
ln -sfn ../$SRC/backgrounds data/synthetic/$DST/backgrounds
rm -rf data/synthetic/$DST/images data/synthetic/$DST/labels

N=$(find data/synthetic/$SRC/images -name '*.png' | wc -l)
echo "[v4] re-compositing $N scenes from $SRC cutouts with feather blend"
python scripts/gen_sd35_cutout.py generate --pool $DST --n "$N" --seed 0 --blend feather --imgsz 1024

echo "[v4] prebuilding hybrid datasets ($DET, ratios=$RATIOS, seed 0)"
python scripts/run_sweep.py --generator $DST --detector $DET \
    --ratios "$RATIOS" --seeds 0 --no-baseline --build-only

echo "[v4] ready. Train with train_sweep_array.slurm --array=0-2 GEN=$DST"
