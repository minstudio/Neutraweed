#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_prebuild
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Assemble every hybrid + matched dataset for the ratio grid ONCE (CPU), so the
# GPU array only trains and never races on dataset writes. Run before the array.
#   sbatch --export=ALL,DETECTOR=yolo11,GEN=sd35cut_lora,SWEEP_RATIOS=0.25:2.0:0.25 scripts/slurm/prebuild_sweep.slurm
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"
DET=${DETECTOR:-yolo11}; GEN=${GEN:-composite}; RATIOS=${SWEEP_RATIOS:-0.25:2.0:0.25}
XARG=""
[ -n "${MAX_OVERLAP:-}" ] && XARG="--max-overlap $MAX_OVERLAP"
[ "${STRICT_UNIQUE:-0}" = "1" ] && XARG="$XARG --strict-unique"
[ -n "$XARG" ] && echo "[prebuild] $XARG"
python scripts/run_sweep.py --generator "$GEN" --detector "$DET" \
  --ratios "$RATIOS" --matched-control --no-baseline --build-only $XARG
echo "[prebuild] datasets ready for $DET/$GEN ratios $RATIOS -> data/datasets/"
