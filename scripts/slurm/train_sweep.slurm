#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_sweep
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Stage E — detector sweep (Stage E/F). Builds real-only + composite
# hybrid datasets, trains YOLO11 over seeds under the frozen protocol, evaluates
# each on the real Parcela-C test set. Fire-and-forget. Submit with:
#     sbatch scripts/slurm/train_sweep.slurm
# Wider sweep: sbatch --export=ALL,SWEEP_RATIOS=0.5:2.0:0.5,SWEEP_SEEDS="0 1 2" scripts/slurm/train_sweep.slurm

set -euo pipefail

# --- paths (CONFIRM SCRATCH) ---
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

# --- sweep scope (start focused: ratio 1.0 only, 3 seeds; widen via env) ---
DETECTOR=${DETECTOR:-yolo11}
GEN=${GEN:-composite}
RATIOS=${SWEEP_RATIOS:-1.0:1.0:1.0}
SEEDS=${SWEEP_SEEDS:-0 1 2}

# no CUDA module (cuDNN ships with torch); apps/2021 makes conda visible
module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
which python && python --version

export TMPDIR="$SCRATCH/tmp"
mkdir -p "$TMPDIR"
pip install -q ultralytics      # idempotent; yolo11m.pt auto-downloads on first use

cd "$PROJECT"

# real-only baseline + composite hybrid(s) + matched-size controls, eval on test
python scripts/run_sweep.py --generator "$GEN" --detector "$DETECTOR" \
  --ratios "$RATIOS" --seeds $SEEDS --matched-control

echo "sweep done -> results/tables/metrics.csv"
echo "aggregate: python -m src.eval.stats --metric map50_95 ; python -m src.eval.plots --metric map50_95"
