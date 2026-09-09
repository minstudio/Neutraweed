#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_evalruns
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Standalone test-set eval -> results/tables/runs/<run>_test.csv for every
# checkpoint matching PATTERN. Use when training produced checkpoints but the
# inline eval did not write CSVs (so sweep_readout.py can see the runs).
#   sbatch --export=ALL,PATTERN='hybrid_yolo26_sd35cut_lora_v4_*' scripts/slurm/eval_runs.slurm
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

PATTERN=${PATTERN:-'hybrid_yolo26_sd35cut_lora_v4_*'}
SPLIT=${SPLIT:-test}
echo "[evalruns] pattern=$PATTERN split=$SPLIT"
shopt -s nullglob
for d in results/checkpoints/$PATTERN; do
  name=$(basename "$d")
  w="$d/weights/best.pt"
  if [ ! -f "$w" ]; then echo "  skip $name (no best.pt)"; continue; fi
  ds="data/datasets/$(echo "$name" | sed -E 's/_seed[0-9]+$//')/dataset.yaml"
  if [ ! -f "$ds" ]; then echo "  skip $name (no $ds)"; continue; fi
  echo "  eval $name"
  python -m src.eval.metrics --weights "$w" --data "$ds" --split "$SPLIT" --name "$name"
done
echo "[evalruns] done -> results/tables/runs/"
