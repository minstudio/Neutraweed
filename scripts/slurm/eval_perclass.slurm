#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_perclass
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%A.out
#-----------------------End job description---------------
# Per-class AP for every annotation-reduction checkpoint, evaluated on the tiled
# real test (Parcela C, 2022). Re-runs model.val per best.pt and writes
# results/tables/runs_perclass/<run>_<split>.csv. Tests whether the small
# composite2022 recovery is concentrated in ECHCG (the 2022-heavy class).
#
# Runs on GPU (val on login falls to CPU and is slow). Single job iterates all
# matching checkpoints.
#   sbatch scripts/slurm/eval_perclass.slurm
#   sbatch --export=ALL,PATTERN='annotred_T*',SPLIT=test scripts/slurm/eval_perclass.slurm

set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

PATTERN=${PATTERN:-'annotred_*'}
SPLIT=${SPLIT:-test}
echo "[perclass] pattern=$PATTERN split=$SPLIT"
python scripts/eval_perclass.py --pattern "$PATTERN" --split "$SPLIT" --skip-existing
echo "[perclass] done -> results/tables/runs_perclass/"
