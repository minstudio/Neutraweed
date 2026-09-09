#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_qc
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# FID + CLIP-IQA across synthetic pools (one GPU pass, ~minutes/pool). Needs a
# GPU for the Inception (FID) and CLIP (IQA) forward passes — never the login node.
#   sbatch --export=ALL,POOLS="composite sd35cut sd35cut_lora" scripts/slurm/qc_pools.slurm
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
export HF_HOME="$SCRATCH/hf"; mkdir -p "$HF_HOME"
cd "$PROJECT"
pip install -q clean-fid torchmetrics 2>/dev/null || true
POOLS=${POOLS:-composite sd35cut sd35cut_lora}
python -c "import torch;print('device=cuda' if torch.cuda.is_available() else 'device=cpu (WRONG)')"
python scripts/qc_pools.py --pools $POOLS --sample "${SAMPLE:-500}"
echo "[qc] done -> results/tables/qc_pools.csv"
