#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_cachepreds
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=standard-gpu
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# One inference pass per checkpoint, dumping raw detections + GT to
# results/tables/preds/. Everything downstream (per-class AP, AP by object size,
# bootstrap CIs over test tiles, paired A-vs-B) then runs on the login node from
# the cache, with no GPU and no retraining.
#   sbatch --export=ALL,PATTERN='realonly_yolo26_seed*' scripts/slurm/cache_preds.slurm
#   sbatch --export=ALL,PATTERN='hybrid_yolo26_sd35cut_lora_v2_r125_seed*' scripts/slurm/cache_preds.slurm
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
module load apps/2021 GCC/11.3.0 CUDA/11.8.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
export HF_HOME="$SCRATCH/hf_cache"
cd "$PROJECT"

PATTERN=${PATTERN:-realonly_*}
SPLIT=${SPLIT:-test}
BATCH=${BATCH:-8}
EXTRA=${EXTRA:-}
[ -n "${IMGSZ:-}" ] && EXTRA="$EXTRA --imgsz $IMGSZ"
[ -n "${TAG:-}" ]   && EXTRA="$EXTRA --tag $TAG"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[cache_preds] pattern=$PATTERN split=$SPLIT"
python scripts/cache_preds.py --pattern "$PATTERN" --split "$SPLIT" \
    --batch "$BATCH" --skip-existing $EXTRA

echo "[cache_preds] done -> results/tables/preds/"
