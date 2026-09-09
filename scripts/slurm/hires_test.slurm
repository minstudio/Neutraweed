#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_hires
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%A_%a.out
#-----------------------End job description---------------
# RESOLUTION PROBE: retrain at imgsz 1536 (native tile size, no downscale to
# 1024) and ask whether giving the small weeds their pixels back lifts the
# classes the error breakdown showed as pure non-detections:
#   CYPRO <64px  recall 0.092    SETVE <64px  recall 0.118
# Both fail with ~zero false positives of any kind, which is the signature of
# missing signal rather than confusion — so resolution is the one lever the
# diagnosis actually supports.
#
# Layout: (config x seed). CONFIGS = {realonly, hybrid v2 r1.25}.
#   2 configs x 3 seeds -> --array=0-5.  Tasks 0-2 = realonly, 3-5 = hybrid.
#   sbatch --array=0-5%3 scripts/slurm/hires_test.slurm
#
# EPOCHS is set explicitly and defaults to 25, matching the 1024 runs. Do NOT
# leave it unset: a walltime kill stops bigger datasets at fewer epochs, which
# silently confounds resolution with training length (this happened to the whole
# v4/v5 sweep). A timeout here is recoverable — train_yolo resumes from last.pt
# toward the same epoch target, so resubmitting the identical command finishes.
#
# Budget at ~60k img/h for 1536: realonly 25 ep ~= 5.5 h, hybrid r125 ~= 12.5 h.
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
module load apps/2021 GCC/11.3.0 CUDA/11.8.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

IMGSZ=${IMGSZ:-1536}
BATCH=${BATCH:-6}
EPOCHS=${EPOCHS:-25}
CONFIG=${CONFIG:-experiments/real_only_yolo26.yaml}
TAG=${TAG:-hires}
read -ra SARR <<< "${SEEDS:-0 1 2}"
NS=${#SARR[@]}

i=$SLURM_ARRAY_TASK_ID
CFG=$(( i / NS ))
SEED=${SARR[$(( i % NS ))]}

if [ "$CFG" = "0" ]; then
  python -m src.datasets.build_hybrid --name realonly_base --composition real-only
  DS=data/datasets/realonly_base/dataset.yaml
  NAME=realonly_yolo26_${TAG}
else
  DS=data/datasets/hybrid_yolo26_sd35cut_lora_v2_r125/dataset.yaml
  NAME=hybrid_yolo26_v2_${TAG}
fi

# A finished run leaves last.pt behind; train_yolo would resume it and
# Ultralytics errors out on an already-complete schedule. Changing TAG gives the
# new configuration its own checkpoint directory.
if [ -f "results/checkpoints/${NAME}_seed${SEED}/weights/last.pt" ]; then
  echo "[hires] NOTE resuming existing ${NAME}_seed${SEED}. If that run already "
  echo "[hires]      finished, pass a different TAG= or delete the directory."
fi

echo "[hires] $NAME seed $SEED  config=$CONFIG imgsz=$IMGSZ batch=$BATCH epochs=$EPOCHS"
echo "[hires] data=$DS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m src.detect.train_yolo --config "$CONFIG" --data "$DS" --seed "$SEED" \
  --name "$NAME" --imgsz "$IMGSZ" --batch "$BATCH" --epochs "$EPOCHS"

python -m src.eval.metrics --weights results/checkpoints/${NAME}_seed${SEED}/weights/best.pt \
  --data "$DS" --split test --name ${NAME}_seed${SEED}

for SP in test val; do
  python scripts/cache_preds.py --pattern "${NAME}_seed${SEED}" --split "$SP" \
    --data "$DS" --imgsz "$IMGSZ"
done

echo "[hires] $NAME seed $SEED done -> results/tables/runs/ + results/tables/preds/"
