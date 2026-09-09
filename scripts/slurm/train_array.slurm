#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_arr
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%A_%a.out
#-----------------------End job description---------------
# Stage E focused comparison as a SLURM ARRAY. Parameterised
# by DETECTOR and SEEDS. 3 configs {real-only, composite-hybrid r50, matched r50}
# x N seeds, each one train + eval on the tiled real test (Parcela C). Tasks
# parallelise; each writes its own result file (results/tables/runs/) so no CSV
# races; train_* resumes from last.pt after a cancellation/walltime kill.
#
# PREREQUISITE: build the 3 datasets once (realonly, hybrid_composite_r50,
# matched_composite_r50). Set --array to (#configs * #seeds - 1). Examples:
#   5-seed YOLO11 (add seeds 3,4):  sbatch --array=0-5 --export=ALL,DETECTOR=yolo11,SEEDS="3 4" scripts/slurm/train_array.slurm
#   YOLO26, 3 seeds:                sbatch --array=0-8 --export=ALL,DETECTOR=yolo26,SEEDS="0 1 2" scripts/slurm/train_array.slurm
#   RT-DETRv2, 3 seeds:             sbatch --array=0-8 --export=ALL,DETECTOR=rtdetrv2,SEEDS="0 1 2" scripts/slurm/train_array.slurm

set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

DETECTOR=${DETECTOR:-yolo11}
read -ra SEEDS_ARR <<< "${SEEDS:-0 1 2}"
CONFIGS=(realonly hybrid_composite_r50 matched_composite_r50)
NS=${#SEEDS_ARR[@]}

i=$SLURM_ARRAY_TASK_ID
NAME=${CONFIGS[$(( i / NS ))]}
SEED=${SEEDS_ARR[$(( i % NS ))]}

case "$DETECTOR" in
  yolo11)   CONFIG=experiments/real_only_yolo11.yaml;   MODULE=src.detect.train_yolo;;
  yolo26)   CONFIG=experiments/real_only_yolo26.yaml;   MODULE=src.detect.train_yolo;;
  rtdetrv2) CONFIG=experiments/real_only_rtdetrv2.yaml; MODULE=src.detect.train_rtdetr;;
  *) echo "unknown DETECTOR=$DETECTOR"; exit 1;;
esac

RUN=${NAME}_${DETECTOR}
DY=data/datasets/$NAME/dataset.yaml
echo "[task $i] detector=$DETECTOR config=$NAME data=$DY seed=$SEED run=$RUN"

python -m "$MODULE" --config "$CONFIG" --data "$DY" --seed "$SEED" --name "$RUN"
W=results/checkpoints/${RUN}_seed${SEED}/weights/best.pt
python -m src.eval.metrics --weights "$W" --data "$DY" --name "${RUN}_seed${SEED}"

echo "[task $i] done -> results/tables/runs/${RUN}_seed${SEED}_test.csv"
