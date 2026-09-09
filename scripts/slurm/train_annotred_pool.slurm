#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_annotred_pool
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%A_%a.out
#-----------------------End job description---------------
# Train ONE annotation-reduction dataset (any name under data/datasets/) across
# N seeds on a GPU node, then evaluate on the tiled real test. Use this for the
# 2022-background arm-T reruns (annotred_T_composite2022 / _composite2022strict)
# that the hardcoded train_array_annotred.slurm doesn't cover.
#
# MUST run via sbatch on the GPU partition — training on the login node falls to
# CPU and gets OOM-killed (everything as sbatch, L40S).
#
# PREREQUISITE: build the dataset first, e.g.
#   python -m src.datasets.build_annotation_reduction --arm T --generator composite2022
#
# Array index = seed index. Set --array to 0-(#seeds - 1). Examples:
#   sbatch --array=0-2 --export=ALL,NAME=annotred_T_composite2022,DETECTOR=yolo11,SEEDS="0 1 2" \
#       scripts/slurm/train_annotred_pool.slurm
#   sbatch --array=0-2 --export=ALL,NAME=annotred_T_composite2022strict,DETECTOR=yolo26,SEEDS="0 1 2" \
#       scripts/slurm/train_annotred_pool.slurm
#   # RT-DETR needs longer walltime: add --time=24:00:00
#
# train_* resumes from last.pt after a walltime/cancellation kill.

set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

NAME=${NAME:?set NAME=<dataset dir under data/datasets/, e.g. annotred_T_composite2022>}
DETECTOR=${DETECTOR:-yolo11}
read -ra SEEDS_ARR <<< "${SEEDS:-0 1 2}"
SEED=${SEEDS_ARR[$SLURM_ARRAY_TASK_ID]}

case "$DETECTOR" in
  yolo11)   CONFIG=experiments/real_only_yolo11.yaml;   MODULE=src.detect.train_yolo;;
  yolo26)   CONFIG=experiments/real_only_yolo26.yaml;   MODULE=src.detect.train_yolo;;
  rtdetrv2) CONFIG=experiments/real_only_rtdetrv2.yaml; MODULE=src.detect.train_rtdetr;;
  *) echo "unknown DETECTOR=$DETECTOR"; exit 1;;
esac

RUN=${NAME}_${DETECTOR}
DY=data/datasets/$NAME/dataset.yaml
[ -f "$DY" ] || { echo "missing $DY — build it first (build_annotation_reduction)"; exit 1; }
echo "[seed $SEED] detector=$DETECTOR data=$DY run=$RUN"

python -m "$MODULE" --config "$CONFIG" --data "$DY" --seed "$SEED" --name "$RUN"
W=results/checkpoints/${RUN}_seed${SEED}/weights/best.pt
python -m src.eval.metrics --weights "$W" --data "$DY" --name "${RUN}_seed${SEED}"

echo "[seed $SEED] done -> results/tables/runs/${RUN}_seed${SEED}_test.csv"
