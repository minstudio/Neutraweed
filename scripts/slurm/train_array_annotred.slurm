#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_annotred
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%A_%a.out
#-----------------------End job description---------------
# Annotation-reduction experiment as a SLURM ARRAY.
# 4 train arms {annotred_A, annotred_B, annotred_T, annotred_M} x N seeds, each
# trained under the frozen protocol and evaluated on the tiled real test
# (Parcela C, 2022). Parameterised by DETECTOR and SEEDS, same as train_array.slurm.
# Each task writes its own results/tables/runs/ file (no CSV races); train_*
# resumes from last.pt after a cancellation/walltime kill.
#
# PREREQUISITE: build the 4 datasets once (composite pool must exist for arm T):
#   python -m src.datasets.build_annotation_reduction --arm all --seed 0
#
# Set --array to (4 * #seeds - 1). Task i -> config CONFIGS[i/NS], seed SEEDS[i%NS].
# Examples:
#   YOLO11, 3 seeds: sbatch --array=0-11 --export=ALL,DETECTOR=yolo11,SEEDS="0 1 2" scripts/slurm/train_array_annotred.slurm
#   YOLO26, 3 seeds: sbatch --array=0-11 --export=ALL,DETECTOR=yolo26,SEEDS="0 1 2" scripts/slurm/train_array_annotred.slurm

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
CONFIGS=(annotred_A annotred_B annotred_T annotred_M)
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
