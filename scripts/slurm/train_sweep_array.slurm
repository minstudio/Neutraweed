#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_sweep_arr
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%A_%a.out
#-----------------------End job description---------------
# Ratio sweep as a SLURM ARRAY: ONE array task = ONE training (a single kind x
# ratio x seed), evaluated on the real test. One training per task keeps every
# task well under the 6h walltime, so nothing can wall the sweep; train_* resumes
# from last.pt after any kill; already-done cells re-eval instantly.
#
# Task layout: kinds {hybrid, matched} x ratios x seeds.
#   #tasks = 2 * (#ratios) * (#seeds).  For 8 ratios x 3 seeds -> 48 -> --array=0-47
#
# PREREQUISITE: datasets prebuilt (prebuild_sweep.slurm).
# Submit:
#   sbatch --array=0-47%8 --export=ALL,DETECTOR=yolo11,GEN=sd35cut_lora,SWEEP_RATIOS=0.25:2.0:0.25,SEEDS="0 1 2" \
#       scripts/slurm/train_sweep_array.slurm
#   # rtdetrv2: add --time=12:00:00
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

DET=${DETECTOR:-yolo11}; GEN=${GEN:-composite}
RATIOS_SPEC=${SWEEP_RATIOS:-0.25:2.0:0.25}
read -ra SARR <<< "${SEEDS:-0 1 2}"
readarray -t RARR < <(python3 -c "
lo,hi,st=[float(x) for x in '${RATIOS_SPEC}'.split(':')]
r=lo
while r<=hi+1e-9:
    print(round(r,3)); r+=st")
KINDS=(hybrid matched)
NS=${#SARR[@]}; NR=${#RARR[@]}; NCELLS=$(( NR * NS ))
i=$SLURM_ARRAY_TASK_ID
KIND=${KINDS[$(( i / NCELLS ))]}
rem=$(( i % NCELLS ))
RATIO=${RARR[$(( rem / NS ))]}
SEED=${SARR[$(( rem % NS ))]}
PCT=$(python3 -c "print(int(round(${RATIO}*100)))")
NAME=${KIND}_${DET}_${GEN}_r${PCT}
# Crowding-filtered datasets carry a suffix so they cannot be confused with the
# unfiltered ones. Must match scripts/run_sweep.py.
if [ -n "${MAX_OVERLAP:-}" ]; then
  XOV=$(python3 -c "print(int(round(float('${MAX_OVERLAP}')*100)))")
  [ "$XOV" -lt 100 ] && NAME="${NAME}_x${XOV}"
fi

case "$DET" in
  yolo11)   CONFIG=experiments/real_only_yolo11.yaml;   MODULE=src.detect.train_yolo;;
  yolo26)   CONFIG=experiments/real_only_yolo26.yaml;   MODULE=src.detect.train_yolo;;
  rtdetrv2) CONFIG=experiments/real_only_rtdetrv2.yaml; MODULE=src.detect.train_rtdetr;;
  *) echo "unknown DETECTOR=$DET"; exit 1;;
esac

DY=data/datasets/${NAME}/dataset.yaml
[ -f "$DY" ] || { echo "missing $DY — run prebuild_sweep.slurm first"; exit 1; }
EP=""
[ -n "${EPOCHS:-}" ] && EP="--epochs $EPOCHS"
echo "[cell $i] $NAME seed $SEED ${EPOCHS:+(epoch budget $EPOCHS)}"
python -m "$MODULE" --config "$CONFIG" --data "$DY" --seed "$SEED" --name "$NAME" $EP
python -m src.eval.metrics \
  --weights results/checkpoints/${NAME}_seed${SEED}/weights/best.pt --data "$DY" \
  --name ${NAME}_seed${SEED}
echo "[cell $i] done -> results/tables/runs/${NAME}_seed${SEED}_test.csv"
