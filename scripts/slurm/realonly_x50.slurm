#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_ro_x50
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%A_%a.out
#-----------------------End job description---------------
# The missing real-only baseline under the crowding filter. Without it every
# `d vs real` in the yolo26 [x50] block is nan, so v9, v9B and v9_blur — nine
# already-trained cells — cannot be quoted against anything.
#
# The dataset build is CPU but goes inside the GPU job because the login node
# kills anything over 10 minutes. Task 0 builds it; run serialised (%1) so the
# other two cannot race the build or the Ultralytics label cache.
#
#   sbatch --array=0-2%1 scripts/slurm/realonly_x50.slurm
#
# Then:
#   python scripts/sweep_readout.py --metric map50
#   sbatch --export=ALL,PATTERN='realonly_yolo26_x50_seed[0-9]',SPLIT=test \
#       scripts/slurm/cache_preds.slurm
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

DET=${DETECTOR:-yolo26}
GEN=${GEN:-sd35cut_v9}
MAX_OVERLAP=${MAX_OVERLAP:-0.50}
EPOCHS=${EPOCHS:-25}
read -ra SARR <<< "${SEEDS:-0 1 2}"
SEED=${SARR[$SLURM_ARRAY_TASK_ID]}

XOV=$(python3 -c "print(int(round(float('${MAX_OVERLAP}')*100)))")
NAME=realonly_${DET}_x${XOV}
DY=data/datasets/realonly_${DET}_real_x${XOV}/dataset.yaml

case "$DET" in
  yolo11)   CONFIG=experiments/real_only_yolo11.yaml;   MODULE=src.detect.train_yolo;;
  yolo26)   CONFIG=experiments/real_only_yolo26.yaml;   MODULE=src.detect.train_yolo;;
  rtdetrv2) CONFIG=experiments/real_only_rtdetrv2.yaml; MODULE=src.detect.train_rtdetr;;
  *) echo "unknown DETECTOR=$DET"; exit 1;;
esac

if [ ! -f "$DY" ]; then
  echo "[ro_x50] building $NAME (--max-overlap $MAX_OVERLAP)"
  python scripts/run_sweep.py --generator "$GEN" --detector "$DET" \
      --ratios "1.25:1.25:0.25" --max-overlap "$MAX_OVERLAP" --build-only
fi
[ -f "$DY" ] || { echo "[ro_x50] build produced no $DY — check run_sweep.py naming"; ls data/datasets | grep -i x${XOV}; exit 1; }

# Every dataset resolves back to data/real/labels/train, so concurrent cells
# thrash one shared Ultralytics cache. Serialised array + a clean slate.
find data/real data/datasets -name "*.cache" -delete 2>/dev/null || true

echo "[ro_x50] $NAME seed $SEED (epoch budget $EPOCHS)"
python -m "$MODULE" --config "$CONFIG" --data "$DY" --seed "$SEED" \
    --name "$NAME" --epochs "$EPOCHS"
python -m src.eval.metrics \
    --weights results/checkpoints/${NAME}_seed${SEED}/weights/best.pt --data "$DY" \
    --name ${NAME}_seed${SEED}
echo "[ro_x50] done -> results/tables/runs/${NAME}_seed${SEED}_test.csv"
