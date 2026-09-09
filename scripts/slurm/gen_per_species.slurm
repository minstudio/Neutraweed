#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_gen_sp
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%A_%a.out
#-----------------------End job description---------------
# Render cutouts for one species with that species' own LoRA.
#
#   sbatch --array=0-4%2 scripts/slurm/gen_per_species.slurm
#
# Writes data/synthetic/sd35cut_<cls>/cutouts/<CLS>/. Merge them afterwards with
#   python scripts/merge_cutout_pools.py --out sd35cut_perspecies
#
# LORA_DIR points at the per-species weights by default; set LORA_DIR to the
# control (lora_multi_strat) with SUFFIX="" to render the control pool with the
# same prompts and counts, which is the only way the two are comparable.
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

CLASSES=(SOLNI POROL SETVE CYPRO ECHCG)
PER_CLASS=${PER_CLASS:-600}
SEED=${SEED:-0}
RENDER=${RENDER:-1024}
MATTE=${MATTE:-exg}
GUIDANCE=${GUIDANCE:-4.5}
STEPS=${STEPS:-28}
POOL_PREFIX=${POOL_PREFIX:-sd35cut}
LORA_DIR=${LORA_DIR:-data/synthetic/sd35/weights}
SHARED_LORA=${SHARED_LORA:-}

module load apps/2021
module load GCC/11.3.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH:-8.9}"
export HF_HUB_CACHE="$SCRATCH/hf_cache"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"
cd "$PROJECT"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"

CLS=${CLASSES[$SLURM_ARRAY_TASK_ID]}
if [ -n "$SHARED_LORA" ]; then
  LORA="$SHARED_LORA"
  POOL="${POOL_PREFIX}_shared_${CLS,,}"
else
  LORA="${LORA_DIR}/lora_${CLS,,}"
  POOL="${POOL_PREFIX}_${CLS,,}"
fi
[ -d "$LORA" ] || { echo "no LoRA at $LORA — run lora_per_species.slurm first"; exit 1; }

echo "[gen] $CLS  lora=$LORA  pool=$POOL  per_class=$PER_CLASS  render=$RENDER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --stages varies the growth-stage phrase. It only produces real morphological
# variety if the LoRA was trained with size-matched captions (export_weed_crops
# --size-captions); otherwise every phrase maps to the same seedling appearance.
python scripts/gen_sd35_cutout.py cutouts \
  --backend sd35 --pool "$POOL" --classes "$CLS" \
  --per-class "$PER_CLASS" --seed "$SEED" \
  --lora "$LORA" --matte "$MATTE" --render-size "$RENDER" \
  --steps "$STEPS" --guidance "$GUIDANCE" --stages --keep-rejects

echo "[gen] native cutout sizes for $CLS:"
python3 - "$POOL" "$CLS" <<'PY'
import sys, glob, numpy as np
from PIL import Image
pool, cls = sys.argv[1], sys.argv[2]
f = glob.glob(f"data/synthetic/{pool}/cutouts/{cls}/*.png")
if f:
    a = np.array([max(Image.open(x).size) for x in f])
    print("  n=%d  p10 %.0f  p50 %.0f  p90 %.0f  max %.0f"
          % (a.size, *np.percentile(a, [10, 50, 90]), a.max()))
else:
    print("  NONE PRODUCED")
PY

echo "[gen] $CLS done -> data/synthetic/$POOL"
