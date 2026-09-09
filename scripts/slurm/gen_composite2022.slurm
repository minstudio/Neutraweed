#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_composite2022
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard      # CPU partition: no GPU needed, far less contended
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Annotation-reduction HEADLINE rerun. Rebuild the composite
# pool so it carries the 2022 TARGET DOMAIN at ZERO annotation cost: backgrounds
# are weed-free 2022-Finca soil (harvested by the no-box + greenness heuristic,
# no boxes needed), cutouts are the existing real weed silhouettes. The original
# 2021-domain 'composite' pool is left untouched.
#
# Submit both variants:
#     sbatch scripts/slurm/gen_composite2022.slurm            # POOL=composite2022 (default)
#     sbatch --export=ALL,POOL=composite2022strict scripts/slurm/gen_composite2022.slurm
#
# POOL=composite2022        : 2022-Finca backgrounds + all cutouts (2021 + ECHCG-2022)
# POOL=composite2022strict  : 2022-Finca backgrounds + 2021-ONLY cutouts (no 2022 boxes
#                             anywhere -> strictly zero annotation cost)
# Override count/seed:  --export=ALL,POOL=composite2022,GEN_N=4000,GEN_SEED=0
# Dense 2022 field yields too few soil tiles? loosen the greenness gate / scan more:
#   --export=ALL,POOL=composite2022,MAX_GREEN=0.10,PER_IMAGE=4

set -euo pipefail

# --- paths (CONFIRM SCRATCH for your account) ---
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

# --- run parameters ---
POOL=${POOL:-composite2022}
N=${GEN_N:-2500}
SEED=${GEN_SEED:-0}
# soil-tile harvesting knobs (defaults match the 2021 pool; 2022-Finca is denser
# so raising MAX_GREEN / PER_IMAGE is often needed to get enough backgrounds)
MAX_GREEN=${MAX_GREEN:-0.05}
PER_IMAGE=${PER_IMAGE:-2}
MAX_IMAGES=${MAX_IMAGES:-400}

# Cutouts are reused from the already-built 'composite' pool (same real weed
# silhouettes; only the backgrounds change for the 2022 experiment) — no SAM2
# mask files needed on the cluster. CUTOUTS_FROM overridable.
CUTOUTS_FROM=${CUTOUTS_FROM:-composite}
CUT_ARGS="--cutouts-from $CUTOUTS_FROM"
# strict variant additionally restricts those cutouts to 2021 (no 2022 boxes at all)
if [ "$POOL" = "composite2022strict" ]; then
  CUT_ARGS="$CUT_ARGS --cut-sources TOMATO_1__"
fi

# no CUDA module (pure CPU / NumPy-OpenCV); apps/2021 makes conda visible
module load apps/2021

source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
which python && python --version

export TMPDIR="$SCRATCH/tmp"
mkdir -p "$TMPDIR"

cd "$PROJECT"

# prep: harvest 2022-Finca weed-free soil backgrounds (--bg-sources TOMATO_2__)
# + cutouts (all, or 2021-only for the strict variant). Idempotent.
python scripts/gen_composite.py prep --pool "$POOL" --bg-sources TOMATO_2__ $CUT_ARGS \
    --max-green "$MAX_GREEN" --per-image "$PER_IMAGE" --max-images "$MAX_IMAGES"
python scripts/gen_composite.py generate --pool "$POOL" --n "$N" --seed "$SEED"
python scripts/draw_overlays.py --pool "$POOL" --n 24   # QC overlays, ready to scp

echo "pool '$POOL' ($N imgs, seed $SEED) -> data/synthetic/$POOL/images"
echo "QC overlays -> data/synthetic/$POOL/overlays (scp these to eyeball 2022 backgrounds)"
