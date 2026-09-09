#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_composite
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard      # CPU partition: no GPU needed, far less contended
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Cut-and-composite synthetic pool (Stage C, Option B). CPU-only,
# fire-and-forget — survives logout, no login-node CPU-limit kill. Submit with:
#     sbatch scripts/slurm/gen_composite.slurm
# Override count: sbatch --export=ALL,GEN_N=4000 scripts/slurm/gen_composite.slurm

set -euo pipefail

# --- paths (CONFIRM SCRATCH for your account) ---
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

# --- run parameters ---
N=${GEN_N:-2500}
SEED=${GEN_SEED:-0}

# no CUDA module (pure CPU / NumPy-OpenCV); apps/2021 makes conda visible
module load apps/2021

source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
which python && python --version

export TMPDIR="$SCRATCH/tmp"
mkdir -p "$TMPDIR"

cd "$PROJECT"

# prep is idempotent (cutout/background files are keyed by source id) — safe to
# re-run; extracts real weed cutouts + soil backgrounds from the SAM2 masks.
python scripts/gen_composite.py prep
python scripts/gen_composite.py generate --n "$N" --seed "$SEED"
python scripts/draw_overlays.py --pool composite --n 24   # QC overlays, ready to scp

echo "composite pool ($N imgs, seed $SEED) -> data/synthetic/composite/images"
echo "QC overlays -> data/synthetic/composite/overlays (scp these to eyeball)"
