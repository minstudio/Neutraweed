#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_sd35cut_paste
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Phase 2 of the SD-cutout arm (CPU): copy real-soil backgrounds + composite
# paste the SD cutouts into a labelled pool. This is the step that OOMs on the
# login node (its cgroup kills non-trivial jobs, same as CPU training) — run it
# here on the standard CPU partition with real memory, exactly like the
# composite2022 pool was built.
#
# PREREQUISITE: cutouts already rendered (gen_sd35_cutout.slurm, GPU).
#   sbatch --export=ALL,POOL=sd35cut,N=2500,SEED=0,FROM_POOL=composite2022 \
#       scripts/slurm/gen_sd35_cutout_paste.slurm

set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

POOL=${POOL:-sd35cut}
N=${N:-2500}
SEED=${SEED:-0}
FROM_POOL=${FROM_POOL:-composite2022}

[ -d "data/synthetic/$POOL/cutouts" ] || { echo "no cutouts for $POOL — run gen_sd35_cutout.slurm (GPU) first"; exit 1; }

python scripts/gen_sd35_cutout.py backgrounds --pool "$POOL" --from-pool "$FROM_POOL"
python scripts/gen_sd35_cutout.py generate    --pool "$POOL" --n "$N" --seed "$SEED"
python scripts/draw_overlays.py --pool "$POOL" --n 24 2>/dev/null || echo "(overlays QC skipped)"
echo "[sd35cut] pool ready -> data/synthetic/$POOL/{images,labels}. Sweep with GEN=$POOL."
