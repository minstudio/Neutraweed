#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_crops
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Export per-class real weed crops (+ captions) for the species LoRA. CPU, but
# it scans all 13,110 train tiles -> too heavy for the login node (gets killed).
# Run on the standard partition.
#   sbatch --export=ALL,MAXPC=400,SIZE=768 scripts/slurm/export_weed_crops.slurm

set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

MAXPC=${MAXPC:-400}
SIZE=${SIZE:-768}
python scripts/export_weed_crops.py --max-per-class "$MAXPC" --size "$SIZE"
echo "crops done -> data/synthetic/lora_crops/ (train the LoRA next)"
