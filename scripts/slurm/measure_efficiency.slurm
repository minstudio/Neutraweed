#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_eff
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Params / FLOPs / latency per detector. Latency/params are architecture-only,
# so ANY checkpoint of a given detector works -- adjust the paths below to a
# checkpoint that exists (e.g. a hybrid run) if a real-only one is missing.
#   sbatch scripts/slurm/measure_efficiency.slurm
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"
python scripts/measure_efficiency.py \
  --runs yolo11=results/checkpoints/realonly_yolo11_seed0/weights/best.pt \
         yolo26=results/checkpoints/realonly_yolo26_seed0/weights/best.pt \
         rtdetrv2=results/checkpoints/realonly_rtdetrv2_seed0/weights/best.pt \
  --data data/datasets/realonly_yolo11_real/dataset.yaml --imgsz 1536
echo "[eff] done -> results/tables/efficiency.csv"
