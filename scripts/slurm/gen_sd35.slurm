#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_sd35_gen
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Stage C — generate the SD3.5 + ControlNet synthetic pool.
# Magerit3 (CESVIMA/UPM). Submit with:
#     sbatch scripts/slurm/gen_sd35.slurm
# Set N / SEED / CONTROL below. Run the 'prep' step once (on a login node or a
# short job) before the first generate.

set -euo pipefail

# --- paths (CONFIRM SCRATCH for your account) ---
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

# --- run parameters (override at submit, e.g. --export=ALL,GEN_N=8) ---
N=${GEN_N:-2000}
SEED=${GEN_SEED:-0}
CONTROL=${GEN_CONTROL:-seg}   # layout-mask conditioning -> free labels
CN_SCALE=${GEN_CN_SCALE:-1.0} # raise (1.5-3.0) to force weeds onto the conditioned boxes

# Do NOT load the system CUDA module: pip torch bundles its own CUDA + cuDNN; a
# system CUDA module shadows them -> CUDNN_STATUS_NOT_INITIALIZED.
module load apps/2021
module load GCC/11.3.0
# module load CUDA/11.8.0   # intentionally omitted (pure-PyTorch job)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
which python && python --version

export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH:-8.0}"   # 8.0=A100, 8.9=L40S, 7.0=V100
export HF_HUB_CACHE="$SCRATCH/hf_cache"     # model cache on BeeGFS; token stays in private home
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"

cd "$PROJECT"

python scripts/gen_sd35.py generate --n "$N" --control "$CONTROL" --seed "$SEED" --cn-scale "$CN_SCALE"

echo "Generated $N images (seed $SEED) -> data/synthetic/sd35/images"
