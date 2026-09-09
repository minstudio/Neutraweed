#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_sd35_cn
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G      # models load fine in 128G; the OOM was the 1000-caption T5
#                       # precompute, fixed by --dataset_preprocess_batch_size below.
#                       # (--mem=0 needs a fully-empty node -> multi-day queue; avoid.)
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Stage B (3c) — train the layout ControlNet on the (layout -> image) pairs from
# `gen_sd35.py prep`. Conditioning = the layout mask, so labels are free
# Submit with:
#     sbatch scripts/slurm/train_controlnet_sd35.slurm
# Edit PROJECT/SCRATCH and the conda env for your account.

set -euo pipefail

# --- paths (CONFIRM SCRATCH for your account) ---
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

# --- training length ---
MAX_STEPS=20000         # longer train for layout adherence; resumes from the 5k checkpoint
# CAL_STEPS=50          # uncomment for a ~5-min CALIBRATION run: read the it/s the
                        # trainer prints, then re-comment and set MAX_STEPS/walltime.
STEPS=${CAL_STEPS:-$MAX_STEPS}

# Do NOT load the system CUDA module: pip torch bundles its own CUDA + cuDNN; a
# system CUDA module shadows them -> CUDNN_STATUS_NOT_INITIALIZED.
module load apps/2021
module load GCC/11.3.0
# module load CUDA/11.8.0   # intentionally omitted (pure-PyTorch job)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
which python && python --version

export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH:-8.0}"   # 8.0=A100, 8.9=L40S, 7.0=V100
export HF_HUB_CACHE="$SCRATCH/hf_cache"     # big model cache on BeeGFS (token stays in private home)
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"

cd "$PROJECT"

accelerate launch third_party/diffusers/examples/controlnet/train_controlnet_sd3.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
  --train_data_dir data/synthetic/sd35/controlnet_pairs/train \
  --image_column image --conditioning_image_column conditioning_image --caption_column text \
  --resolution 1024 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 --max_train_steps "$STEPS" \
  --dataset_preprocess_batch_size 8 \
  --checkpointing_steps 1000 --checkpoints_total_limit 3 --resume_from_checkpoint latest \
  --mixed_precision fp16 --gradient_checkpointing \
  --output_dir data/synthetic/sd35/weights/controlnet

echo "ControlNet trained -> data/synthetic/sd35/weights/controlnet"
