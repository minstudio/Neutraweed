#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_sd35_lora
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G      # SD3.5 loads T5-XXL + 2 CLIP + transformer into host RAM; 64G OOM-kills
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Stage B — domain LoRA fine-tune of SD3.5 on the real TRAIN split.
# Magerit3 (CESVIMA/UPM): partition standard-gpu, 1x A100. Submit with:
#     sbatch scripts/slurm/finetune_lora_sd35.slurm
# Adjust PROJECT and the conda env name to your account first.

set -euo pipefail

# --- paths (CONFIRM SCRATCH for your account) ---
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

# --- training length ---
MAX_STEPS=2000          # full run
# CAL_STEPS=50          # uncomment for a ~5-min CALIBRATION run: read the it/s the
                        # trainer prints, then re-comment and set MAX_STEPS/walltime.
STEPS=${CAL_STEPS:-$MAX_STEPS}

# --- modules ---
# Do NOT load the system CUDA module: the pip torch wheel bundles its own CUDA +
# cuDNN, and a system CUDA module shadows them -> CUDNN_STATUS_NOT_INITIALIZED.
module load apps/2021
module load GCC/11.3.0
# module load CUDA/11.8.0   # intentionally omitted (pure-PyTorch job, no nvcc needed)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
which python && python --version

# A100 is sm_80. Big model cache + temp on BeeGFS SCRATCH (8 GB / is tiny);
# the HF token stays in your private home, NOT on shared SCRATCH.
export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH:-8.0}"   # 8.0=A100, 8.9=L40S, 7.0=V100
export HF_HUB_CACHE="$SCRATCH/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"

cd "$PROJECT"

# Diffusers SD3 DreamBooth-LoRA example (clone diffusers under $PROJECT/third_party
# or pip-install it).
accelerate launch third_party/diffusers/examples/dreambooth/train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
  --instance_data_dir data/real/images/train \
  --instance_prompt "a top-down photo of a tomato field with weed seedlings" \
  --resolution 1024 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --max_train_steps "$STEPS" \
  --checkpointing_steps 500 --resume_from_checkpoint latest \
  --mixed_precision fp16 --gradient_checkpointing \
  --output_dir data/synthetic/sd35/weights/lora

echo "LoRA fine-tune done -> data/synthetic/sd35/weights/lora"
