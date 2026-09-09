#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_flux_lora
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Species LoRA for FLUX.1-dev, reusing the SAME captioned crop set built for the
# SD LoRA (export_weed_crops.py -> data/synthetic/lora_crops). Fixes FLUX's
# wrong-species renders exactly like the SD species LoRA did. FLUX is 12B: needs
# an A100 + gradient checkpointing; FLUX.1-dev is gated on HF (accept license +
# token). PREREQUISITE: data/synthetic/lora_crops/metadata.jsonl already exists.
#
#   sbatch scripts/slurm/finetune_lora_flux_species.slurm
#   # calibration first: --export=ALL,CAL_STEPS=50
# After it finishes, render species-correct FLUX cutouts:
#   sbatch --export=ALL,BACKEND=flux,POOL=fluxcut_lora,PER_CLASS=300,\
#LORA=data/synthetic/flux/weights/lora_species scripts/slurm/gen_sd35_cutout.slurm
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
MAX_STEPS=2000
STEPS=${CAL_STEPS:-$MAX_STEPS}
DATA=${DATA:-data/synthetic/lora_crops}
OUT=${OUT:-data/synthetic/flux/weights/lora_species}
module load apps/2021
module load GCC/11.3.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH:-8.0}"
export HF_HUB_CACHE="$SCRATCH/hf_cache"; export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$HF_HUB_CACHE" "$TMPDIR"
cd "$PROJECT"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"   # gated-model auth
[ -f "$DATA/metadata.jsonl" ] || { echo "missing $DATA/metadata.jsonl — run export_weed_crops.py first"; exit 1; }
# diffusers FLUX dreambooth-lora example (clone diffusers under third_party or pip-install).
accelerate launch third_party/diffusers/examples/dreambooth/train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
  --dataset_name "$DATA" --caption_column text \
  --instance_prompt "a top-down photo of a weed seedling on bare soil" \
  --resolution 768 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps "$STEPS" --rank 16 --guidance_scale 1.0 \
  --checkpointing_steps 500 --resume_from_checkpoint latest \
  --mixed_precision bf16 --gradient_checkpointing \
  --seed 0 --output_dir "$OUT"
echo "FLUX species LoRA -> $OUT"
