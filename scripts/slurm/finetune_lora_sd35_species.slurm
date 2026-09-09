#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_sd35_lora_sp
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G      # SD3.5 loads T5-XXL + 2 CLIP + transformer into host RAM; 64G OOM-kills
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Stage B (SPECIES LoRA) — fixes the wrong-species SD cutouts. The old
# finetune_lora_sd35.slurm fine-tunes on WHOLE field images with one generic
# prompt: it teaches scene STYLE, not what each weed looks like, so base SD3.5
# still draws generic flowers/leaves for these rare EPPO species. This LoRA
# instead trains on per-class REAL weed crops captioned by species
# (export_weed_crops.py), so SD learns the actual appearance and the cutout's
# species label becomes trustworthy.
#
# PREREQUISITE (CPU / login): export the captioned crop set first:
#   python scripts/export_weed_crops.py --max-per-class 400 --size 768
#   -> data/synthetic/lora_crops/{*.png, metadata.jsonl}
#
# Then:  sbatch scripts/slurm/finetune_lora_sd35_species.slurm
# After it finishes, re-render species-correct cutouts with the LoRA (GPU):
#   sbatch --export=ALL,POOL=sd35cut_lora,PER_CLASS=300,SEED=0,\
#LORA=data/synthetic/sd35/weights/lora_species scripts/slurm/gen_sd35_cutout.slurm
#   # then CPU paste + sweep GEN=sd35cut_lora (see gen_sd35_cutout_paste.slurm).

set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

MAX_STEPS=${MAX_STEPS:-2000}
RANK=${RANK:-16}
STEPS=${CAL_STEPS:-$MAX_STEPS}      # set CAL_STEPS=50 for a ~5-min calibration run
DATA=${DATA:-data/synthetic/lora_crops}
OUT=${OUT:-data/synthetic/sd35/weights/lora_species}

module load apps/2021
module load GCC/11.3.0
# NB: no system CUDA module — the pip torch wheel bundles its own (see the note
# in finetune_lora_sd35.slurm; a system CUDA shadows it -> CUDNN errors).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
which python && python --version

export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH:-8.0}"   # 8.0=A100
export HF_HUB_CACHE="$SCRATCH/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"
cd "$PROJECT"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"   # gated-model auth

[ -f "$DATA/metadata.jsonl" ] || { echo "missing $DATA/metadata.jsonl — run scripts/export_weed_crops.py first"; exit 1; }

# Diffusers SD3 DreamBooth-LoRA with a captioned image-folder (--dataset_name +
# --caption_column): every crop carries its own species caption, so ONE LoRA
# learns all 5 species (the 'single conditioned multi-class' generator).
# If your diffusers build's dreambooth-lora script lacks --caption_column,
# --instance_prompt is REQUIRED by argparse even with --caption_column (the
# per-image captions still win for training; it's just a mandatory fallback).
# If your build lacks --caption_column entirely, update diffusers or fall back
# to 5 per-class runs (--instance_data_dir per
# class + --instance_prompt "a photo of <species> weed").
accelerate launch third_party/diffusers/examples/dreambooth/train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
  --dataset_name "$DATA" --caption_column text \
  --instance_prompt "a top-down photo of a weed seedling on bare soil" \
  --resolution 768 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps "$STEPS" --rank "$RANK" \
  --checkpointing_steps 500 --resume_from_checkpoint latest \
  --mixed_precision fp16 --gradient_checkpointing \
  --seed 0 --output_dir "$OUT"

echo "species LoRA -> $OUT"
echo "next: sbatch --export=ALL,POOL=sd35cut_lora,PER_CLASS=300,LORA=$OUT scripts/slurm/gen_sd35_cutout.slurm"
