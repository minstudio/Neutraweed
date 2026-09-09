#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_lora_sp
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --output=slurm_%x_%A_%a.out
#-----------------------End job description---------------
# One LoRA per species, plus a control.
#
# The POROL result confounds two changes: a dedicated LoRA, AND size-stratified
# crops with per-size captions. This array separates them.
#
#   --array=0-4   one LoRA per species (SOLNI POROL SETVE CYPRO ECHCG)
#   --array=5     CONTROL: one multi-class LoRA on the SAME stratified,
#                 size-captioned crops. If this matches the five separate ones,
#                 the captions did the work and the single conditioned
#                 multi-class generator stays intact.
#   --array=0-5   both
#
#   sbatch --array=0-5%2 scripts/slurm/lora_per_species.slurm
#
# Each task exports its own crops first (CPU, a few minutes) so the job is
# self-contained. Outputs:
#   data/synthetic/lora_crops_<CLS>/        crops + metadata.jsonl
#   data/synthetic/sd35/weights/lora_<cls>  the LoRA
#   ..._multi_strat                          the control LoRA
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

CLASSES=(SOLNI POROL SETVE CYPRO ECHCG)
MAX_STEPS=${MAX_STEPS:-2000}
RANK=${RANK:-16}
STEPS=${CAL_STEPS:-$MAX_STEPS}
PER_CLASS=${PER_CLASS:-600}
STRATA=${STRATA:-4}
SIZE=${SIZE:-768}

module load apps/2021
module load GCC/11.3.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH:-8.0}"
export HF_HUB_CACHE="$SCRATCH/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"
cd "$PROJECT"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"

i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 5 ]; then
  CLS=${CLASSES[$i]}
  DATA=data/synthetic/lora_crops_${CLS,,}
  OUT=data/synthetic/sd35/weights/lora_${CLS,,}
  ONLY="--classes $CLS"
  PROMPT="a top-down close-up photo of a ${CLS} weed on bare soil"
  echo "[lora] per-species: $CLS"
else
  CLS=MULTI
  DATA=data/synthetic/lora_crops_multi_strat
  OUT=data/synthetic/sd35/weights/lora_multi_strat
  ONLY=""
  PROMPT="a top-down photo of a weed seedling on bare soil"
  echo "[lora] CONTROL: multi-class on stratified, size-captioned crops"
fi

# Stratified sampling across each class's own box-size range, with the growth
# stage phrase matched to the size bucket. Without this the crop set is
# dominated by whatever appears first, which for POROL meant seedlings only.
python scripts/export_weed_crops.py $ONLY \
  --size-strata "$STRATA" --size-captions \
  --max-per-class "$PER_CLASS" --size "$SIZE" \
  --out "$(basename "$DATA")"

[ -f "$DATA/metadata.jsonl" ] || { echo "export produced no metadata.jsonl"; exit 1; }
echo "[lora] $(wc -l < "$DATA/metadata.jsonl") crops -> $OUT"

accelerate launch third_party/diffusers/examples/dreambooth/train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
  --dataset_name "$DATA" --caption_column text \
  --instance_prompt "$PROMPT" \
  --resolution "$SIZE" --train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps "$STEPS" --rank "$RANK" \
  --checkpointing_steps 500 --resume_from_checkpoint latest \
  --mixed_precision fp16 --gradient_checkpointing \
  --seed 0 --output_dir "$OUT"

echo "[lora] $CLS done -> $OUT"
echo "[lora] next: sbatch --array=0-4 scripts/slurm/gen_per_species.slurm"
