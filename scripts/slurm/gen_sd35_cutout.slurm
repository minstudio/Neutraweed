#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_sd35cut
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Phase 1 of the SD-cutout arm (GPU): SD3.5 renders single isolated weeds and
# mattes them into cutouts/<CLS>/. Backgrounds + composite paste are CPU and run
# on the login node afterwards (see below). SD needs the GPU — never the login
# node.
#
#   sbatch --export=ALL,POOL=sd35cut,PER_CLASS=300,SEED=0 scripts/slurm/gen_sd35_cutout.slurm
#   # domain LoRA (if fine-tuned): add ,LORA=data/synthetic/sd35/weights/lora
#
# Then, on the login node (CPU):
#   python scripts/gen_sd35_cutout.py backgrounds --pool sd35cut --from-pool composite2022
#   python scripts/gen_sd35_cutout.py generate    --pool sd35cut --n 2500 --seed 0
#   # QC before training:
#   python scripts/gen_sd35_cutout.py cutouts ... --keep-rejects   # inspect rejects/
#
# Sweep it like any generator once the pool exists:
#   sbatch --export=ALL,DETECTOR=yolo11,GEN=sd35cut,SWEEP_RATIOS=0.25:2.0:0.25,SWEEP_SEEDS="0 1 2" \
#       scripts/slurm/train_sweep.slurm

set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH

module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
export HF_HOME="$SCRATCH/hf"; mkdir -p "$HF_HOME"
cd "$PROJECT"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"   # gated-model auth

BACKEND=${BACKEND:-sd35}          # sd35 | flux
POOL=${POOL:-}                    # default set by backend below
PER_CLASS=${PER_CLASS:-300}
SEED=${SEED:-0}
MATTE=${MATTE:-exg}
if [ "$BACKEND" = flux ]; then
  POOL=${POOL:-fluxcut}; GUIDANCE=${GUIDANCE:-3.5}
  [ -n "${CPU_OFFLOAD:-}" ] && EXTRA_B="--cpu-offload" || EXTRA_B=""
else
  POOL=${POOL:-sd35cut}; GUIDANCE=${GUIDANCE:-4.5}; EXTRA_B=""
fi
EXTRA=""
[ -n "${LORA:-}" ] && EXTRA="$EXTRA --lora $LORA"
[ -n "${MIN_SHARP:-}" ] && EXTRA="$EXTRA --min-sharpness $MIN_SHARP"
[ -n "${BUFFER:-}" ] && EXTRA="$EXTRA --buffer-frac $BUFFER"
[ -n "${KEEP_REJECTS:-}" ] && EXTRA="$EXTRA --keep-rejects"
[ -n "${MIN_CLIPIQA:-}" ] && EXTRA="$EXTRA --min-clipiqa $MIN_CLIPIQA"
[ -n "${RENDER_SIZE:-}" ] && EXTRA="$EXTRA --render-size $RENDER_SIZE"
[ -n "${CLASSES:-}" ] && EXTRA="$EXTRA --classes $CLASSES"
[ -n "${STAGES:-}" ] && EXTRA="$EXTRA --stages"

pip install -q "diffusers>=0.31" transformers accelerate peft sentencepiece protobuf scipy rembg torchmetrics piq 2>/dev/null || \
pip install -q "diffusers>=0.31" transformers accelerate peft sentencepiece protobuf scipy torchmetrics piq

python -c "import torch; print('device=cuda' if torch.cuda.is_available() else 'device=cpu (WRONG - not on GPU)')"

python scripts/gen_sd35_cutout.py cutouts --backend "$BACKEND" --pool "$POOL" \
    --per-class "$PER_CLASS" --seed "$SEED" --matte "$MATTE" --guidance "$GUIDANCE" \
    $EXTRA_B $EXTRA
echo "[sd35cut] cutouts done -> data/synthetic/$POOL/cutouts/. Next (CPU/login):"
echo "  python scripts/gen_sd35_cutout.py backgrounds --pool $POOL --from-pool composite2022"
echo "  python scripts/gen_sd35_cutout.py generate    --pool $POOL --n 2500 --seed 0"
