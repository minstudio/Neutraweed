#!/bin/bash
##---------------Start job description-------------------
#SBATCH --job-name=wg_comp_plain
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.out
#-----------------------End job description---------------
# Composite a pool with NO scene arguments, i.e. exactly the command that built
# sd35cut_lora_v2:
#
#   python scripts/gen_sd35_cutout.py generate --pool <P> --n <N> --seed 0
#
# composite_pool.slurm cannot be used for a v2 replication: its defaults
# (SCALE_MODE=prior, BG_MODE=tilematch, --class-weights, --n-instances) are all
# compositor features added after v2 was built, so it would vary four things at
# once. This script passes nothing, so today's argparse defaults apply — the same
# situation v2 was built under.
#
#   sbatch --export=ALL,POOL=sd35cut_lora_v2b,SRC_POOL=sd35cut_lora_v2,N=26220 \
#       scripts/slurm/composite_plain.slurm
#
# VALIDATE the result against v2 before training (see the echo at the end). If
# the argparse defaults have drifted since v2, v2b will not be a clean
# uniqueness-only comparison and the check will show it.
set -euo pipefail
PROJECT=/media/beegfs/home/x548/x548850/neutraweed
SCRATCH=/media/beegfs/home/x548/x548850/SCRATCH
module load apps/2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neutraweed
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TMPDIR"
cd "$PROJECT"

POOL=${POOL:?set POOL=<new pool name under data/synthetic/>}
SRC_POOL=${SRC_POOL:-}
N=${N:-2500}
SEED=${SEED:-0}
PREBUILD=${PREBUILD:-1}
DET=${DET:-yolo26}
RATIOS=${RATIOS:-2.0:2.0:0.25}
SEEDS_LIST=${SEEDS_LIST:-0 1 2}

# Reuse the donor's cutouts and backgrounds by symlink so the only difference
# between the pools is how many scenes were drawn from them.
if [ -n "$SRC_POOL" ]; then
  [ -d "data/synthetic/$SRC_POOL/cutouts" ] || { echo "no cutouts in SRC_POOL=$SRC_POOL"; exit 1; }
  mkdir -p "data/synthetic/$POOL"
  for sub in cutouts backgrounds; do
    if [ ! -e "data/synthetic/$POOL/$sub" ]; then
      ln -s "$(readlink -f "data/synthetic/$SRC_POOL/$sub")" "data/synthetic/$POOL/$sub"
      echo "[plain] linked $sub <- $SRC_POOL"
    fi
  done
fi

[ -d "data/synthetic/$POOL/cutouts" ] || { echo "no cutouts at data/synthetic/$POOL/cutouts"; exit 1; }
[ -d "data/synthetic/$POOL/backgrounds" ] || { echo "no backgrounds at data/synthetic/$POOL/backgrounds"; exit 1; }

echo "[plain] pool=$POOL n=$N seed=$SEED  (no scene args — argparse defaults)"
echo "[plain] cutouts per class:"
for d in data/synthetic/$POOL/cutouts/*/; do
  echo "  $(basename "$d")  $(find -L "$d" -name '*.png' | wc -l)"
done

python scripts/gen_sd35_cutout.py generate --pool "$POOL" --n "$N" --seed "$SEED"

echo "[plain] composited $(ls data/synthetic/$POOL/images | wc -l) scenes"

if [ "$PREBUILD" = "1" ]; then
  echo "[plain] prebuilding hybrid datasets ($DET, ratios=$RATIOS)"
  python scripts/run_sweep.py --generator "$POOL" --detector "$DET" \
      --ratios "$RATIOS" --seeds $SEEDS_LIST --no-baseline --build-only
fi

cat <<EOF

[plain] VALIDATE before training — v2b must match v2 on everything but count:

  python scripts/sharpness_spread.py --src v2=data/synthetic/$SRC_POOL \\
      --src v2b=data/synthetic/$POOL

  python - <<'PY'
import glob, numpy as np
for name in ("$SRC_POOL", "$POOL"):
    s = []
    for f in glob.glob(f"data/synthetic/{name}/labels/*.txt"):
        for line in open(f):
            p = line.split()
            if len(p) >= 5:
                s.append(np.sqrt(float(p[3]) * float(p[4])) * 1024)
    s = np.asarray(s)
    print(f"{name:28s} n={len(s):7d}  p10={np.percentile(s,10):6.1f}"
          f"  p50={np.percentile(s,50):6.1f}  p90={np.percentile(s,90):6.1f}")
PY

If the sharpness medians and the box-size percentiles agree within a few percent,
the defaults have not drifted and the comparison is uniqueness-only. Then:

  sbatch --array=0-2 --export=ALL,DETECTOR=$DET,GEN=$POOL,\\
SWEEP_RATIOS=$RATIOS,SEEDS="$SEEDS_LIST",EPOCHS=25 scripts/slurm/train_sweep_array.slurm

Compare against v2 r2.00 = 0.7527 (same volume, same cutouts, 16x duplicated).
EOF
