#!/bin/bash
#SBATCH --job-name=prelude-midtrain-census
#SBATCH --account=cstdl
#SBATCH --partition=dc-cpu-bigmem
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=06:00:00

set -euo pipefail
REPO=/p/project1/laionize/marianna/dc_agent/monorepo
PY=${PY:-/p/project1/ccstdl/envs/marianna/lmeval/bin/python}
OUT=${OUT:-/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step1/census_v2}
WORKERS=${WORKERS:-4}
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-32} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
SCRIPT="$OUT/tools/prelude_midtrain_census.py"
test -f "$SCRIPT"
srun --cpu-bind=none "$PY" -I "$SCRIPT" run --out "$OUT" --workers "$WORKERS"
"$PY" -I "$SCRIPT" summarize --out "$OUT"
