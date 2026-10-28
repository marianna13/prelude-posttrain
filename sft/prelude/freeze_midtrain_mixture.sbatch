#!/bin/bash
#SBATCH --job-name=prelude-freeze-mixture
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=02:00:00

set -euo pipefail
export MIDTRAIN_PY=${MIDTRAIN_PY:-/p/project1/ccstdl/envs/marianna/lmeval/bin/python}
export MIDTRAIN_OUT=${MIDTRAIN_OUT:-/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step4}
export MIDTRAIN_MIX_OUT=${MIDTRAIN_MIX_OUT:-$MIDTRAIN_OUT/mixture_v1}
export MIDTRAIN_MIX_SCRIPT=${MIDTRAIN_MIX_SCRIPT:-$MIDTRAIN_OUT/tools/freeze_prelude_midtrain_mixture.py}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=none "$MIDTRAIN_PY" -I -u "$MIDTRAIN_MIX_SCRIPT" \
  --census "$MIDTRAIN_OUT" --out "$MIDTRAIN_MIX_OUT" --threads 64 ${MIDTRAIN_MIX_ARGS:-}
