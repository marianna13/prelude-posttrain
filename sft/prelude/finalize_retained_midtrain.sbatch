#!/bin/bash
#SBATCH --job-name=prelude-retained-finalize
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
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export USE_TORCH=0 USE_TF=0 USE_FLAX=0
srun --cpu-bind=none "$MIDTRAIN_PY" -I -u "$MIDTRAIN_OUT/tools/finalize_prelude_midtrain_census.py" \
  --out "$MIDTRAIN_OUT" --threads 64
