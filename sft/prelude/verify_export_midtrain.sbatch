#!/bin/bash
#SBATCH --job-name=prelude-export-audit
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=02:00:00
set -euo pipefail
export MIDTRAIN_PY=${MIDTRAIN_PY:-/p/project1/ccstdl/envs/marianna/lmeval/bin/python}
export MIDTRAIN_OUT=${MIDTRAIN_OUT:-/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step5}
export MIDTRAIN_VERIFY=${MIDTRAIN_VERIFY:-$MIDTRAIN_OUT/tools/verify_prelude_midtrain_export.py}
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 USE_TORCH=0 USE_TF=0 USE_FLAX=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MIDTRAIN_WORKERS=${MIDTRAIN_WORKERS:-16}
srun --cpu-bind=none bash -c 'exec "$MIDTRAIN_PY" -I -u "$MIDTRAIN_VERIFY" audit \
  --out "$MIDTRAIN_OUT" --workers "$MIDTRAIN_WORKERS" \
  --rank "$SLURM_PROCID" --world "$SLURM_NTASKS"'
"$MIDTRAIN_PY" -I -u "$MIDTRAIN_VERIFY" finalize --out "$MIDTRAIN_OUT"
