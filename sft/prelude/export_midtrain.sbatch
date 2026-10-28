#!/bin/bash
#SBATCH --job-name=prelude-token-export
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=04:00:00
set -euo pipefail
export MIDTRAIN_PY=${MIDTRAIN_PY:-/p/project1/ccstdl/envs/marianna/lmeval/bin/python}
export MIDTRAIN_BASE=${MIDTRAIN_BASE:-/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1}
export MIDTRAIN_OUT=${MIDTRAIN_OUT:-$MIDTRAIN_BASE/step5}
export MIDTRAIN_SCRIPT=${MIDTRAIN_SCRIPT:-$MIDTRAIN_OUT/tools/export_prelude_midtrain.py}
export MIDTRAIN_WORKERS=${MIDTRAIN_WORKERS:-32}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export USE_TORCH=0 USE_TF=0 USE_FLAX=0
if [[ ${MIDTRAIN_MODE:-run} == prepare ]]; then
    "$MIDTRAIN_PY" -I -u "$MIDTRAIN_SCRIPT" prepare --base "$MIDTRAIN_BASE" --out "$MIDTRAIN_OUT" --threads 32 ${MIDTRAIN_PREPARE_ARGS:-}
else
    srun --cpu-bind=none bash -c 'exec "$MIDTRAIN_PY" -I -u "$MIDTRAIN_SCRIPT" run \
      --out "$MIDTRAIN_OUT" --workers "$MIDTRAIN_WORKERS" \
      --rank "$SLURM_PROCID" --world "$SLURM_NTASKS" ${MIDTRAIN_EXPORT_ARGS:-}'
fi
