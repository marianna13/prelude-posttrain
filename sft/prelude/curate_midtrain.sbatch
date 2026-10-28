#!/bin/bash
#SBATCH --job-name=prelude-midtrain-curate
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=06:00:00

set -euo pipefail
export MIDTRAIN_REPO=${MIDTRAIN_REPO:-/p/project1/laionize/marianna/dc_agent/monorepo}
export MIDTRAIN_PY=${MIDTRAIN_PY:-/p/project1/ccstdl/envs/marianna/lmeval/bin/python}
export MIDTRAIN_OUT=${MIDTRAIN_OUT:-/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step3}
export MIDTRAIN_SCRIPT=${MIDTRAIN_SCRIPT:-$MIDTRAIN_REPO/rl/scripts/data/curate_prelude_midtrain.py}
export MIDTRAIN_STAGE=${MIDTRAIN_STAGE:-scan}
export MIDTRAIN_WORKERS=${MIDTRAIN_WORKERS:-32}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
srun --cpu-bind=none bash -c 'exec "$MIDTRAIN_PY" -I -u "$MIDTRAIN_SCRIPT" run \
  --out "$MIDTRAIN_OUT" --stage "$MIDTRAIN_STAGE" --workers "$MIDTRAIN_WORKERS" \
  --rank "$SLURM_PROCID" --world "$SLURM_NTASKS"'
