#!/bin/bash
#SBATCH --job-name=prelude-retained-bounded
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=06:00:00

set -euo pipefail
export MIDTRAIN_PY=${MIDTRAIN_PY:-/p/project1/ccstdl/envs/marianna/lmeval/bin/python}
export MIDTRAIN_OUT=${MIDTRAIN_OUT:-/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step4}
export MIDTRAIN_WORKERS=${MIDTRAIN_WORKERS:-64}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-2}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true
export USE_TORCH=0 USE_TF=0 USE_FLAX=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=none bash -c 'exec "$MIDTRAIN_PY" -I -u "$MIDTRAIN_OUT/tools/finish_prelude_retained_census.py" run \
  --out "$MIDTRAIN_OUT" --workers "$MIDTRAIN_WORKERS" --rank "$SLURM_PROCID" --world "$SLURM_NTASKS"'
"$MIDTRAIN_PY" -I -u "$MIDTRAIN_OUT/tools/finish_prelude_retained_census.py" merge --out "$MIDTRAIN_OUT"
