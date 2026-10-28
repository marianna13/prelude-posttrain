#!/bin/bash
#SBATCH --job-name=prelude-cache
#SBATCH --account=ehpc1048
#SBATCH --partition=gpp
#SBATCH --qos=gp_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-2
set -euo pipefail
module load singularity/4.1.5
ROOT=${MIDTRAIN_MN5_ROOT:-/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1}
BUNDLE=${MIDTRAIN_HANDOFF_ROOT:-$ROOT/handoff}
CACHE=${MIDTRAIN_CACHE_ROOT:-$ROOT/cache}
SMOKE=${MIDTRAIN_LOADER_SMOKE_ROOT:-$ROOT/loader_smoke}
IMG=${MIDTRAIN_IMAGE:-/gpfs/scratch/ehpc532/ot/containers/levanter_jax_26.09-py3_torch.sif}
contexts=(16384 32768 65536)
context=${contexts[$SLURM_ARRAY_TASK_ID]}
SCRIPT=$BUNDLE/runtime/rl/sft/prelude/prepare_midtrain_levanter.py
RUN=(singularity exec --bind "$ROOT:$ROOT" "$IMG" env JAX_PLATFORMS=cpu OMP_NUM_THREADS=2
     OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
     HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 /marin/.venv/bin/python)
"${RUN[@]}" "$SCRIPT" build --release "$ROOT/release" --context "$context" --out "$CACHE/$context"
"${RUN[@]}" "$SCRIPT" smoke --release "$ROOT/release" --context "$context" --out "$SMOKE/$context"
