#!/bin/bash
#SBATCH --job-name=prelude-midtrain-reduce
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=06:00:00

set -euo pipefail
export MIDTRAIN_REPO=${MIDTRAIN_REPO:-/p/project1/laionize/marianna/dc_agent/monorepo}
export MIDTRAIN_PY=${MIDTRAIN_PY:-/p/project1/ccstdl/envs/marianna/lmeval/bin/python}
export MIDTRAIN_OUT=${MIDTRAIN_OUT:-/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step3}
export MIDTRAIN_OPERATION=${MIDTRAIN_OPERATION:-materialize}
export MIDTRAIN_WORKERS=${MIDTRAIN_WORKERS:-32}
export MIDTRAIN_MEMORY=${MIDTRAIN_MEMORY:-256GB}
export MIDTRAIN_THREADS=${MIDTRAIN_THREADS:-64}
export MIDTRAIN_EXCLUSIONS=${MIDTRAIN_EXCLUSIONS:-$MIDTRAIN_OUT/benchmark_exclusions.json}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

case "$MIDTRAIN_OPERATION" in
  materialize)
    srun --cpu-bind=none bash -c 'exec "$MIDTRAIN_PY" -I -u \
      "$MIDTRAIN_REPO/rl/scripts/data/reduce_prelude_midtrain.py" materialize \
      --out "$MIDTRAIN_OUT" --workers "$MIDTRAIN_WORKERS" \
      --rank "$SLURM_PROCID" --world "$SLURM_NTASKS"'
    ;;
  summarize|decide|finalize)
    if [[ "$SLURM_NTASKS" != 1 ]]; then
      echo "Global reduction/finalization requires exactly one Slurm task" >&2
      exit 2
    fi
    if [[ "$MIDTRAIN_OPERATION" == finalize ]]; then
      srun --cpu-bind=none "$MIDTRAIN_PY" -I -u \
        "$MIDTRAIN_REPO/rl/scripts/data/finalize_prelude_midtrain_step3.py" \
        --out "$MIDTRAIN_OUT" --memory "$MIDTRAIN_MEMORY" --threads "$MIDTRAIN_THREADS"
    elif [[ "$MIDTRAIN_OPERATION" == summarize ]]; then
      srun --cpu-bind=none "$MIDTRAIN_PY" -I -u \
        "$MIDTRAIN_REPO/rl/scripts/data/reduce_prelude_midtrain.py" summarize \
        --out "$MIDTRAIN_OUT" --verify-bytes --memory "$MIDTRAIN_MEMORY" --threads "$MIDTRAIN_THREADS"
    else
      srun --cpu-bind=none "$MIDTRAIN_PY" -I -u \
        "$MIDTRAIN_REPO/rl/scripts/data/reduce_prelude_midtrain.py" decide \
        --out "$MIDTRAIN_OUT" --exclusions "$MIDTRAIN_EXCLUSIONS" \
        --memory "$MIDTRAIN_MEMORY" --threads "$MIDTRAIN_THREADS"
    fi
    ;;
  *) echo "Unknown MIDTRAIN_OPERATION: $MIDTRAIN_OPERATION" >&2; exit 2 ;;
esac
