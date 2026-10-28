#!/bin/bash
#SBATCH --job-name=prelude-loader-cpu
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=01:00:00
set -euo pipefail
ROOT=${MIDTRAIN_REPO:-/p/project1/laionize/marianna/dc_agent/monorepo}
OUT=${MIDTRAIN_OUT:-/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step5}
IMG=${MIDTRAIN_IMAGE:-/p/data1/mmlaion/container/levanter_jax_26.09-py3_torch.sif}
RUN=(apptainer exec --bind /p:/p "$IMG" env JAX_PLATFORMS=cpu OMP_NUM_THREADS=2
     OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
     HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 /marin/.venv/bin/python)
if [[ ${MIDTRAIN_MODE:-smoke} == tests ]]; then
    "${RUN[@]}" "$ROOT/rl/sft/prelude/test_midtrain_loader.py"
elif [[ ${MIDTRAIN_MODE:-smoke} == training ]]; then
    "${RUN[@]}" "$ROOT/rl/sft/prelude/smoke_midtrain_training.py" \
        --step2 "$(dirname "$OUT")/step2" --out "${MIDTRAIN_TRAIN_SMOKE_OUT:-$OUT/tiny_training}"
else
    for context in 16384 32768 65536; do
        "${RUN[@]}" "$ROOT/rl/sft/prelude/prepare_midtrain_levanter.py" smoke \
            --release "$OUT/release" --context "$context" --out "${MIDTRAIN_LOADER_SMOKE_OUT:-$OUT/loader_smoke}/$context"
    done
fi
