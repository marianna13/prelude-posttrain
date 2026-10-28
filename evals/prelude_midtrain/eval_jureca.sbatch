#!/bin/bash
#SBATCH --job-name=prelude-gates-eval
#SBATCH --account=synthlaion
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
set -euo pipefail
W=${MIDTRAIN_REPO:-/p/project1/laionize/marianna/dc_agent/prelude_gates_work}
G=/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/gates_20261018
CONDA_ENV=/p/project1/ccstdl/envs/marianna/py3.12
set +u
source /p/project1/ccstdl/nezhurina1/miniconda/miniconda/bin/activate "$CONDA_ENV"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v "^${CONDA_ENV}/lib" | tr '\n' ':' | sed 's/:$//')
source /p/project1/ccstdl/envs/marianna/venv/bin/activate
set -u
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=$G/cache/vllm TRITON_CACHE_DIR=$G/cache/triton XDG_CACHE_HOME=$G/cache/xdg
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_NET_GDR_LEVEL=0 NCCL_NET=Socket FI_PROVIDER=tcp
RESUME=()
if [[ ${GATE_RESUME:-0} == 1 ]]; then RESUME=(--resume); fi
exec python "$W/rl/evals/prelude_midtrain/infer_gates.py" \
    --model "${GATE_MODEL:-/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step2/base_hf}" \
    --inputs "$GATE_INPUTS" --out "$G/results/${GATE_RUN_ID:?}" --context "${GATE_CONTEXT:-4096}" "${RESUME[@]}"
