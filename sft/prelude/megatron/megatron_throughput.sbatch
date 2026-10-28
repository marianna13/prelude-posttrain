#!/bin/bash
# Megatron-Bridge throughput smoke for Prelude 9B; compare with the Levanter A/B in ../ab_20261019.
#SBATCH --job-name=prelude-mcore-tp
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp/%x_%j.log
set -euo pipefail
module load singularity/4.1.5

HERE=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp
SIF=/gpfs/scratch/ehpc532/ot/containers/mn5_sft_cu13.sif
export MODEL=${MODEL:-/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/base_hf}
export OUT=$HERE/runs RUN_NAME=${RUN_NAME:-$SLURM_JOB_NAME-$SLURM_JOB_ID}
CACHE=$HERE/cache/$SLURM_JOB_ID; mkdir -p $OUT $CACHE

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
ENVS=(MODEL OUT RUN_NAME SEQ_LEN GBS MBS ITERS TP CP RECOMPUTE FSDP LOAD_WEIGHTS GRAD_FP32)
ENV_FLAGS=()
for v in "${ENVS[@]}"; do [ -n "${!v:-}" ] && ENV_FLAGS+=(--env "$v=${!v}"); done
ENV_FLAGS+=(--env "MASTER_ADDR=$MASTER_ADDR" --env MASTER_PORT=29512 --env WORLD_SIZE=$SLURM_NTASKS
    --env PYTHONPATH=$HERE/stubs --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env NCCL_SOCKET_IFNAME=ib0
    --env NCCL_DEBUG=${NCCL_DEBUG:-WARN} --env CUDA_DEVICE_MAX_CONNECTIONS=1
    --env TRITON_CACHE_DIR=$CACHE/triton --env CUDA_CACHE_PATH=$CACHE/cuda
    --env TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor --env HF_HOME=$CACHE/hf)

echo "nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR $(for v in "${ENVS[@]}"; do printf '%s=%s ' $v "${!v:-}"; done)"
srun --kill-on-bad-exit=1 bash -c '
    exec singularity exec --nv --cleanenv "$@" --env RANK=$SLURM_PROCID --env LOCAL_RANK=$SLURM_LOCALID \
        '"$SIF"' /opt/venvs/sft-cu13/bin/python '"$HERE"'/megatron_throughput.py' bash "${ENV_FLAGS[@]}"
