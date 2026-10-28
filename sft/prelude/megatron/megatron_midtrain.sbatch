#!/bin/bash
# Prelude 9B mid-training on Megatron-Bridge (megatron_midtrain.py) from the exported Levanter stages.
# Stage defaults: 16K stage1 (steps 0-607); set STAGE_CONTEXT/STAGE_START/END_ITER for later stages.
#SBATCH --job-name=prelude-mcore-midtrain
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp/%x_%j.log
set -euo pipefail
module load singularity/4.1.5

HERE=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp
SIF=/gpfs/scratch/ehpc532/ot/containers/mn5_sft_cu13.sif
: "${ROPE_THETA:?set ROPE_THETA (the production value is not decided yet)}"
: "${RUN_NAME:?set RUN_NAME}"
STAGE_CONTEXT=${STAGE_CONTEXT:-16384}
export MODEL=${MODEL:-/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/base_hf}
export STAGE_DIR=${STAGE_DIR:-/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_packed/$STAGE_CONTEXT}
export SEQ_LEN=$STAGE_CONTEXT GBS=${GBS:-$((1048576 / STAGE_CONTEXT))}
export OUT=$HERE/runs CKPT_DIR=${CKPT_DIR:-$HERE/runs/$RUN_NAME/checkpoints}
CACHE=$HERE/cache/$SLURM_JOB_ID; mkdir -p $CKPT_DIR $CACHE

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
ENVS=(MODEL OUT RUN_NAME CKPT_DIR STAGE_DIR SEQ_LEN GBS MBS TP CP ROPE_THETA END_ITER SCHEDULE_ITERS STAGE_START
      STAGE_START_SAMPLE RECOMPUTE GRAD_FP32 SAVE_INTERVAL EXIT_MINS LR WARMUP YARN_FACTOR YARN_ORIG CONCAT DECAY_START DECAY_ITERS)
ENV_FLAGS=()
for v in "${ENVS[@]}"; do [ -n "${!v:-}" ] && ENV_FLAGS+=(--env "$v=${!v}"); done
ENV_FLAGS+=(--env "MASTER_ADDR=$MASTER_ADDR" --env MASTER_PORT=29513 --env WORLD_SIZE=$SLURM_NTASKS
    --env PYTHONPATH=$HERE/stubs --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env NCCL_SOCKET_IFNAME=ib0
    --env NCCL_DEBUG=${NCCL_DEBUG:-WARN} --env CUDA_DEVICE_MAX_CONNECTIONS=1
    --env TRITON_CACHE_DIR=$CACHE/triton --env CUDA_CACHE_PATH=$CACHE/cuda
    --env TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor --env HF_HOME=$CACHE/hf)
# Optional allocator setting (e.g. expandable_segments:True against fragmentation OOMs at GBS 8 on 8 nodes).
[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && ENV_FLAGS+=(--env "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF")
# The image has no rdma-core, so NCCL fell back to TCP sockets (NET/Socket) between nodes. Bind the host's verbs
# + mlx5 provider (RHEL 9 glibc 2.34 < image 2.39) so NCCL uses InfiniBand. IB=0 restores the old behaviour.
if [ "${IB:-1}" = 1 ]; then
    RD=/gpfs/scratch/ehpc532/ot/pylibs/rdma_host VENV=/opt/venvs/sft-cu13/lib/python3.12/site-packages
    ENV_FLAGS+=(-B $RD/lib:/opt/rdma/lib -B $RD/libibverbs:/usr/lib64/libibverbs -B /etc/libibverbs.d:/etc/libibverbs.d
        --env LD_LIBRARY_PATH=$VENV/nvidia/nccl/lib:$VENV/nvidia/cudnn/lib:$VENV/nvidia/cublas/lib:$VENV/torch/lib:/opt/venvs/sft-cu13/lib:/.singularity.d/libs:/opt/rdma/lib)
fi

echo "nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR $(for v in "${ENVS[@]}"; do printf '%s=%s ' $v "${!v:-}"; done)"
srun --kill-on-bad-exit=1 bash -c '
    exec singularity exec --nv --cleanenv "$@" --env RANK=$SLURM_PROCID --env LOCAL_RANK=$SLURM_LOCALID \
        '"$SIF"' /opt/venvs/sft-cu13/bin/python '"$HERE"'/megatron_midtrain.py' bash "${ENV_FLAGS[@]}"
