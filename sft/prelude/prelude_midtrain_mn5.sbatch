#!/bin/bash
#SBATCH --job-name=prelude-midtrain
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/logs/%x_%j.log

# Draft recipe: context quality targets failed in the smoke checkpoints.
# See rl/evals/prelude_midtrain/README.md before a production launch.
set -euo pipefail
module load singularity/4.1.5

ROOT=${MIDTRAIN_MN5_ROOT:-/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1}
RUNTIME=${MIDTRAIN_RUNTIME:-$ROOT/gates_20261018/runtime_production_v1}
IMAGE=${MIDTRAIN_IMAGE:-/gpfs/scratch/ehpc532/ot/containers/levanter_jax_26.09-py3_torch.sif}
PYLIBS=${MIDTRAIN_PYLIBS:-/gpfs/scratch/ehpc532/ot/marin-sft/pylibs}
MODEL=${MIDTRAIN_MODEL:-$ROOT/base_hf}
STAGE=${MIDTRAIN_STAGE:-stage1}
ROPE_THETA=${MIDTRAIN_ROPE_THETA:?Set the theta for the preregistered pilot; the former 6M default failed retrieval gates}
MODEL_MAX_SEQ_LEN=${MIDTRAIN_MODEL_MAX_SEQ_LEN:-65536}
TOKENS_PER_STEP=${MIDTRAIN_TOKENS_PER_STEP:-1048576}
FINITE_STAGE=${MIDTRAIN_FINITE_STAGE:-1}

case "$STAGE" in
    stage1) CONTEXT=16384; STAGE_START=0; DEFAULT_END_STEP=607 ;;
    stage2) CONTEXT=32768; STAGE_START=607; DEFAULT_END_STEP=1821 ;;
    stage3) CONTEXT=65536; STAGE_START=1821; DEFAULT_END_STEP=4419 ;;
    *) echo "unknown MIDTRAIN_STAGE: $STAGE" >&2; exit 2 ;;
esac
CONTEXT=${MIDTRAIN_CONTEXT:-$CONTEXT}
END_STEP=${MIDTRAIN_END_STEP:-$DEFAULT_END_STEP}
SCHEDULE_STEPS=${MIDTRAIN_SCHEDULE_STEPS:-4419}
CONTEXT_SHARD=${MIDTRAIN_CONTEXT_SHARD:-4}
if (( TOKENS_PER_STEP % CONTEXT != 0 )); then
    echo "TOKENS_PER_STEP must be divisible by CONTEXT" >&2
    exit 2
fi
if (( CONTEXT_SHARD < 1 || SLURM_NTASKS % CONTEXT_SHARD != 0 )); then
    echo "CONTEXT_SHARD must be positive and divide SLURM_NTASKS" >&2
    exit 2
fi
DATA_SHARD=$((SLURM_NTASKS / CONTEXT_SHARD))
if (( CONTEXT_SHARD > 4 || 4 % CONTEXT_SHARD != 0 )); then
    echo 'CONTEXT_SHARD must divide the four GPUs within each MN5 node' >&2
    exit 2
fi
LOCAL_DATA_SHARD=$((4 / CONTEXT_SHARD))
BATCH_SIZE=$((TOKENS_PER_STEP / CONTEXT))
CACHE=$ROOT/cache_v3/$CONTEXT
OUTPUT_DIR=$ROOT/runs/${MIDTRAIN_RUN_ID:-prelude-midtrain-$STAGE}
RUN_ID=${MIDTRAIN_RUN_ID:-prelude-midtrain-$STAGE}
LOGS_DIR=$OUTPUT_DIR/logs
RANK_LOGS=$LOGS_DIR/ranks_${SLURM_JOB_ID}
mkdir -p "$OUTPUT_DIR" "$LOGS_DIR" "$RANK_LOGS"
export RANK_LOGS

export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MIDTRAIN_XLA_MEM_FRACTION:-0.95}
export JAX_ENABLE_PGLE=false
export JAX_COMPILATION_CACHE_DIR=${MIDTRAIN_JAX_CACHE:-/gpfs/scratch/ehpc532/ot/marin-sft/jax-cache/prelude-midtrain}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true ${MIDTRAIN_XLA_EXTRA:-}"
export SINGULARITYENV_XLA_FLAGS="$XLA_FLAGS"
export PYTHONPATH=$RUNTIME/rl/sft/levanter:$PYLIBS/site:$PYLIBS/cutlass462/nvidia_cutlass_dsl/dsl_packages
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=$OUTPUT_DIR/wandb
mkdir -p "$WANDB_DIR"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=${MIDTRAIN_NCCL_IFACE:-ib0}

MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_NODE_I="$MASTER_NODE"
if [[ "$MASTER_NODE" == *"jrc"* ]]; then
    MASTER_NODE_I="${MASTER_NODE}i"
fi
MASTER_ADDR=$(nslookup "$MASTER_NODE_I" | awk '/^Address: / {print $2}' | tail -n1)
export MASTER_ADDR
export JAX_COORDINATOR_ADDRESS=${MASTER_ADDR}:19000
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

if [[ ! -f "$CACHE/train/.stats.json" ]]; then
    echo "missing native cache: $CACHE" >&2
    exit 3
fi

RESUME_ARGS=()
RESUME_FROM=${MIDTRAIN_RESUME_FROM:-}
shopt -s nullglob
OWN_CHECKPOINTS=("$OUTPUT_DIR"/checkpoints/step-*/manifest.json)
shopt -u nullglob
if (( ${#OWN_CHECKPOINTS[@]} )); then
    RESUME_FROM=$OUTPUT_DIR/checkpoints
    echo "continuing this stage from its own latest valid checkpoint: $RESUME_FROM"
fi
if [[ "$STAGE" != stage1 && -z "$RESUME_FROM" ]]; then
    echo "stage2/stage3 require MIDTRAIN_RESUME_FROM with the previous stage optimizer state" >&2
    exit 2
fi
if [[ -n "$RESUME_FROM" ]]; then
    RESUME_ARGS=(--resume-from "$RESUME_FROM")
fi

if [[ "$FINITE_STAGE" == 1 ]]; then
    RESUME_ARGS+=(--finite-stage --stage-start-step "$STAGE_START")
fi

MESH=$(printf '{"axes":{"data":%d,"replica":1,"model":1,"context":%d,"expert":1},"dcn_axes":{"replica_dcn":1,"data_dcn":%d},"compute_mapping":{"batch":["data_dcn","data"],"position":"context"},"param_mapping":{"embed":"context","mlp":["data_dcn","data"],"heads":["data_dcn","data"],"kv_head":["data_dcn","data"],"vocab":["data_dcn","data"]}}' "$LOCAL_DATA_SHARD" "$CONTEXT_SHARD" "$SLURM_NNODES")
EXP_SCRIPT=$RUNTIME/rl/sft/levanter/scripts_grug/sft_main.py
EXP_SCRIPT_DIR=$RUNTIME/rl/sft/levanter
RUN=(singularity exec --nv --bind "$ROOT:$ROOT,$PYLIBS:/pylibs" --pwd /marin "$IMAGE" env
    PYTHONPATH="$EXP_SCRIPT_DIR:/pylibs/site:/pylibs/cutlass462/nvidia_cutlass_dsl/dsl_packages"
    JAX_PLATFORM_NAME=gpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 WANDB_MODE=offline
    JAX_COORDINATOR_ADDRESS="$JAX_COORDINATOR_ADDRESS" MASTER_ADDR="$MASTER_ADDR" /marin/.venv/bin/python)

echo "stage=$STAGE context=$CONTEXT batch=$BATCH_SIZE end_step=$END_STEP schedule_steps=$SCHEDULE_STEPS"
echo "cache=$CACHE output=$OUTPUT_DIR"
echo "per-rank logs: $RANK_LOGS"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NTASKS" --ntasks-per-node=4 \
    bash -c '[ "$SLURM_PROCID" -eq 0 ] || exec >"$RANK_LOGS/rank_${SLURM_PROCID}.log" 2>&1; exec "$@"' \
    bash "${RUN[@]}" "$EXP_SCRIPT" train \
    --model-path "$MODEL" --model-type qwen3 --format midtrain \
    --cache-path "$CACHE" --output-path "$OUTPUT_DIR" --run-id "$RUN_ID" \
    --seq-len "$CONTEXT" --model-max-seq-len "$MODEL_MAX_SEQ_LEN" --rope-theta "$ROPE_THETA" \
    --disable-sliding-window --attention-backend fa4 --pack \
    --batch-size "$BATCH_SIZE" --per-device-parallelism "${MIDTRAIN_PER_DEVICE_PARALLELISM:-1}" --steps "$END_STEP" --schedule-steps "$SCHEDULE_STEPS" \
    --lr 0.00001 --warmup 100 --checkpoint-minutes 30 \
    --mesh "$MESH" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
