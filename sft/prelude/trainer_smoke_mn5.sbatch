#!/bin/bash
#SBATCH --job-name=prelude-9b-smoke
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
set -euo pipefail
module load singularity/4.1.5
ROOT=${MIDTRAIN_MN5_ROOT:-/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1}
BUNDLE=${MIDTRAIN_HANDOFF_ROOT:-$ROOT/handoff_v3}
IMG=${MIDTRAIN_IMAGE:-/gpfs/scratch/ehpc532/ot/containers/levanter_jax_26.09-py3_torch.sif}
PYLIBS=${MIDTRAIN_PYLIBS:-/gpfs/scratch/ehpc532/ot/marin-sft/pylibs}
EXP=$BUNDLE/runtime/rl/sft/levanter
RUN=(singularity exec --nv --bind "$ROOT:$ROOT,$PYLIBS:/pylibs" --pwd /marin "$IMG" env
     OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
     HF_HUB_OFFLINE=1 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
     PYTHONPATH="$EXP:/pylibs/site:/pylibs/cutlass462/nvidia_cutlass_dsl/dsl_packages"
     /marin/.venv/bin/python)
# One process owns all four H100s; data=4 uses the native embed->data parameter
# sharding for FSDP. This uses only the synthetic mixed-format correctness cache,
# not any of the frozen production allocations or the long-context curriculum.
srun "${RUN[@]}" -c 'import jax; print(jax.devices(), flush=True); assert len(jax.devices()) == 4'
for steps in 2 3; do
    srun "${RUN[@]}" "$EXP/scripts_grug/sft_main.py" train \
        --model-path "$ROOT/base_hf" --model-type qwen3 --format midtrain \
        --cache-path "$BUNDLE/canary_cache" --output-path "$ROOT/smoke_9b_gpu_fa4" \
        --run-id prelude-9b-fa4-loader-smoke --seq-len 128 --batch-size 4 \
        --model-max-seq-len 65536 --rope-theta 6000000 --disable-sliding-window \
        --attention-backend fa4 \
        --steps "$steps" --lr 0.00001 --warmup 0 --checkpoint-minutes 1 \
        --mesh '{"axes":{"data":4,"replica":1,"model":1,"context":1,"expert":1},"compute_mapping":{"batch":["data"]}}'
done
