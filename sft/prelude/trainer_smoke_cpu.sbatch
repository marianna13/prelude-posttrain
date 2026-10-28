#!/bin/bash
#SBATCH --job-name=prelude-trainer-cpu
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
set -euo pipefail
ROOT=${MIDTRAIN_REPO:-/p/project1/laionize/marianna/dc_agent/monorepo}
OUT=${MIDTRAIN_OUT:-/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step5}
IMG=${MIDTRAIN_IMAGE:-/p/data1/mmlaion/container/levanter_jax_26.09-py3_torch.sif}
exec apptainer exec --bind /p:/p --pwd /marin "$IMG" env JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
    WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/rl/sft/levanter" \
    /marin/.venv/bin/python "$ROOT/rl/sft/levanter/scripts_grug/sft_main.py" train \
    --model-path "$OUT/tiny_training_v2/source_hf" --model-type qwen3 --format midtrain \
    --cache-path "$OUT/tiny_training_v2/cache" --output-path "${MIDTRAIN_TRAINER_OUT:-$OUT/trainer_cpu}" \
    --model-max-seq-len 128 --rope-theta 6000000 --disable-sliding-window --attention-backend vanilla \
    --seq-len 128 --batch-size 1 --steps "${MIDTRAIN_TRAINER_STEPS:-2}" --lr 0.0001 --warmup 0 --hf-save-dtype float32 --allow-cpu \
    --mesh '{"axes":{"data":1,"replica":1,"model":1,"context":1,"expert":1},"compute_mapping":{"batch":["data"]}}'
