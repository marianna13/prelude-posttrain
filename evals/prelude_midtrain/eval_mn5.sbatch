#!/bin/bash
#SBATCH --job-name=prelude-gates-eval
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
set -euo pipefail
module load singularity/4.1.5
G=${PRELUDE_GATE_ROOT:-/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/gates_20261018}
ROOT=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR=${SLURM_TMPDIR:-/tmp}/pg-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
exec singularity exec --nv --bind "$ROOT:$ROOT,$TMPDIR:$TMPDIR" \
    /gpfs/scratch/ehpc532/ot/containers/mn5_rl_cu13.sif python "$G/infer_gates.py" \
    --model "${GATE_MODEL:-$ROOT/base_hf}" --inputs "$GATE_INPUTS" \
    --out "$G/results/${GATE_RUN_ID:?}" --context "${GATE_CONTEXT:-4096}"
