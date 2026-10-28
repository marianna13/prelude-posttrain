#!/bin/bash
#SBATCH --job-name=prelude-gates-grade
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
set -euo pipefail
W=${MIDTRAIN_REPO:-/p/project1/laionize/marianna/dc_agent/prelude_gates_work}
G=/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/gates_20261018
export HF_HOME=/p/scratch/synthlaion/dc-agent-shared/lmeval_hf
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
export HF_METRICS_CACHE=$G/grade_cache_$SLURM_JOB_ID
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
exec /p/project1/ccstdl/envs/marianna/lmeval/bin/python "$W/rl/evals/prelude_midtrain/score_gates.py" \
    --outputs "${GATE_OUTPUTS:?}"
