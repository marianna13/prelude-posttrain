#!/bin/bash
# Diagnostic ChatML (thinking-off) RULER replay of the frozen documents; not a gate.
# Env: RUNS="name=hf_dir:ctx,ctx name=hf_dir:ctx ..."
#SBATCH --job-name=prelude-ruler-chatml
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp/%x_%j.log
set -euo pipefail
module load singularity/4.1.5
ROOT=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1
G=$ROOT/gates_20261018
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR=${SLURM_TMPDIR:-/tmp}/pg-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
for spec in ${RUNS:?}; do
    name=${spec%%=*}; rest=${spec#*=}; model=${rest%%:*}; ctxs=${rest#*:}
    for C in ${ctxs//,/ }; do
        ID=$name-ruler-chatml-$C
        echo "=== $ID model=$model"
        singularity exec --nv --bind "$ROOT:$ROOT,$TMPDIR:$TMPDIR" /gpfs/scratch/ehpc532/ot/containers/mn5_rl_cu13.sif \
            python $G/infer_ruler_chatml.py --model $model --inputs $G/inputs_ruler_$C --out $G/results/$ID --context $C
        PYTHONPATH=$G/eval_venv/lib/python3.12/site-packages singularity exec --bind $ROOT:$ROOT \
            /gpfs/scratch/ehpc532/ot/containers/mn5_sft_cu13.sif /opt/venvs/sft-cu13/bin/python $G/score_gates.py \
            --outputs $G/results/$ID/outputs.jsonl > $G/results/$ID/scores.json
        python3 -c "import json;d=json.load(open('$G/results/$ID/scores.json'));print('$ID', 'RULER mean %.2f%%' % d['ruler_mean_pct'], 'passkey %.0f%%' % d['niah_single_1']['score_pct'], 'squad %.1f hotpot %.1f' % (d['ruler_qa_squad']['score_pct'], d['ruler_qa_hotpot']['score_pct']))"
    done
done
