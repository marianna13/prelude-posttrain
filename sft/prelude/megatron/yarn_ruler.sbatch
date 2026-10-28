#!/bin/bash
# RULER (plain + ChatML diagnostic) on a YaRN-config variant; env MODEL NAME CONTEXTS.
#SBATCH --job-name=prelude-yarn-ruler
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp/%x_%j.log
set -uo pipefail
module load singularity/4.1.5
R=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1
G=$R/gates_20261018
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
score() {
    PYTHONPATH=$G/eval_venv/lib/python3.12/site-packages singularity exec --bind $R:$R \
        /gpfs/scratch/ehpc532/ot/containers/mn5_sft_cu13.sif /opt/venvs/sft-cu13/bin/python $G/score_gates.py \
        --outputs $G/results/$1/outputs.jsonl > $G/results/$1/scores.json
    python3 -c "import json;d=json.load(open(\"$G/results/$1/scores.json\"));print(\"$1\", \"RULER mean %.2f%%\" % d[\"ruler_mean_pct\"], \"passkey %.0f%%\" % d[\"niah_single_1\"][\"score_pct\"])"
}
for C in ${CONTEXTS:?}; do
    ID=${NAME:?}-ruler-$C
    echo "=== $ID"
    GATE_MODEL=${MODEL:?} GATE_RUN_ID=$ID GATE_CONTEXT=$C GATE_INPUTS=$G/inputs_ruler_$C bash $G/eval_mn5.sbatch && score $ID
    ID=$NAME-ruler-chatml-$C
    echo "=== $ID"
    TMPDIR=${SLURM_TMPDIR:-/tmp}/pgc-$SLURM_JOB_ID; mkdir -p $TMPDIR
    singularity exec --nv --bind "$R:$R,$TMPDIR:$TMPDIR" --env HF_HUB_OFFLINE=1,VLLM_WORKER_MULTIPROC_METHOD=spawn,VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,TMPDIR=$TMPDIR \
        /gpfs/scratch/ehpc532/ot/containers/mn5_rl_cu13.sif python $G/infer_ruler_chatml.py --model $MODEL --inputs $G/inputs_ruler_$C --out $G/results/$ID --context $C && score $ID
done
