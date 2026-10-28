#!/bin/bash
# After a Megatron mid-training stage: export its latest checkpoint to HF (with the loss parity check), then run and
# score the frozen RULER gates on MN5. The 4K RULER niah_single_1 inputs are byte-identical to the matched 4K passkey
# control (inputs_niah_4096_diagnostic), so the 4K run also gives the control.
# Env: RUN_NAME THETA GATE_CONTEXTS (e.g. "4096 16384")
#SBATCH --job-name=prelude-post-stage
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
R=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1
H=$R/megatron_tp
G=$R/gates_20261018
RUN=$H/runs/${RUN_NAME:?}
export THETA=${THETA:?}
ITER=$(printf iter_%07d "$(cat $RUN/checkpoints/latest_checkpointed_iteration.txt)")
export CKPT=$RUN/checkpoints/$ITER OUT=$RUN/hf/$ITER
echo "run=$RUN_NAME checkpoint=$CKPT export=$OUT theta=$THETA contexts=${GATE_CONTEXTS:?}"

bash $H/export_check.sbatch

for C in $GATE_CONTEXTS; do
    ID=$RUN_NAME-$ITER-ruler-$C
    GATE_MODEL=$OUT GATE_RUN_ID=$ID GATE_CONTEXT=$C GATE_INPUTS=$G/inputs_ruler_$C bash $G/eval_mn5.sbatch
    singularity exec --bind $R:$R \
        --env HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1,OMP_NUM_THREADS=1,PYTHONPATH=$G/eval_venv/lib/python3.12/site-packages \
        /gpfs/scratch/ehpc532/ot/containers/mn5_sft_cu13.sif /opt/venvs/sft-cu13/bin/python $G/score_gates.py \
        --outputs $G/results/$ID/outputs.jsonl > $G/results/$ID/scores.json
    python3 -c "import json;d=json.load(open('$G/results/$ID/scores.json'));print('$ID', 'RULER mean %.2f%%' % d['ruler_mean_pct'], 'passkey %.0f%%' % d['niah_single_1']['score_pct'], 'target_met', d['target_met'])"
done
