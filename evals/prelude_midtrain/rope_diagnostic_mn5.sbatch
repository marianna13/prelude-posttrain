#!/bin/bash
#SBATCH --job-name=prelude-rope-control
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
set -euo pipefail
G=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/gates_20261018
while IFS=$'\t' read -r name model; do
    GATE_MODEL="$model" GATE_RUN_ID="diagnostic-$name" GATE_CONTEXT=4096 \
        GATE_INPUTS="$G/inputs_niah_4096_diagnostic" bash "$G/eval_mn5.sbatch"
done < "$G/rope-diagnostic-cases.tsv"
