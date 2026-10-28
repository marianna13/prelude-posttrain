#!/bin/bash
# After stage 4 (128K, YaRN x2 trained): export the latest checkpoint to HF with the loss check, write the YaRN
# rope_parameters into the export config, then RULER plain + ChatML at CONTEXTS. Env: RUN_NAME CONTEXTS
#SBATCH --job-name=prelude-stage4-post
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=09:00:00
#SBATCH --output=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp/%x_%j.log
set -uo pipefail
H=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp
RUN=$H/runs/${RUN_NAME:?}
ITER=$(printf iter_%07d "$(cat $RUN/checkpoints/latest_checkpointed_iteration.txt)")
export CKPT=$RUN/checkpoints/$ITER OUT=$RUN/hf/$ITER THETA=9000000 YARN_FACTOR=2 YARN_ORIG=65536
echo "checkpoint=$CKPT export=$OUT"
bash $H/export_check.sbatch || exit 1
python3 - $OUT/config.json <<"PY"
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["rope_parameters"] = {"rope_type": "yarn", "rope_theta": 9e6, "factor": 2.0, "original_max_position_embeddings": 65536}
c["max_position_embeddings"] = 131072
json.dump(c, open(p, "w"), indent=2)
print("config", c["rope_parameters"], c["max_position_embeddings"])
PY
MODEL=$OUT NAME=$RUN_NAME-$ITER CONTEXTS=${CONTEXTS:?} bash $H/yarn_ruler.sbatch
