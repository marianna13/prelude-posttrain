#!/bin/bash
# Export one SFT v2 full checkpoint (from stage 4, YaRN x2 trained) to HF with the loss check, then write the YaRN
# rope_parameters into the export config as stage4_post.sbatch does. Env: RUN_NAME ITER (number)
#SBATCH --job-name=prelude-sft-full-export
#SBATCH --account=ehpc533
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp/%x_%j.log
set -uo pipefail
H=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp
RUN=$H/runs/${RUN_NAME:?}
I=$(printf iter_%07d "${ITER:?}")
export CKPT=$RUN/checkpoints/$I OUT=$RUN/hf/$I THETA=9000000 YARN_FACTOR=2 YARN_ORIG=65536
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
