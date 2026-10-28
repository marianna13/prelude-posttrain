#!/bin/bash
# Export a Megatron mid-training checkpoint to HF, then check that the export reproduces the checkpoint's loss:
# HF per-document on the export vs Megatron THD on the checkpoint, on Levanter packed sequences 0-15.
#SBATCH --job-name=prelude-export-hf
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp/%x_%j.log
set -uo pipefail
module load singularity/4.1.5
H=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp
SIF=/gpfs/scratch/ehpc532/ot/containers/mn5_sft_cu13.sif
BASE=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/base_hf
CKPT=${CKPT:?CKPT=<checkpoints>/iter_NNNNNNN}
OUT=${OUT:?OUT=<hf export dir>}
THETA=${THETA:-6000000}
N=$H/levanter_stage1_seq0-127.npz
[ -f $OUT/megatron_export.json ] && { echo "export exists: $OUT (skipping export and loss check)"; exit 0; }
O=$H/export_check_$SLURM_JOB_ID; mkdir -p $O/cache
ENVS=PYTHONPATH=$H/stubs,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,BASE_HF=$BASE
# megatron_midtrain.py reads its training env at import; only its shims are used here
ENVS=$ENVS,STAGE_DIR=/nonexistent,SEQ_LEN=16384,GBS=64,ROPE_THETA=$THETA,END_ITER=1
ENVS=$ENVS,YARN_FACTOR=${YARN_FACTOR:-0},YARN_ORIG=${YARN_ORIG:-65536}  # a YaRN checkpoint needs both
singularity exec --nv --cleanenv --env $ENVS,HF_HOME=$O/cache/hf \
    $SIF /opt/venvs/sft-cu13/bin/python $H/export_hf.py $BASE $CKPT $OUT || exit 1
python3 -c "import json;c=json.load(open('$OUT/config.json'));print({k:c.get(k) for k in ['rope_theta','rope_parameters','max_position_embeddings']})"
ls -la $OUT
run() {  # gpu engine model tag port
    singularity exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$1,$ENVS,TRITON_CACHE_DIR=$O/cache/t$1,HF_HOME=$O/cache/hf$1 \
        $SIF /opt/venvs/sft-cu13/bin/python $H/parity_loss.py $2 $3 $N 0 15 $THETA $O/$4.json $5 > $O/$4.log 2>&1
    echo "$4 exit=$?"
}
run 0 hf         $OUT  hf_export       29611 &
run 1 mcore_ckpt $CKPT mcore_ckpt      29612 &
run 2 hf         $BASE hf_base         29613 &
wait
python3 - $O <<'PY'
import json, sys
o = sys.argv[1]
r = {t: json.load(open(f"{o}/{t}.json")) for t in ["hf_export", "mcore_ckpt", "hf_base"]}
for t, v in r.items():
    print(t, "mean loss %.5f" % (sum(x["loss_sum"] for x in v) / sum(x["count"] for x in v)))
d = [abs(a["loss_sum"] / a["count"] - b["loss_sum"] / b["count"]) for a, b in zip(r["hf_export"], r["mcore_ckpt"])]
print("max per-seq |hf_export - mcore_ckpt| %.5f" % max(d))
PY
