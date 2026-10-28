#!/bin/bash
# Attention backend A/B on one node (attn_bench.py): GPU0 cuDNN fused, GPU1 FA2, GPU2 FA3 3.0.1, GPU3 TE auto
# (as in training, FA3 not on the path; NVTE_DEBUG logs the choice). 64K and 128K packed rows.
#SBATCH --job-name=prelude-attn-ab
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp/%x_%j.log
set -uo pipefail
module load singularity/4.1.5
H=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp
P=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_packed
SIF=/gpfs/scratch/ehpc532/ot/containers/mn5_sft_cu13.sif
FA3=/gpfs/scratch/ehpc532/ot/pylibs/fa3_301
O=$H/attn_ab_$SLURM_JOB_ID; mkdir -p $O
run() {  # gpu label extra-env
    singularity exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$1,TRITON_CACHE_DIR=$O/t$1,$3 $SIF bash -c \
        "for s in 65536 131072; do /opt/venvs/sft-cu13/bin/python $H/attn_bench.py $2 $P/\$s 8 $O/results.jsonl; done" \
        > $O/$2.log 2>&1
    echo "$2 exit=$?"
}
run 0 fused "NVTE_FLASH_ATTN=0,NVTE_FUSED_ATTN=1" &
run 1 fa2 "NVTE_FLASH_ATTN=1,NVTE_FUSED_ATTN=0" &
run 2 fa3 "NVTE_FLASH_ATTN=1,NVTE_FUSED_ATTN=0,PYTHONPATH=$FA3" &
run 3 auto "NVTE_DEBUG=1,NVTE_DEBUG_LEVEL=2" &
wait
grep -h -m3 -i -E "selected backend|available backends|flash_attn_3|FlashAttention 3|use_flash|use_fused" $O/*.log | cut -c1-250
grep -h -E "^\{" $O/*.log
