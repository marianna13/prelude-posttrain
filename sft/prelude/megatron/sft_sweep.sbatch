#!/bin/bash
# SFT throughput sweep from the stage-3 export: 12 steps per config (1M tokens/step) on the mid-training packed
# rows of the config's context, one allocation, configs in sequence. Per-config logs: sweep_<job>/<cfg>.log.
#SBATCH --job-name=prelude-sft-sweep
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp/%x_%j.log
set -uo pipefail
H=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp
O=$H/sweep_$SLURM_JOB_ID; mkdir -p $O
export MODEL=$H/runs/stage3-64k-theta9m/hf/iter_0004419 ROPE_THETA=9000000
export END_ITER=12 SCHEDULE_ITERS=2000 LR=1e-5 GRAD_FP32=1 SAVE_INTERVAL=1000000
# config: <tp>-<cp>-<recompute>-<context>, e.g. 4-1-full-65536, 4-1-block:24-65536 (GBS keeps 1M tokens per step)
for cfg in ${CONFIGS:-4-1-full-65536}; do
    IFS=- read -r TP CP RECOMPUTE STAGE_CONTEXT <<< "$cfg"
    GBS=$((1048576 / STAGE_CONTEXT))
    export TP CP RECOMPUTE STAGE_CONTEXT GBS RUN_NAME=sweep-$SLURM_JOB_ID-${cfg//:/_}
    export CKPT_DIR=$H/runs/sweep/$RUN_NAME/checkpoints
    echo "=== $cfg $(date +%T)"
    L=$O/${cfg//:/_}.log
    timeout 25m bash $H/megatron_midtrain.sbatch > $L 2>&1
    echo "$cfg exit=$? $(date +%T)"
    grep -E "iteration +1[0-2]/" $L | cut -c1-200
    echo "oom_lines $(grep -h -c -E "OutOfMemoryError|CUDA out of memory" $L)"
    rm -rf $H/runs/sweep/$RUN_NAME
done
