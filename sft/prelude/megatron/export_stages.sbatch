#!/bin/bash
# Export the three frozen Prelude mid-training stages (16K/32K/64K) to Megatron packed arrays (export_stage.py).
#SBATCH --job-name=prelude-export-megatron
#SBATCH --account=ehpc1048
#SBATCH --partition=gpp
#SBATCH --qos=gp_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp/%x_%j.log
set -uo pipefail
module load singularity/4.1.5
P=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1
R=$P/gates_20261018/runtime_production_v1/rl/sft
OUT=$P/megatron_packed
X=(singularity exec --bind /gpfs/scratch/ehpc532/ot/marin-sft/pylibs:/pylibs --pwd /marin
   /gpfs/scratch/ehpc532/ot/containers/levanter_jax_26.09-py3_torch.sif
   env JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 PYTHONPATH=$R/prelude:$R/levanter:/pylibs/site
   /marin/.venv/bin/python $P/megatron_tp/export_stage.py)
mkdir -p $OUT
stage() {
    local c=$1
    { "${X[@]}" prepare $P/cache_v3/$c $P/base_hf $c $OUT/$c \
      && "${X[@]}" fill $P/cache_v3/$c $P/base_hf $c $OUT/$c 0 1000000000 \
      && "${X[@]}" verify $P/cache_v3/$c $P/base_hf $c $OUT/$c; } > $OUT/export_$c.log 2>&1
    echo "stage $c exit=$?"
}
stage 16384 & stage 32768 & stage 65536 &
wait
du -sh $OUT/*
