#!/bin/bash
# Loss parity on Levanter's first packed batches: HF (per-document) vs Megatron (packed THD), two RoPE thetas.
#SBATCH --job-name=prelude-parity
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
M=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/base_hf
N=$H/levanter_stage1_seq0-127.npz
O=$H/parity_$SLURM_JOB_ID; mkdir -p $O/cache
run() {  # gpu engine first last theta tag port
    singularity exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$1,PYTHONPATH=$H/stubs,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,TRITON_CACHE_DIR=$O/cache/t$1,HF_HOME=$O/cache/hf$1 \
        $SIF /opt/venvs/sft-cu13/bin/python $H/parity_loss.py $2 $M $N $3 $4 $5 $O/$6.json $7 > $O/$6.log 2>&1
    echo "$6 exit=$?"
}
run 0 hf    0 63  100000  hf_theta100k    29601 &
run 1 mcore 0 63  100000  mcore_theta100k 29602 &
run 2 hf    0 127 6000000 hf_theta6m      29603 &
run 3 mcore 0 63  6000000 mcore_theta6m   29604 &
wait
ls -la $O
