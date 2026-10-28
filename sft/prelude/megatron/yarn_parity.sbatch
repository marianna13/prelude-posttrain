#!/bin/bash
# HF YaRN export loss on the first stage-4 batch (rows 0-7), to compare with Megatron step-1 lm loss.
#SBATCH --job-name=prelude-yarn-parity
#SBATCH --account=ehpc532
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp/%x_%j.log
module load singularity/4.1.5
H=/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp
M=$H/runs/stage3-64k-theta9m/hf/iter_0004419-yarn2-128k
O=$H/yarn_parity_$SLURM_JOB_ID; mkdir -p $O
for g in 0 1 2 3; do
  singularity exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$g,HF_HOME=$O/hf$g --bind /gpfs/scratch/ehpc532:/gpfs/scratch/ehpc532 \
    /gpfs/scratch/ehpc532/ot/containers/mn5_sft_cu13.sif /opt/venvs/sft-cu13/bin/python $H/parity_loss.py hf $M $H/stage4_rows0-7.npz $((2*g)) $((2*g+1)) 9000000 $O/hf$g.json > $O/hf$g.log 2>&1 &
done
wait
python3 - $O <<"PY"
import json, sys, glob
r = [x for f in sorted(glob.glob(sys.argv[1] + "/hf*.json")) for x in json.load(open(f))]
print(len(r), "rows; HF YaRN mean loss %.5f (Megatron step-1 lm loss 0.95970)" % (sum(x["loss_sum"] for x in r) / sum(x["count"] for x in r)))
PY
