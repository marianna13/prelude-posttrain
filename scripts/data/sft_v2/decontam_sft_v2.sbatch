#!/bin/bash
# Prelude SFT v2 dedup + decontamination flags (issue 0058). STEP=refs|flags|finalize; refs and flags run in
# parallel, finalize after both:
#   R=$(STEP=refs sbatch --parsable decontam_sft_v2.sbatch); F=$(STEP=flags sbatch --parsable decontam_sft_v2.sbatch)
#   STEP=finalize sbatch --dependency=afterok:$R:$F decontam_sft_v2.sbatch
#SBATCH --job-name=prelude-sft-v2-decontam
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=06:00:00
set -euo pipefail
OUT=${OUT:-/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v2/decontam}
HERE=${HERE:-$(dirname "$(readlink -f "$0")")}
PY=/p/project1/ccstdl/envs/marianna/lmeval/bin/python
export OMP_NUM_THREADS=1
mkdir -p $OUT
echo "=== ${STEP:?} $(date +%T)"
$PY -I $HERE/decontam_sft_v2.py $STEP --out $OUT --procs ${PROCS:-120} ${SOURCE:+--source $SOURCE}
echo "=== done $(date +%T)"
