#!/bin/bash
# Build the published Prelude data recipe (build_recipe.py) on a JURECA CPU node. Env: OUT, ONLY (midtrain|sft).
# Submit from recipe/ (Slurm runs a copy of this file, so $0 does not point here).
#SBATCH --job-name=prelude-recipe
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=06:00:00
set -euo pipefail
PY=${PY:-/p/project1/ccstdl/envs/marianna/lmeval/bin/python}
OUT=${OUT:-/p/scratch/synthlaion/marianna/prelude_recipe/hf}
HERE=${HERE:-$SLURM_SUBMIT_DIR}
"$PY" -u "$HERE/build_recipe.py" --out "$OUT" --procs 96 ${ONLY:+--only $ONLY}
