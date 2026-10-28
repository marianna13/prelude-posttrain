#!/bin/bash
# Prelude SFT v2 step-1 census (issue 0058): token/shape census of every candidate row + SWE overlap.
#SBATCH --job-name=prelude-sft-v2-census
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
set -euo pipefail
ROOT=/p/data1/mmlaion/marianna/sft_sources
OUT=${OUT:-/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v2/census}
HERE=${HERE:-$(dirname "$(readlink -f "$0")")}
PY=/p/project1/ccstdl/envs/marianna/lmeval/bin/python
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
until grep -q "ALL DONE" $ROOT/sft_v2_logs/download.log; do sleep 60; done
mkdir -p $OUT
for s in nebius agenttrove i3:am_chat i3:am_if i3:openreasoning_tool i3:toucan_tool i3:swe_swiss \
         i3:openreasoning_science i3:openreasoning_code i3:openreasoning_math; do
    name=$(echo "$s" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_]+/_/g')
    [ -f $OUT/$name.summary.json ] && { echo "=== $s already done"; continue; }  # resubmits continue
    echo "=== $s $(date +%T)"
    $PY $HERE/census_sft_v2.py --source $s --root $ROOT --out $OUT --procs 96
done
echo "=== swe overlap $(date +%T)"
$PY $HERE/swe_overlap.py --root $ROOT --out $OUT
echo "=== done $(date +%T)"
