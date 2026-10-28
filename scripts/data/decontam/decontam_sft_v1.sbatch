#!/bin/bash
#SBATCH --job-name=prelude-sft-v1-decontam
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v1/logs/%x_%j.out
# Benchmark decontamination flags for Prelude SFT v1 (index_prelude_v1, 13-gram, any hit).
set -euo pipefail
O=/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v1
/p/project1/ccstdl/envs/marianna/lmeval/bin/python -I \
    /p/project1/laionize/marianna/dc_agent/monorepo/rl/scripts/data/decontam/decontam_conversations.py \
    --index /p/data1/mmlaion/marianna/decontam/index_prelude_v1 --inp "$O/normalized/*/part-*.parquet" \
    --out "$O/decontam" --workers 120
