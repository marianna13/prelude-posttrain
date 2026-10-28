#!/bin/bash
#SBATCH --job-name=prelude-sft-v1-build
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v1/logs/%x_%j.out
#
# Prelude SFT v1 data build (math / science / code / IF-chat / SWE), CPU only:
#   1. normalize the five sources -> normalized/<subset>/part-*.parquet  (materialize_prelude_sft_v1.py)
#   2. render each part with the Prelude chat tokenizer (ChatML + <think> + tools), cap at MAX_TOKENS
#      -> rendered/<subset>/part-*.parquet {id, ids, loss, mix_source} + .report.json  (prerender_levanter.py)
#   3. summary.json: rows / tokens / trained tokens per subset
#
#   sbatch build_sft_v1.sbatch            # knobs: MAX_TOKENS (65536), SUBSETS, RENDER_JOBS (120)
set -euo pipefail
REPO=/p/project1/laionize/marianna/dc_agent/monorepo/rl
PY=${PY:-/p/project1/ccstdl/envs/marianna/lmeval/bin/python}
SRC=${SRC:-/p/data1/mmlaion/marianna/sft_sources/nvidia}
TOK=${TOK:-/p/data1/mmlaion/marianna/sft_sources/prelude_chat_tokenizer}
OUT=${OUT:-/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v1}
MAX_TOKENS=${MAX_TOKENS:-65536}
RENDER_JOBS=${RENDER_JOBS:-120}
SUBSETS=${SUBSETS:-math,science_syn_mcq,science_rqa,science_so,science_vendor,if_instruction_following,if_chat,swe,ocr}
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p "$OUT/normalized" "$OUT/rendered" "$OUT/logs"

echo "[$(date +%T)] 1. normalize: $SUBSETS"
"$PY" -I "$REPO/scripts/data/materialize_prelude_sft_v1.py" --src "$SRC" --out "$OUT/normalized" \
    --subsets "$SUBSETS" --workers 64 > "$OUT/logs/normalize_${SLURM_JOB_ID}.log" 2>&1
tail -60 "$OUT/logs/normalize_${SLURM_JOB_ID}.log"

echo "[$(date +%T)] 2. render (max $MAX_TOKENS tokens, $RENDER_JOBS parallel)"
render() {   # in, out
    mkdir -p "$(dirname "$2")"
    "$PY" -I "$REPO/scripts/sft_gen/prerender_levanter.py" "$1" "$2" --tokenizer "$TOK" --enable-thinking unset \
        --max-tokens "$MAX_TOKENS" --skip-unrenderable > "${2%.parquet}.report.json" 2> "${2%.parquet}.err" \
        || echo "FAILED $1" >&2
}
export -f render; export PY REPO TOK MAX_TOKENS
find "$OUT/normalized" -name 'part-*.parquet' | sort | \
    while read -r i; do echo "$i" "${i/$OUT\/normalized/$OUT/rendered}"; done | \
    xargs -n 2 -P "$RENDER_JOBS" bash -c 'render "$0" "$1"'

echo "[$(date +%T)] 3. summary"
"$PY" -I - "$OUT" <<'EOF'
import glob, json, os, sys
from collections import Counter, defaultdict
out = sys.argv[1]
by = defaultdict(Counter)
for p in glob.glob(f"{out}/rendered/*/part-*.report.json"):
    sub = os.path.basename(os.path.dirname(p))
    try:
        r = json.load(open(p))
    except json.JSONDecodeError:
        by[sub]["failed_parts"] += 1
        continue
    by[sub]["parts"] += 1
    for k, v in r.items():
        if isinstance(v, int):
            by[sub][k] += v
summary = {s: dict(c) for s, c in sorted(by.items())}
summary["_total"] = dict(sum(by.values(), Counter()))
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
for s, c in summary.items():
    print(f"{s:26s} rows {c.get('rows', 0):>9,}  tokens {c.get('tokens', 0)/1e9:7.2f}B  trained {c.get('trained_tokens', 0)/1e9:7.2f}B"
          f"  too_long {c.get('skipped_too_long', 0):>8,}  failed_parts {c.get('failed_parts', 0)}")
EOF
echo "[$(date +%T)] done: $OUT"
