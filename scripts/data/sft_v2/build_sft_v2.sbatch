#!/bin/bash
#SBATCH --job-name=prelude-sft-v2-build
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v2/logs/%x_%j.out
#
# Prelude SFT v2 data build (issue 0058), CPU only. Needs census/ and decontam/flags/ (census_sft_v2.sbatch,
# decontam_sft_v2.sbatch).
#   1. select + normalize -> selection/, normalized/<mix_source>/part-*.parquet  (materialize_prelude_sft_v2.py)
#   2. render with the Prelude chat tokenizer, cap at MAX_TOKENS -> rendered/<mix_source>/part-*.parquet
#      (prerender_levanter.py, enable_thinking unset: thinking-off rows carry their empty think block)
#   3. summary.json: rows / tokens / trained tokens per mix_source
# Resumable: finished normalized and rendered parts are skipped.
set -euo pipefail
REPO=${REPO:-/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v2/tools/rl}
PY=${PY:-/p/project1/ccstdl/envs/marianna/lmeval/bin/python}
TOK=${TOK:-/p/data1/mmlaion/marianna/sft_sources/prelude_chat_tokenizer}
OUT=${OUT:-/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v2}
MAX_TOKENS=${MAX_TOKENS:-65536}
RENDER_JOBS=${RENDER_JOBS:-120}
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
mkdir -p "$OUT/logs"

echo "[$(date +%T)] 1. select + normalize"
[ -f "$OUT/selection/selection.json" ] || "$PY" -I "$REPO/scripts/data/sft_v2/materialize_prelude_sft_v2.py" select --out "$OUT"
"$PY" -I "$REPO/scripts/data/sft_v2/materialize_prelude_sft_v2.py" normalize --out "$OUT" --procs 120 | tail -80

echo "[$(date +%T)] 2. render (max $MAX_TOKENS tokens, $RENDER_JOBS parallel)"
render() {   # in, out
    [ -f "$2" ] && return 0
    mkdir -p "$(dirname "$2")"
    "$PY" -I "$REPO/scripts/sft_gen/prerender_levanter.py" "$1" "$2.tmp" --tokenizer "$TOK" --enable-thinking unset \
        --max-tokens "$MAX_TOKENS" --skip-unrenderable > "${2%.parquet}.report.json" 2> "${2%.parquet}.err" \
        && mv "$2.tmp" "$2" || echo "FAILED $1" >&2
}
export -f render; export PY REPO TOK MAX_TOKENS
find "$OUT/normalized" -name 'part-*.parquet' -size +0 | sort | \
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
    skipped = sum(v for k, v in c.items() if k.startswith("skipped_"))
    print(f"{s:30s} rows {c.get('rows', 0):>9,}  tokens {c.get('tokens', 0)/1e9:7.2f}B  trained {c.get('trained_tokens', 0)/1e9:7.2f}B"
          f"  skipped {skipped:>7,}  failed_parts {c.get('failed_parts', 0)}")
EOF
echo "[$(date +%T)] done: $OUT"
