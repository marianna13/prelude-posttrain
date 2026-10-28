# Prelude mid-training adapters (issue 0054, step 2)

Step 2 implements source normalization, target masks, tokenizer/model compatibility,
and loader checks. The artifacts are under
`/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step2`.
The bounded pilot covers every included source/subset/language; it is not a full
source materialization or a cleaned training release. Deduplication, benchmark
decontamination, mixture weights, long-document chunking, the context curriculum,
and training were assigned to steps 3–6 of [issue 0054](../../../issues/0054-build-prelude-9b-midtraining-data.md).
Steps 3–4 subsequently completed; the portable token release and current loader/
training qualification are documented in [EXPORT_MIDTRAIN.md](EXPORT_MIDTRAIN.md).

## Canonical rows and source adapters

[`prelude_midtrain.py`](../../scripts/data/prelude_midtrain.py) is independent of
the assistant-only SFT normalizer. It preserves raw document whitespace and code
indentation. Its canonical Parquet schema is:

```
id, kind, text, messages_json, tools_json, provenance_json, policy, mix_source
```

Only documents have `text`; conversations have `messages_json` and optional tools.
Provenance includes the source/revision, physical input SHA256, relative path,
record locator and hash, source UUID/teacher/metadata, and selected repository/file
identities. IDs hash the source identity and locator, independently of task
partitioning or filtering. JSONL locators are byte offsets; Parquet and compressed
replay locators are global decoded row indices.

- Cascade: all nine frozen JSONL files, including each math and SWE variant.
- Specialized: the five selected categories, read from their actual `text` column.
- Terminal Corpus: only `terminus-2`; first user message is the task, later user
  messages are environment feedback. Other agents are rejected explicitly.
- UltraData: raw file `content` with repository, commit, path, content ID, license
  and quality metadata retained. Vendor files are excluded.
- Stack: only the step 1 selected repository IDs; files must be non-vendor and
  `permissive` or `no_license`. Sort by `(file_path, content_id)` and assemble the
  step 1 representation: repository name, then `File: path` and unmodified content.
  Each included file retains its provenance; code/long-document routing is retained.
- Replay/long PDFs: the 49 pinned files and their first-N row cuts from step 1.
  Normalized rows remain candidates: the short-replay/long-document eligibility
  lengths and final routing must still be applied when building the mixture.

Whole-message `<tool_response>...</tool_response>` wrappers in user or tool roles
become semantic tool responses, including multiple consecutive wrapped responses.
Payload whitespace stays intact; the template inserts exactly one response wrapper.
Wrappers quoted inside a genuine user question remain trained text. Boundary-mixed,
nested, partial, or orphan response wrappers are rejected with a stable reason.
System prompts containing example tags remain system prompts. Literal ChatML tags
in content are retained and counted; masking uses message provenance, not a regex
over role-like text in payloads.

Earlier assistant reasoning remains trained. Inline reasoning/calls stay inline;
separate reasoning and structured calls use the frozen template. Identical duplicate
reasoning is collapsed once; conflicting reasoning or simultaneous inline and
structured calls is rejected. Source `train_loss` restrictions are ignored in this
mode. Empty text turns are retained; withheld/non-text prompts are rejected.
Every applied normalization rule and rejection has a counter. Rule counters include
attempted normalization of subsequently rejected records. Rejection audits identify
the exact source task and record locator without copying rejected payloads.

## Target-mask contract

Rendered rows are `{id, ids: list<int32>, loss: list<int8>, mix_source}`. `loss[t]`
describes the token at `ids[t]`, before causal shifting:

| Span | Target loss |
| --- | ---: |
| Initial BOS | 0 |
| Document text and native EOS (2) | 1 |
| Ordinary system/user/assistant headers, bodies, delimiters and trailing LF | 1 |
| Assistant reasoning, tool calls and tool definitions in system context | 1 |
| Complete tool response group: shared user header, wrappers, payload, end delimiter, trailing LF | 0 |
| Padding and the next packed example's BOS | 0 |

Consecutive tool responses share a user header in the Prelude template. Mask
boundaries are derived at complete semantic group boundaries and checked for
prefix consistency and exact native token alignment. The final tool token can
predict the following assistant header: the assistant header is a trained target.

Levanter's actual `ChatDataset` rolls the target mask left by one. At predictor
position `p`, the loss weight is `loss[p+1]`; the final predictor weight is zero.
The packer zero-pads IDs/masks (its padding ID is 0, not tokenizer pad ID 262144)
and assigns padding segment -1. Segment IDs block attention between documents.
Untrained BOS prevents loss across packed boundaries. Use `--format midtrain` in
[`sft_main.py`](../levanter/scripts_grug/sft_main.py): it consumes the stored mask,
rejects implicit slicing, and selects the Prelude checkpoint import/export hooks.
Existing `chat`/`prerendered` SFT defaults remain unchanged.

Normalization retains overlength rows. Rendering defers rows above 65,536 tokens;
it never silently truncates them. Documents use actual native IDs at verified,
unmergeable LF barriers and stop allocating when overlength is certain. A single
LF-free piece above 8 MiB is explicitly deferred. The census's count-only proxy
is never used as training IDs. Later chunking must retain parent IDs and account
for every deferred row.

## Frozen tokenizer and initialized base

`tokenizer_contract.json` records input and derived bundle hashes, token IDs, the
base revision/config, and both safetensors matrix shapes: `[262272, 4096]`.
`tokenizer.json` and the template are byte-identical to step 1. The derived
`special_tokens_map.json` fixes stale native EOS metadata to agree with ChatML EOS
in `tokenizer_config.json`; generation stops on either 262146 or native EOS 2.

The six added IDs 262145–262150 occupy existing model padding rows. In each of
`model.embed_tokens.weight` and `lm_head.weight`, initialize those rows to that
matrix's mean native row over `[0,262144)`, excluding pad. Accumulate in float64,
then round to storage dtype (BF16 uses nearest-even). Do this once at fresh base
import. Resume and export never reinitialize. Every other model byte, including
the native pad row, is checked unchanged. `base_hf` contains the initialized shard
and symlinks to unchanged source shards, plus the frozen tokenizer; keep the
original base available when using it.

The dedicated trainer hook disables Levanter's default shrinking from padded
model vocabulary to tokenizer length and uses that same padded axis when building
the native checkpoint resume exemplar. The tokenizer stays at 262151 entries.
Each local HF export retains 262272 model
rows and receives all eight frozen tokenizer files and the contract. This is
verified by importing/exporting a real tiny Qwen3 model with the full vocabulary.
The base still has its original 4K model configuration: these checks do not enable
64K model attention or constitute a training smoke run.

## Reproduce

From the repository root (use a CPU allocation for materialization/initialization):

```bash
PY=/p/project1/ccstdl/envs/marianna/lmeval/bin/python
STEP1=/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step1
STEP2=/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step2
BASE=/p/data1/mmlaion/shared/models/openeurollm/prelude-anneal300b-ml1-dominant2-iter0989075

"$PY" -I rl/sft/prelude/prepare_midtrain_tokenizer.py freeze --step1 "$STEP1" --out "$STEP2" --model "$BASE"
"$PY" -I rl/scripts/data/materialize_prelude_midtrain.py prepare --step2 "$STEP2"
"$PY" -I rl/scripts/data/materialize_prelude_midtrain.py pilot --step2 "$STEP2" --limit 3
# Full individual work unit, selected from adapter_manifest.json (6740 units):
"$PY" -I rl/scripts/data/materialize_prelude_midtrain.py run --step2 "$STEP2" --task core-0
# Run once to a new destination, never over a trained checkpoint:
"$PY" -I rl/sft/prelude/prepare_midtrain_tokenizer.py initialize-hf --contract "$STEP2/tokenizer_contract.json" --out "$STEP2/base_hf"

# Explicit renderer mode for canonical normalized Parquet:
"$PY" -I rl/scripts/sft_gen/prerender_levanter.py IN.parquet OUT.parquet \
  --tokenizer "$STEP2/tokenizer" --mask-policy midtrain --max-tokens 65536 --skip-unrenderable
```

Outputs and task completion records are checksummed. Resume validates the frozen
code/input/tokenizer build, task configuration and completed artifacts. Code
changes require a new build directory; never combine completed tasks across builds.

```bash
"$PY" -I - <<'PY'
import sys, unittest
from pathlib import Path
sys.path[:0] = [str(Path('rl/scripts/data').resolve()), str(Path('rl/sft/prelude').resolve())]
suite = unittest.defaultTestLoader.loadTestsFromNames([
  'test_prelude_midtrain', 'test_prelude_midtrain_census',
  'test_select_prelude_midtrain_repos', 'test_prelude_lf_counting',
  'test_midtrain_initialization'])
raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
PY
apptainer exec --bind /p:/p /p/data1/mmlaion/container/levanter_jax_26.09-py3_torch.sif \
  env JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  /marin/.venv/bin/python rl/sft/prelude/test_midtrain_loader.py
```
