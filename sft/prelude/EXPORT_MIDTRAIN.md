# Prelude portable token export and Levanter handoff (step 5)

Build root: `/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step5`.
The frozen input is step 4's `mixture_v3`, build
`7239c4ca51352ad5bafc4565cb0675b1ebdb73549fcf92c37401c4f71f493f1b`.
The [mixture policy](MIXTURE_MIDTRAIN.md) defines all parent allocations and chunk counts.

The full export and independent read-back completed on 2026-10-17:
1,207,832 parents, 1,224,141 chunks, 3,495,329,751 original tokens,
3,499,521,164 seen tokens and 3,398,497,611 trainable tokens. No selected parent
or original token was dropped or repeated. There are 26,103 data shards and
5,570,423,420 bytes of release artifacts. None of the selected chunks has zero
targets; the zero-target policy is covered by synthetic tool-window tests.
Release build: `e4e594c1190b2c02b576875e94484467d2a7983218fc98fe860ec581fa875819`;
manifest SHA256: `1acbb7b946512f3a25708e05ca1f84178d04e2e3c3c980ea99c4af276aaa7e1a`.

## Portable data

`release/train/{16384,32768,65536}/*.parquet` stores **both original document and
conversation formats as token IDs and target masks**. This avoids any tokenizer,
template or masking reinterpretation on MN5. `kind` preserves their distinction;
the original document text and structured messages remain in the immutable step 3
canonical dataset. Validation remains held out there and is not included in this
training-only export. A separate evaluation selection belongs to the evaluation recipe.

Each row has `id`, `parent_id`, `ids: list<int32>`, `loss: list<int8>`, `context`,
`chunk_index`, original/new/overlap token intervals, `mix_source`, `pool`, category,
kind, split/group, canonical part and full provenance. Parent ID/mask SHA256 values
use little-endian int32 IDs and int8 masks before chunking. All original tokens
occur exactly once as new content. Initial BOS is masked. Continuations prepend a
masked BOS 1 and 256 masked preceding original tokens. No intermediate EOS is added;
the final native document EOS 2 survives. Short tails and entirely masked windows
are retained. The first new token in a continuation remains a target whenever its
original mask was 1. No implicit slicing, padding-as-content, or tail dropping occurs.

Levanter shifts target masks left once and zero-pads packs with segment -1. Every
chunk is a separate attention segment; overlap supplies continuation context.
An entirely masked tool window is accepted only by the Prelude format. Existing
SFT defaults still reject a row with no targets. The tiny training smoke also
checks finite zero loss for a fully masked example.

## Build, resume and verification

The CPU implementation uses MV's independent shard execution and metadata/hash
reduction patterns already adapted in steps 3–4. It adds no VM build dependency.
JURECA uses `synthlaion`, partition `dc-cpu`, four nodes with 32 workers/node and
four tokenizer threads/worker. Preparation verifies all 8,932 canonical checksums
once, locates only frozen selected IDs using Parquet ID columns, then creates
bounded tasks covering at most 128 row groups each. Workers read only selected
row groups and check input state before/after processing.

The build key covers upstream manifests (including source selection, adapters,
dedup/decontamination and split identities), selections, tokenizer/template and
contract hashes, transform code, chunk/mask policy, Python and package versions,
and resource partitioning. Changed inputs require a new output directory. Workers
resume only completed shards with matching build identity and file checksum.
Frozen implementation snapshots live in `step5/tools`; use them for a resume.

From the monorepo root:

```bash
export MIDTRAIN_BASE=/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1
export MIDTRAIN_OUT=$MIDTRAIN_BASE/step5
# Preparation is a single CPU node; it freezes tools before the export starts.
MIDTRAIN_MODE=prepare \
MIDTRAIN_SCRIPT=$PWD/rl/scripts/data/export_prelude_midtrain.py \
sbatch --nodes=1 --output="$MIDTRAIN_OUT/logs/prepare-%j.log" \
  rl/sft/prelude/export_midtrain.sbatch
# After preparation (or to resume the same frozen build):
sbatch --output="$MIDTRAIN_OUT/logs/export-%j.log" rl/sft/prelude/export_midtrain.sbatch
# After all export tasks finish, freeze the verifier and run a full read-back:
cp rl/scripts/data/verify_prelude_midtrain_export.py "$MIDTRAIN_OUT/tools/"
sbatch --output="$MIDTRAIN_OUT/logs/audit-%j.log" rl/sft/prelude/verify_export_midtrain.sbatch
# Real cache, packing, masks and finite shuffle, at all three context lengths:
sbatch --output="$MIDTRAIN_OUT/logs/loader-release-%j.log" rl/sft/prelude/loader_midtrain.sbatch
```

Use `--dependency=afterok:<jobid>` when submitting dependent stages in advance.
The independent auditor reads every exported ID and mask, reconstructs original
parent digests, verifies every new-token interval/overlap, loss mask, native EOS,
schema, source/split metadata and selected-parent count. It reconciles every
source/stage against step 4, including seen and trained tokens, and rejects extra
shards. `release_manifest.json` and `SHA256SUMS` identify the immutable token release.
The manifest also records length bins, zero-target windows and bounded loader
smoke row locators covering every active source pool and continuation/mask cases.

## Levanter on either site

The exact image is `levanter_jax_26.09-py3_torch.sif`, SHA256
`b22c005916883c27ba697fd82c8973cd5f28c115eb96bddac4ac46da47ca9c7a`.
The MN5 copy under `/gpfs/scratch/ehpc532/ot/containers/` was independently hashed
and matches JURECA byte for byte. Code/runtime and CPU evidence are bundled
separately under `mn5_bundle/`, `mn5_bundle_v2/` and `mn5_bundle_v3/`, with their
own checksums. The final numeric reader is staged as `handoff_v3` on MN5. Earlier
bundles and interrupted native cache attempts remain preserved.

`prepare_midtrain_levanter.py build --release RELEASE --context CONTEXT --out CACHE`
imports stored IDs/masks using the real `SerialCacheWriter`; it never renders or
tokenizes text. Its `stage_dataset` factory packs with `slice_strategy=raise` and
provides a finite bijective shuffle (seed 20261017). Qwen3 consumes the result via
Levanter's `NamedLmDataset` adapter, as in the native training data path.
Each stage is consumed once. A production trainer must stop at its finite length
and pad any partial final batch, rather than repeat examples. The generic SFT
CLI's epoch/token estimate and default repeating mixture are not the finite
curriculum launcher. That launcher and final optimization/context settings are
step 6 work.

`smoke` builds a bounded real cache from audited row locators and verifies exact
packed IDs, shifted masks, padding, segment IDs, finite length, shuffle coverage,
all active source pools, continuation windows and zero-loss windows. It also
opens the cache after moving its root, exercising cache relocation. Use a fresh
smoke output directory for each run. Full native cache imports fail explicitly on
existing or interrupted output paths; a new cache path can be used without
repeating tokenization.

Bulk import reads only `id/ids/loss/context/split`, and writes at most 32 Mi tokens
or 16,384 rows per batch. Levanter's Zarr3 layout has 128M-element shards with
256K inner chunks; small synchronous appends repeatedly update a much larger
storage object. The bulk batch comparison read back identical IDs/masks and row/
token counts for 1,198 real rows / 8,396,926 tokens. Writes took 2.949s with small
batches versus 0.654s with bulk batching on JURECA. Full 16K/32K/64K smoke reports
also match exactly across the batching change. This affects native cache I/O;
the immutable tokenized release and its build/checksum remain unchanged.

The final reader also avoids expanding Arrow numeric buffers into Python integers.
It passes NumPy views into the existing validating preprocessor and uses int64
mask reductions. A separate native-cache read-back compared all IDs/masks for
14,274 real rows / 33,554,527 tokens: the list reader took 34.744s and the array
reader 2.507s on JURECA, with identical digests/counts. All three loader reports
remain identical. This qualified reader is the V3 handoff.

## Transfer and MN5 import

Run bulk transfers **on Jupiter**, which sees the shared `/p` paths and can reach
MN5 directly. From JURECA, first log into `jsc-vm`, then SSH from there to `jupiter`;
JURECA cannot reach Jupiter directly. Run the user's existing Jupiter `~/bin/rclone`
with its `mn5:` remote, so the bulk bytes flow from Jupiter to MN5 without crossing
the VM. See the [existing Jupiter staging notes](../../notes/mn5_sft.md). This session's
noninteractive VM-to-Jupiter login was rejected with `Permission denied (publickey)`;
the current release/model had already finished transfer and passed all checksums.

```bash
# On Jupiter, using its configured rclone remote:
~/bin/rclone copy "$MIDTRAIN_OUT/release" \
  mn5:/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/release --progress --multi-thread-streams=8
# For the initialized model, --copy-links must materialize the unchanged shards:
~/bin/rclone copy --copy-links "$MIDTRAIN_BASE/step2/base_hf" \
  mn5:/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/base_hf --progress --multi-thread-streams=8
# Alternatively, from this checkout with working BSC SSH authentication:
bash rl/sft/prelude/stage_midtrain_to_mn5.sh --with-model
```

The completed initial release/model transfer used `MIDTRAIN_SSH_JUMP=jsc-vm`
from JURECA because its direct BSC connections were refused. All destination
checksums passed; no repeat transfer is needed. Keep that route as a fallback
for small control/evidence operations. For it, BSC host keys are pinned in
`step5/network_known_hosts` using `MIDTRAIN_KNOWN_HOSTS`; private keys remain in
the user's existing SSH configuration and are never copied into the release.

The destination is `/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/{release,handoff,base_hf}`.
The staging script verifies data/runtime/model checksums at the destination.
With rclone, also copy the versioned handoff bundle and verify `release/SHA256SUMS`,
`handoff/SHA256SUMS` and `handoff/MODEL_SHA256SUMS` on MN5 before use. Model copy
uses `rsync -L` or rclone `--copy-links`: step 2's unchanged-shard symlinks must become actual files on MN5.
The initialized six embedding/head rows are transferred intact; never initialize
them again on resume or export. The base's original 4K model configuration is
retained until the context recipe is qualified.

On MN5, `cache_midtrain_mn5.sbatch` imports the three contexts as independent CPU
array tasks (`ehpc1048`, `gpp`, `gp_debug`) and then runs each loader smoke. This
requires no GPU allocation. The target 9B GPU canary and final context curriculum
are separate from the portable token export and the tiny CPU correctness check.
For parallel full imports use `--qos=gp_ehpc`: debug QoS permits only one job per
user. The final cache qualification uses `MIDTRAIN_HANDOFF_ROOT=.../handoff_v3`,
`MIDTRAIN_CACHE_ROOT=.../cache_v3` and `MIDTRAIN_LOADER_SMOKE_ROOT=.../loader_smoke_v3`.
Three independent CPU imports can also share one allocation; the recorded MN5
`import_all_v3.sbatch` runs the same three entrypoints concurrently. The GPU canary
uses `handoff_v2`, whose training code and synthetic cache are unchanged in V3.

Full MN5 imports and all three loader smokes passed in job `47456439` (4m38s).
Counts match the immutable release; each target loader report is identical to
its JURECA counterpart. Native finite packed lengths are:

| Context | Chunks | Seen tokens | Trainable tokens | Packed sequences |
|---|---:|---:|---:|---:|
| 16K | 190,524 | 499,918,595 | 489,184,884 | 38,791 |
| 32K | 352,809 | 999,867,143 | 973,966,082 | 38,825 |
| 64K | 680,808 | 1,999,735,426 | 1,935,346,645 | 41,554 |

The packed lengths include padding and the packer's 64-segment bound; use these
finite lengths when setting optimizer steps and padding the last batch in step 6.

The CPU checks passed all three contexts, each with 50 audited rows covering
49 active pools and a continuation. They produced 23 / 23 / 22 packed examples.
The tiny full-vocabulary Qwen3 test performs two real Adam updates, verifies
nonzero gradients for all six added embedding/head rows, checks bit-identical
checkpoint resume, and reimports its HF export with exact model leaves. The full
native trainer also completed a fresh run and resumed from its saved checkpoint.
Resume exposed and fixed a separate vocabulary issue: the trainer's resume
exemplar must use the frozen 262272-row model axis while the tokenizer remains
262151 entries. The Prelude-only hook enforces this for both fresh and resumed
runs and rejects meshes requiring any further vocabulary padding. Source model
rows are never reinitialized by the hook.

`trainer_smoke_mn5.sbatch` uses one four-H100 node, data-axis parameter sharding,
and the already qualified 128-token synthetic mixed-format cache. It performs a
short real 9B run, then starts a new process to resume and export again. These
synthetic smoke updates do not consume the frozen production allocations and do
not qualify 16K–64K model attention, the final RoPE settings, or the production
curriculum's finite-batch launcher; those remain step 6 gates.
This canary passed in job `47456121` (34m34s); independent final read-back
`47456853` passed in 36s. The recorded fresh/resumed losses are 5.7836275 /
5.2237182. Native resume restored 101.64 GiB of model/optimizer state and continued
from step 2. Both exports preserve all 399 tensor names/shapes / 9,094,607,872
parameters, every weight is finite, and all eight tokenizer hashes, vocabulary
262272 and EOS IDs 2/262146 match the frozen contract. All six added embedding
and head rows changed. The final canary artifacts are under
`smoke_9b_gpu/{checkpoints,hf}/step-2` on MN5.
The current image reports that Transformer Engine is absent and falls back to
reference attention. A fused backend and measured 16K–64K memory/runtime are
required gates for the production context recipe.

`python verify_midtrain_hf.py --root /gpfs/scratch/ehpc532/ot/prelude_midtrain_v1 --step 2`
independently reads the canary's BF16 HF export on CPU. It compares every tensor
name and shape with the initialized base, scans every weight for NaN/Inf, checks
the padded vocabulary/EOS IDs and all eight frozen tokenizer hashes, inspects
changes in the six added embedding/head rows, and checks native checkpoint metadata.
Reports are saved under the MN5 root's `target_evidence/`.

## Evidence

`step5_evidence.json` records completed jobs, counts, checksums, CPU loader/training
checks, transfer verification and any outstanding target checks. The issue remains
in progress until its target training/evaluation gates are satisfied. Source data,
step 1–4 snapshots and the downstream SFT v1 release are preserved.
The immutable release manifest records the initial token-export state; subsequent
loader and canary qualification is recorded in the separate evidence file while
the release identity and checksums remain fixed.
