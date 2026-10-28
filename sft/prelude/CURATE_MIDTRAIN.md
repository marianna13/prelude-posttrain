# Prelude mid-training curation (issue 0054, step 3)

The step 3 run uses JURECA CPU nodes under `synthlaion`. `csjsc` is not a listed
association for this login. Artifacts live under
`/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step3`.
Use `curation_manifest.json` to determine full-run completion; a pilot manifest
or submitted job does not establish a completed corpus release.
The final checked tools are saved under `production_tools_v3/`, including their
checksums and runtime versions. To resume after repository changes, set
`MIDTRAIN_REPO="$MIDTRAIN_OUT/production_tools_v3"` for the batch entrypoints.

## Completed run (2026-10-17)

The full run scanned 199,371,691 source rows, accepted 197,537,512 through the
frozen adapters, and retained **169,315,668 canonical rows**: 167,639,429 training
and 1,676,239 validation. All 8,932 parts (566,043,995,688 bytes) passed checksum,
row-count, unique-ID, split-separation, schema, version and decision-agreement
checks. Final verification job `15718887` completed successfully in 52m28s under
`synthlaion`. Eight regression checks pass.

`curation_manifest.json` records status `step3-complete` and build ID
`128263cb0e97dcc66720515095d5c6a4b5f724511ed8cc081ba60aa9e58bb6d1`.
Its SHA256 is `852dfef3a912d2270a97c9ed0319c2f5049d1ea8e5800c24ee6fff0ccc3d463d`.
The [execution evidence](step3_evidence.json) contains the source/removal/split
counts, tool/runtime pins and successful production jobs.

These are canonical candidates. [Step 4](MIXTURE_MIDTRAIN.md) completed the fresh
token census, long-PDF/replay eligibility cuts, virtual chunk policy, fixed weights
and verified parent selections. Token-ID chunk export, the loader smoke run and
training remain steps 5–6. Fuzzy deduplication and model training have not run.

## Execution

[`curate_prelude_midtrain.py`](../../scripts/data/curate_prelude_midtrain.py)
pins the step 2 input/adapter inventory, the 126 SFT v1 reference parts, benchmark
index, source code and native scanner. Its 8,932 source work units partition the
same candidate pool, splitting replay Parquet files into bounded row groups.
Both permitted code license groups remain eligible. AgentTrove, RTS/TMax teacher
traces, WikiRewrite, and Cascade `terminal_agent` are excluded from the inventory.
The older AgentTrove mid-training proposal is superseded by this frozen selection.

Workers read source payloads locally and write compact IDs, exact hashes,
provenance locators and benchmark-hit evidence. They do not tokenize or render.
Global reduction shuffles this metadata; the second source pass reconstructs the
full canonical rows for retained IDs, checks their original record hashes, and
writes `curated/<task>/rows.parquet`. Reasoning, tool data, code whitespace and
source provenance survive this pass.

The native scanner uses the same Unicode lowercase/whitespace word semantics,
13-word windows, MD5 first eight bytes and big-endian hash integers as MV. The
original 5,185,587-hash index is unsorted: a separate sorted, unique view preserves
its exact membership, and both original/derived file hashes are pinned. Only
matched hashes cross the native/Python boundary. All distinct matches are retained,
including low-diversity matches; there is no 50-hit cap. Tests compare exact
hashes, 13-word strings and diversity flags against Python, including Unicode,
empty/repeated inputs and more than 1,024 matches in one record.

On JURECA, `dc-cpu` nodes have about 512 GB RAM; the full scan used four nodes,
64 single-process workers per node. Global reduction can use `dc-cpu` or
`dc-cpu-bigmem`; this run used a standard CPU node with a 256 GB DuckDB memory
limit for global operations.
Disjoint ranks can also run on JUWELS after verifying mounts, dependencies and
native-library compatibility. Use the same manifest and rank/world assignment;
never independently deduplicate each cluster and concatenate its survivors.
Each rank must have one active owner; stop the old scan before changing the world
size or reassigning its ranks to another cluster.

## Identities and retention policy

- Conversation task key: exact UTF-8 SHA256 of the first genuine nonempty user
  message. System/assistant turns and semantic tool/environment responses do not
  define the task. No case folding, whitespace collapse, stripping, punctuation
  changes or Unicode normalization is applied to task/code deduplication keys.
- Explicit upstream task/problem/question IDs add source-scoped alias edges.
  Terminal names include the original dataset source, so two datasets' `task_1`
  names do not collide. Connected components link alternate prompts for a known
  task before SFT exclusion, representative selection and validation assignment.
- The SFT reference is exactly the rows present in `prelude_sft_v1/rendered_clean`.
  Their one-based per-part renderer IDs identify the corresponding normalized
  rows. Both normalized and cleaned input file checksums are recorded. The full
  reference scan accounts for all 4,412,306 retained SFT rows.
- Documents/files use exact UTF-8 SHA256. Selected Stack repositories retain
  eligible files, and UltraData files already contained byte-for-byte in retained
  repositories are removed. Content IDs supply extra overlap/conflict evidence;
  a reused ID with different bytes does not justify removal.
- Deduplication follows decontamination/SFT exclusion, so a removed Stack parent
  cannot suppress its otherwise usable UltraData files. Source priority is Stack,
  UltraData, Specialized, then replay. Within UltraData, higher finite quality and
  algorithmic relevance win; stable ID breaks ties. Conversation ties use stable
  ID. Removed IDs, retained representatives and source/subset counts persist in
  partitioned decision Parquet.

## Benchmark evidence and review

Scan documents and every conversation role, including masked tool outputs,
earlier reasoning, structured tool calls and root tool definitions. Scan fields
separately to avoid creating 13-grams across unrelated message boundaries.

Persist `raw_hits` and `eligible_hits` per record, global document frequencies and
example 13-grams. Windows with fewer than seven distinct words are retained as raw
evidence but are not sufficient to remove a record. Frequency alone never excludes
a benchmark hash. In particular, SFT's absolute `max_df=20` rule is not reused.

`boilerplate_review.json` lists the top 1,000 candidates for inspection; complete
statistics are in `hit_review.parquet`. A reviewed, versioned exclusions JSON must
name every excluded hash, exact 13-gram and reason. Common exception/stdlib trace
fragments can be whitelisted with evidence; question stems, constraints and actual
benchmark solutions remain evidence. One remaining eligible hash removes the
record. Raw evidence remains available after this decision.
The summary's `example_id`, `example_source` and `example_field` are independent
minima across samples, so they are not a joined provenance tuple. Use the matching
`scan/*/samples.parquet` row for attributable record/field examples.

The full-corpus [reviewed policy](benchmark_exclusions_v1.json) explicitly excludes
290 inspected 13-grams: 206 generated Sphinx configuration/documentation comments,
38 standard Python traceback fragments, seven standard Python/NumPy diagnostics,
and 39 generic issue-template instructions. All 1,000 candidates were inspected;
unlisted matches remain evidence. Mathematical expressions, algorithm/solution
fragments, number sequences, benchmark instructions and question stems remain
eligible even when frequent. This conservative whitelist is not an exhaustive
classification of all matching hashes.

The exclusions file must have `scan_build_id`, `status: "reviewed"`,
`scope: "full"` (or `"pilot"`), `frequency_cutoff: null`, and an `excluded_hashes`
list of `{h, ngram, reason}` objects. The reducer verifies the text/hash pairs and
copies this exact policy into the decision build. Pilot policies cannot be applied
as full-corpus review policies.

## Validation and verification

Reserve 1% by a fixed hash of task component, canonical GitHub repository, or exact
document identity. Repository chunks and repeated task identities stay together.
Assignments are persisted before packing. An exact duplicate receives a single
representative/category. This is exact deduplication and known-index
decontamination; fuzzy/semantic deduplication is not performed.

The finalizer checks every output checksum and row count, unique retained IDs,
agreement with frozen keep/drop decisions, canonical payload schema, and absence
of train/validation group overlap. Parquet metadata is cached across the global queries to avoid reparsing millions
of small row groups. The completion manifest records the verification memory,
thread and checksum-worker settings. Downstream token recounting, eligibility and
mixture selection completed in [step 4](MIXTURE_MIDTRAIN.md); token-ID chunk export,
loader verification and training remain steps 5–6.

## Commands

Run from the repository root. Preparation creates a new frozen build; code changes
require a separate directory. Resume reuses only completed, checksum-verified units.

```bash
PY=/p/project1/ccstdl/envs/marianna/lmeval/bin/python
OUT=/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step3
STEP2=/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step2
mkdir -p "$OUT/native"
/usr/bin/g++ -O3 -std=c++17 -shared -fPIC -Wno-deprecated-declarations \
  rl/scripts/data/prelude_ngram_scan.cpp -lcrypto -o "$OUT/native/ngram_scan.so"
"$PY" -I rl/scripts/data/curate_prelude_midtrain.py prepare --step2 "$STEP2" --out "$OUT"

# Small allocation first: both reference and source pilots.
"$PY" -I rl/scripts/data/curate_prelude_midtrain.py run --out "$OUT" --stage reference --pilot --workers 8
"$PY" -I rl/scripts/data/curate_prelude_midtrain.py run --out "$OUT" --stage scan --pilot --workers 8
# Production reference scan on one CPU node; source scan on four nodes.
"$PY" -I rl/scripts/data/curate_prelude_midtrain.py run --out "$OUT" --stage reference --workers 16
MIDTRAIN_WORKERS=64 sbatch -A synthlaion --export=ALL \
  --output="$OUT/scan-%j.log" rl/sft/prelude/curate_midtrain.sbatch

# After all source/reference units complete, on a node with sufficient memory:
"$PY" -I rl/scripts/data/reduce_prelude_midtrain.py summarize --out "$OUT" --verify-bytes --memory 256GB --threads 64
# Inspect hit_review.parquet and write reviewed exclusions with this scan_build_id.
"$PY" -I rl/scripts/data/reduce_prelude_midtrain.py decide --out "$OUT" \
  --exclusions "$OUT/benchmark_exclusions.json" --memory 256GB --threads 64
# Run disjoint ranks on CPU allocations; WORLD is the total rank count.
"$PY" -I rl/scripts/data/reduce_prelude_midtrain.py materialize --out "$OUT" --rank 0 --world 4 --workers 32
"$PY" -I rl/scripts/data/finalize_prelude_midtrain_step3.py --out "$OUT" --memory 256GB --threads 64
```

[`reduce_midtrain.sbatch`](reduce_midtrain.sbatch) runs these later passes on
JURECA. Global operations require one task; materialization partitions work across
all allocated ranks:

```bash
MIDTRAIN_OPERATION=decide sbatch -A synthlaion --export=ALL \
  --output="$OUT/decide-%j.log" rl/sft/prelude/reduce_midtrain.sbatch
MIDTRAIN_OPERATION=materialize MIDTRAIN_WORKERS=32 sbatch -A synthlaion \
  --nodes=4 --export=ALL --output="$OUT/materialize-%j.log" \
  rl/sft/prelude/reduce_midtrain.sbatch
MIDTRAIN_OPERATION=finalize MIDTRAIN_THREADS=64 sbatch -A synthlaion \
  --export=ALL --output="$OUT/finalize-%j.log" rl/sft/prelude/reduce_midtrain.sbatch
```

Submit each pass only after its prerequisite succeeds, or use Slurm `afterok`
dependencies. Preserve `benchmark_exclusions_v1.json` under the output root as
`benchmark_exclusions.json` for this frozen build. A new scan build requires its
own review policy and output directory.

For large decision exports, the optional
[`finish_prelude_midtrain_decisions.py`](../../scripts/data/finish_prelude_midtrain_decisions.py)
completes checksums in parallel. Run it only after the frozen reducer has written
`reduction/benchmark_exclusions.json` and the complete split export. It recounts
every decision part, reconciles accepted/retained totals and reconstructs the
identical decision manifest; a regression fixture checks exact manifest bytes.
Reconstruction may depend on this verified completion job while the original
serial checksum tail finishes. `parallel_completion.json` records the helper
hash and resulting manifest checksum for the final parity check.
For this run, the completed original reducer and parallel helper produced the
same full-corpus manifest bytes, not only the same regression-fixture output.

```bash
"$PY" -I rl/scripts/data/finish_prelude_midtrain_decisions.py --out "$OUT" --workers 64
```

Add `--pilot` consistently to reduction/materialization/finalization commands for
the bounded pilot. Its manifest always records that full deduplication and
decontamination are incomplete.
