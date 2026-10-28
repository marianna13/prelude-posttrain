# Prelude retained-token census and mixture (issue 0054, step 4)

Artifacts live under
`/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step4`.
Step 3's verified canonical corpus and frozen train/validation assignments are
the inputs. This step counts and selects data; it does not run model training or
build the final token-ID release.

## Completed run (2026-10-17)

Step 4 completed on JURECA under `synthlaion`. All **169,315,668 canonical rows**
were counted again: 529,343,825,430 seen / 525,737,061,846 trainable tokens before
the replay/PDF eligibility cuts. All 8,932 metric parts passed checksum/footer,
source/split coverage, token accounting and stage accounting verification.

| After replay/PDF eligibility | Rows | Seen tokens | Trainable tokens |
|---|---:|---:|---:|
| Training | 166,594,212 | 512,064,060,367 | 508,494,258,155 |
| Held-out validation | 1,665,678 | 5,118,310,528 | 5,082,404,934 |

The cuts exclude 536,439 replay candidates over 4K and 519,339 long-PDF
candidates under 32K. Per-stage preferred minima further restrict the sampled
pools. Canonical step-3 files and validation assignments are unchanged.

The frozen recipe is **`mixture_v3/mixture_manifest.json`**, status
`step4-complete`, build ID
`7239c4ca51352ad5bafc4565cb0675b1ebdb73549fcf92c37401c4f71f493f1b`,
SHA256 `a03d76a897805785add51d8f93cfd3eb1c38dbd95353a8acdf30ce5931beddc6`.
It selects **1,207,832 unique training parents** across all three stages:

| Context | Parents | Actual seen tokens | Actual trainable tokens | Document share, seen / trained |
|---|---:|---:|---:|---:|
| 16K | 186,595 | 499,918,595 | 489,184,884 | 64.9952% / 66.2029% |
| 32K | 347,764 | 999,867,143 | 973,966,082 | 64.9972% / 66.5708% |
| 64K | 673,473 | 1,999,735,426 | 1,935,346,645 | 64.9979% / 67.0572% |
| **Total** | **1,207,832** | **3,499,521,164** | **3,398,497,611** | |

Whole-parent selection undershoots the 3.5B envelope by 478,836 tokens (0.0137%).
Every category is within 0.004 percentage point of its seen-token target in
each stage. All source and format tolerances passed. No parent repeats across
stages. The maximum effective epoch is 0.5332 over an entire eligible training
pool and 1.0 over a context-qualified pool. Exact per-source/per-stage epochs
are in `source_exposure_report.json`.

The pilot (`15719041`) checked 440 rows across 55 source/subset groups. Initial
census `15719043` preserved 8,929 completed parts / 169,090,857 rows before
bounded completion `15719065` finished 224,811 Terminal conversations in 221
disjoint ranges (four nodes, 64 workers each, 1m24s). Finalizer `15719066`
completed in 2m16s. Balanced mixture `15719080` completed in 7m11s; independent
selection audit `15719081` passed in 20s. All **16 regression tests** pass.
The [execution evidence](step4_evidence.json) pins counts, weights, runtime,
hashes, job accounting and verification results.

## Census and eligibility

Every one of the 169,315,668 canonical survivors is tokenized again. Counts use
the exact step-2 tokenizer/template and masks: document BOS 1 is masked, native
EOS 2 is trained; conversations train ordinary targets and mask entire semantic
tool-response groups. Earlier reasoning, tool calls and root tools remain in
the representation. Source-availability token counts from step 1 are not reused.
The guarded LF counter is equivalent to native document tokenization, including
code whitespace, Unicode and embedded special tokens. Conversations follow the
unchanged production template and mask semantics with native offsets.

The bounded completion helper replaces repeated prefix renders and per-tool
scans over every token with an equivalent linear pass. It instruments the two
known Jinja tool-group boundaries with markers absent from the original text,
removes those markers, requires exact rendered-text equality for every row, and
uses the same native offsets/alignment guards. Regression fixtures compare full
ID/mask vectors, including root tools, earlier reasoning, consecutive/empty tools
and boundary-looking payloads. Real full-unit parity `15719064` passed all 2,767
conversations / 39,242,872 tokens, producing identical counts and Parquet bytes.
The helper code/plan, child checksums and parity evidence are separately pinned
in finalization. Only homogeneous parts are split, preserving emission order.

The census retains metrics for both training and held-out validation rows.
Replay eligibility is at most 4,096 rendered tokens, including BOS/EOS. The
separate long-PDF source requires at least 32,768 rendered tokens. Ineligible
canonical candidates remain recorded with their reason; they are not silently
included in source availability. A parent receives exactly one mixture category.
Specialized QA/reasoning sources remain raw documents in the format accounting.
Selected Stack code and long repositories are disjoint categories.

Metrics include ID, source, pool, category, format, split/group, replay language,
seen and trainable lengths, masked counts, length eligibility and stage counts.
Empty work units still produce checksum-verified, zero-row metric parts. The
separate finalizer handles those parts and checks worker versus SQL totals,
canonical source/split coverage and stage accounting.

## Fixed targets and source capacity

| Category | Seen-token target |
|---|---:|
| Reasoning / math / science | 30% |
| Code / agent | 25% |
| Chat / instruction following | 10% |
| Long documents | 12.5% |
| Short replay | 22.5% |

The final format allocation within each category is:

| Category | Documents | Conversations |
|---|---:|---:|
| Reasoning | 14.526178% | 15.473822% |
| Code / agent | 15.473822% | 9.526178% |
| Chat / IF | 0% | 10% |
| Long documents | 12.5% | 0% |
| Replay | 22.5% | 0% |
| **Total** | **65%** | **35%** |

Format targets are independently **65% documents / 35% conversations** over the
same seen tokens. Trainable shares are reported separately because tool masks
and continuation context reduce the number of trained targets.

The builder minimizes relative squared deviation from explicit source priors
while enforcing both dimensions and conservative one-pass capacity constraints.
The strictly convex objective removes arbitrary allocation ties. Ultra language
ratios are fixed to clean availability at the common 32K original-length
threshold, including when extra weight moves into Ultra. The replay proxy
preserves step 1's family proportions; the 23 EU
language weights are recomputed from clean, training-only eligible tokens.
Unknown sources, nonfinite/negative weights and a non-unit sum are errors.
Missing length buckets and infeasible targets are reported; no natural/uniform
fallback or silent repetition is allowed.

| Context stage | Planning budget (seen tokens) | Preferred original length |
|---|---:|---:|
| 16K | 0.5B | At least 8K |
| 32K | 1.0B | At least 16K |
| 64K | 2.0B | At least 32K |

This freezes the upper 3.5B-token data envelope from the original 3–3.5B range;
it does not authorize training. Category, format and per-source weights stay
fixed across stages. Short replay is explicitly exempt from the preferred
minimum at every stage. The strict first attempt applies the minimum to all
other sources. Measured training chat has only **38,613,004 tokens** at least 32K
long and IF has none, versus the 64K stage's 200M-token chat/IF allocation. The
separately identified `--allow-short-chat` recipe therefore retains short whole
chat/IF conversations. That 10% allocation and 22.5% replay are explicit minimum
length exceptions; at least 67.5% of seen exposure comes from parents meeting the
preferred length. Short exceptions do not count as long-range evidence.

Three pools have no 32K-qualified originals, so their frozen cross-stage weights
are zero: selected short code repositories (`stack-v3-train/data/code`),
Scientific-Coding, and Cascade conversational-agent. Scientific-Coding also has
no 16K-qualified originals. Their canonical candidates/counts remain available;
this recipe does not introduce additional short-source exceptions. Long Stack
repositories and long PDFs each retain 6.25%. Full weights for all 52 pools,
including 11 Ultra languages and 23 EU replay languages, are in the manifest.

## Continuations and deterministic sampling

The virtual chunk policy retains every original token. The first chunk contains
at most the stage context length. Each continuation has a new masked BOS 1,
256 preceding original tokens as masked context, and as many new tokens as fit.
Only the original final EOS is retained; no intermediate EOS is invented.
Original tool-response loss bits are unchanged. Keep short final tails. Related
chunks retain the original split/group and source. For `n` chunks, seen exposure
is `original_tokens + (n-1)*257`; trainable exposure is unchanged.
Step 5 must implement and verify these exact counts with actual token IDs.

The seed is `prelude-midtrain-v1-mixture-20261017`. Parents are ordered within
each pool by `sha256(seed || canonical_id)`, with ID as the tie break. Reserve
64K parents first, then 32K, then 16K, so early training stages cannot consume
scarce later-stage examples. Training order remains 16K, 32K, 64K. Selection
consumes whole parents against **seen-token quotas**, skips parents larger than
the remaining quota, and never reuses a parent across stages. Exhaustion is an
error. Whole-parent selection keeps original targets intact and permits a small,
reported budget undershoot rather than silent truncation.

The audit requires actual category/format shares within 0.1 percentage point,
each source share within 0.02 percentage point, and budget undershoot below
0.2%. It independently checks emitted selection IDs against census metadata,
requires train-only rows and global parent uniqueness, and reports seen/trained
shares and effective trained epochs for every source. Tokenized packing and the
Levanter loader still need their own step-5 checks.

## Reproduction and artifacts

Use the frozen `tools/` snapshot for this build. Changes to input curation,
tokenizer, masks or counting code require a new census build. Mixture identity
also includes census checksum, weights/length policy, seed, planning budgets,
builder code and dependency versions. Never resume changed semantics into an
existing output directory.

For a fresh rebuild, choose an empty output directory. The following defers the
three large Terminal parts, allowing the ordinary census allocation to finish
before bounded completion. It requires the existing frozen step-2/3 inputs and
the Python environment/dependencies referenced by the scripts.

```bash
PY=/p/project1/ccstdl/envs/marianna/lmeval/bin/python
BASE=/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1
OUT="$BASE/step4-rebuild"
export MIDTRAIN_OUT="$OUT"
"$PY" -I rl/scripts/data/census_curated_prelude_midtrain.py prepare \
  --out "$OUT" --step3 "$BASE/step3" --step2 "$BASE/step2"
cp rl/scripts/data/finish_prelude_retained_census.py \
   rl/scripts/data/finalize_prelude_midtrain_census.py \
   rl/scripts/data/freeze_prelude_midtrain_mixture.py "$OUT/tools/"
KEYS=$("$PY" -c 'import json,sys; print(",".join(t["key"] for t in json.load(open(sys.argv[1]))["tasks"] if t["key"] not in {"core-1488","core-1490","core-1492"}))' "$OUT/census_input_manifest.json")
CENSUS_JOB=$(MIDTRAIN_CENSUS_ARGS="--keys $KEYS" sbatch --parsable \
  --output="$OUT/census-%j.out" rl/sft/prelude/census_retained_midtrain.sbatch)
# This allocation waits for the ordinary census, then checks a complete real unit.
USE_TORCH=0 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=2 \
  srun --dependency="afterok:$CENSUS_JOB" --account=synthlaion --partition=dc-cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=00:30:00 \
  "$PY" -I "$OUT/tools/finish_prelude_retained_census.py" parity \
  --out "$OUT" --part core-1496 --workers 4
"$PY" -I "$OUT/tools/finish_prelude_retained_census.py" prepare --out "$OUT"
FINISH_JOB=$(sbatch --parsable --nodes=4 --output="$OUT/bounded-%j.out" \
  rl/sft/prelude/finish_retained_midtrain.sbatch)
FINALIZE_JOB=$(sbatch --parsable --dependency="afterok:$FINISH_JOB" \
  --output="$OUT/finalize-%j.out" rl/sft/prelude/finalize_retained_midtrain.sbatch)
MIDTRAIN_MIX_OUT="$OUT/mixture_v3" MIDTRAIN_MIX_ARGS=--allow-short-chat \
  sbatch --dependency="afterok:$FINALIZE_JOB" --output="$OUT/mixture-%j.out" \
  rl/sft/prelude/freeze_midtrain_mixture.sbatch
```

For the completed production build, use `step4/tools/` for census/completion and
`step4/mixture_tools_v3/` for mixture reconstruction. The production census
snapshot predates the repository's empty-unit guard in the convenience
`summarize` command; production used the separate verified finalizer throughout.
Changes to scheduling utilities produce a new build identity even when the
rendering, token counts and selected parent IDs are unchanged.

Verify/resume the completed production mixture with its exact frozen builder:

```bash
OUT="$BASE/step4"
MIDTRAIN_OUT="$OUT" MIDTRAIN_MIX_OUT="$OUT/mixture_v3" \
MIDTRAIN_MIX_ARGS=--allow-short-chat \
MIDTRAIN_MIX_SCRIPT="$OUT/mixture_tools_v3/freeze_prelude_midtrain_mixture.py" \
  sbatch "$OUT/mixture_tools_v3/freeze_midtrain_mixture.sbatch"
```

Run the 16 regression checks from the repository root:

```bash
PYTHONPATH=/p/project1/laionize/marianna/dc_agent/.prelude_census_deps:rl/scripts/data \
USE_TORCH=0 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false \
  "$PY" -m unittest rl/scripts/data/test_census_curated_prelude_midtrain.py \
  rl/scripts/data/test_finish_prelude_retained_census.py \
  rl/scripts/data/test_freeze_prelude_midtrain_mixture.py -v
```

- `census_input_manifest.json`: pinned canonical files, tokenizer, implementation,
  runtime, counting/length policy and census build identity.
- `census/<work-unit>/{metrics.parquet,stats.json}`: exact retained lengths, masks,
  split/source metadata, exclusions, stage availability and checked task hashes.
- `retained_census.json`: complete aggregation, length bins, source/split coverage,
  metric checksums and finalizer identity.
- `mixture_v1/feasibility.json`: strict-length infeasibility evidence; the 32K
  chat/IF pool supports at most a 1.93065% fixed share at the 64K stage.
- `mixture_v3/feasibility.json`: final source capacities, explicit priors and
  balanced constrained-weight solution.
- `mixture_v3/mixture_manifest.json`: final weights, effective epochs, stage
  budgets/availability, actual seen/trained shares and selection checksums.
- `mixture_v3/selection-{16384,32768,65536}.parquet`: train-only parent IDs, source,
  split group and exact virtual exposure for the step-5 export.
- `source_exposure_report.json`: effective epochs over both full eligible
  training pools and context-qualified pools.
- `selection_audit.json`: independent read-back verification of every selected
  parent, checksums, seed order, chunk accounting, eligibility, source/category/
  format shares, effective epochs and language balance.

These artifacts preserve all upstream provenance and decontamination pins through
the referenced step-3 manifest. They are a data recipe and parent selection, not
an exported training dataset. Loader smoke/resume/export and training gates
remain steps 5–6.
