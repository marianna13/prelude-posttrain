---
title: "Build Prelude 9B mid-training data for reasoning, code, and 64K context"
labels: [agent-generated, data, sft]
status: in-progress
gh_number: 60
created: 2026-10-16
---

## Goal

Turn the downloaded mid-training sources into a reproducible, decontaminated mixture for OELLM Prelude 9B.
Use it to extend context from 4K through 16K, 32K, and 64K while preserving the selected base's math and code
ability. Deliver the source adapters, loss masks, cleaned shards, token census, mixture manifest, and a training
recipe with a smoke run and evaluation gates.

Steps 1–5 are complete: source inclusion/census, adapters/masks/tokenizer compatibility, full canonical
curation, retained-token census and frozen mixture, portable token export, full read-back,
MN5 transfer/native caches, CPU/target loader checks, and the short 9B fresh/resume/export check.
Step 6 has run the prelaunch gates: infrastructure passes, but the 16K/32K/64K
context-quality checks fail. Production is held pending a revised RoPE/context
curriculum and adaptation pilot, with the existing dominant2 4K results retained
as the matched short-context reference.

Base: `prelude-9b-anneal300b-ml1-dominant2-iter0989075`, HF revision
`9d252ade085f74176eb2a3fa321a59f3b5af6deb`, at
`/p/data1/mmlaion/shared/models/openeurollm/prelude-anneal300b-ml1-dominant2-iter0989075`.
The [base-selection analysis](../evals/analysis/prelude-sft-base-selection/README.md) records why this checkpoint
was selected and contains the original mid-training/context-extension proposal.

## Current evidence (2026-10-16)

Sources live under `/p/data1/mmlaion/marianna/midtrain_sources/` (about 1.06 TiB of data files).

| Source directory | Data files | Local size | Observed contents |
|---|---:|---:|---|
| `Nemotron-Cascade-2-SFT-Data` | 9 JSONL | 375.3 GiB | math_notool, math_tool, math_proof, science, SWE agentic/agentless, conversational_agent, chat part 1, IF |
| `Nemotron-Pretraining-Specialized-v1` | 330 Parquet | 326.8 GiB | InfiniByte reasoning, RQA, STEM-SFT, textbooks, scientific coding, Wiki rewrite |
| `UltraData-Stack-v3-L2` | 255 Parquet | 254.5 GiB | 127,133,027 file rows across 11 languages |
| `stack-v3-train` | 300 Parquet | 122.7 GiB | 6,326,354 repository rows; selected shard list is `stack_v3_pick.txt` |
| `Nemotron-Terminal-Corpus` | 29 Parquet | 7.6 GiB | 366,154 conversations; all observed metadata is Terminus-2 / DeepSeek-V3.2, thinking enabled |

All 914 core Parquet footers were readable. Step 1 subsequently processed every row group and Cascade JSONL
range; complete counts and the 46 schema-invalid chat records are documented below.

Pinned source revisions from `midtrain_revs.json`:

| HF dataset | Revision |
|---|---|
| `nvidia/Nemotron-Cascade-2-SFT-Data` | `9f36020daf067f1a8b39336bf619fe30af30bb02` |
| `nvidia/Nemotron-Pretraining-Specialized-v1` | `9ed3718b5f2ae29074c5e34e64115432b7c4320f` |
| `openbmb/UltraData-Stack-v3-L2` | `d6bdb3e707cf17bbfb64067cd4c4ed84b0809baf` |
| `HuggingFaceCode/stack-v3-train` | `1f61b735bc0a5698345ce2196730f24bfa467f33` |
| `nvidia/Nemotron-Terminal-Corpus` | `a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9` |

The existing `prelude_sft_v1/rendered_clean` release is a separate SFT dataset. At the initial inspection,
no finished mid-training build, mixture configuration, or cleaned mid-training release was present.

## Reuse MV data tooling

Implementation reference: `/p/project1/laionize/marianna/mv/mv`, inspected at repository revision
`c8abab4cac73da35c2876a34d2837aac6ab37298`. Paths in the following table are relative to that checkout.
Reuse these drivers and execution patterns behind Prelude-specific adapters; keep the planned Levanter training
interface. Record the MV revision and hashes of the actual reused scripts in the release manifest.

| Component | Existing implementation | Reuse / adaptation for Prelude |
|---|---|---|
| Normalization | `orchestration/jobs/normalize.py`, `normalize_ray.py` | Canonical `id/text/messages/source/subset/split/meta` schema and chunked Ray processing; add source adapters that retain full conversation payloads and stable provenance |
| Labels, filters, categories | `orchestration/jobs/label_ray.py`, `filter_ray.py`, `bucket_ray.py` | Optional language/quality annotations, explicit rule filters, and disjoint priority buckets; preserve upstream source identity before assigning mixture categories |
| Exact deduplication | `orchestration/jobs/dedup_ray.py` | Shuffle compact hashes/IDs, then filter full rows locally; supply code-safe content keys and separate task/prompt keys |
| Benchmark decontamination | `orchestration/jobs/build_decontam_index.py`, `decontaminate_ray.py` | Existing index format/hash function and distributed scans; pass the Prelude index explicitly and preserve hit/exclusion evidence |
| Near-duplicate detection | `orchestration/jobs/fuzzy_dedup.py`, `fuzzy_remove.py` | Optional MinHash/LSH identify-then-remove pass after a measured sample; preserve the identification/removal file partitioning and ID map |
| Tokenization and census | `orchestration/jobs/tokenize_megatron.py`, `tokenize_merge.py` | Reuse CPU `srun` sharding and per-source token manifests; adapt outputs to Levanter and preserve conversation loss masks |
| Mixture weights | `orchestration/jobs/mix_weights.py` | Reuse explicit per-source weight/registry logic; validate against retained tokens and translate into the Levanter mixture configuration |
| Rebuild detection | `orchestration/jobs/mv_fp.py` | Reuse input-file freshness checks as a fast signal; add transform/config/tokenizer/index identity to the Prelude build key |

Important interface differences found in the implementation:

- MV's `_norm_turns` retains only `role` and `content`; it drops separate `reasoning_content`, `tool_calls`, and
  other message metadata. Preserve the original payload and loss-relevant fields in a keyed record alongside any
  flattened analysis text. Use kept/dropped IDs to select the original structured records after curation.
- The generic `clean_text` strips `<...>` tags and collapses spaces/tabs. Applied to training payloads, this would
  erase `<think>` / `<tool_response>` markers and alter code indentation. Keep code/conversation payloads intact;
  any cleaned analysis view must remain separate and must retain all content needed for decontamination.
- `normalize_ray.py` auto-detection calls `read_any` on the whole first file before taking 200 rows. Use explicit
  specs and bounded sample reads for the 50-111 GiB Cascade files. Its JSONL path also loses per-record paths,
  leaving subset/split blank; recover those from the explicit input manifest. Whitelist files and allowed splits.
- MV exact-dedup keys lowercase and collapse whitespace. Reuse the hash-only execution strategy with
  source-appropriate keys; preserve distinctions in case-sensitive code, indentation, and string literals.
- MV decontamination currently removes matching rows without the SFT script's boilerplate guards or separate
  prompt/answer hit report. Align and version that policy, including masked tool content. The driver accepts
  `DECONTAM_INDEX`; the current Dagster asset exports MV's default index constant, so a Prelude wrapper must pass
  `/p/data1/mmlaion/marianna/decontam/index_prelude_v1` explicitly or add a verified asset configuration field.
- MV tokenization emits Megatron `.bin/.idx` with no conversation loss-mask stream. It uses
  `add_special_tokens=False` and appends tokenizer EOS by default. Specify Prelude BOS/document EOS separately
  from ChatML turn EOS, and keep the `{ids, loss}` conversation export required by this plan.
- `mix_weights.py` defaults to random trials. Start with explicit frozen weights and reject missing/unknown
  source names, negative weights, or a zero sum; do not silently fall back to natural or uniform sampling.
- `mv_fp.py` fingerprints input paths, sizes, and integer mtimes, rather than transformation settings or index
  contents. Changing masks, normalization, filters, the tokenizer, or the benchmark index must invalidate outputs
  even when the corpus files are unchanged.

For a standalone build, reuse the cluster drivers through wrappers with explicit input/output paths. Keep this
release under the proposed Prelude output root, separate from existing `/p/data1/datasets/mmlaion/mv_pipeline/`
outputs. If integrated into Dagster later, preserve its per-dataset curation / global composition / training phase
boundaries and named composition paths. Semantic deduplication remains opt-in; no embedding-based removal is
part of the initial recipe. Operational reference: MV's `orchestration/dagster/AGENTS.md` and `README.md`.

## 1. Freeze source inclusion and measure available tokens

Completed 2026-10-16. The decision manifest is
`/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step1/source_selection.json`, status `step1-complete`, SHA256
`7eeb1ab0bfff8f0573ecb53a753faa2b3a302422109839858f8cc035e6803909`. It includes **both `permissive` and `no_license` code files**, excluding vendor files.
Overall issue status remains `in-progress`: the tokenized release, loader smoke run and training are
later steps. The retained-token census and mixture weights are complete in step 4 below.

The frozen core inventory covers 923 data files (1,167,140,630,598 bytes), including exactly 300 selected Stack
shards and Cascade chat part 1. All 7,642 work units completed. All 5,964 Parquet row-group record counts match
footers. All 923 files passed SHA256 verification; 919 also match upstream LFS checksums, while four are pinned
by local SHA256 where download checksum metadata is absent. Replay/PDF inputs add 49 checksum-verified files,
for **972 pinned source data files** altogether.

Counts use the frozen Prelude tokenizer and reference representation: documents have original BOS/EOS IDs 1/2;
conversations use the pinned ChatML template, retain earlier reasoning, and train all targets except the initial
BOS and defined tool-response bodies/wrappers. Role headers remain trainable. These are source-availability
counts before production normalization, deduplication, decontamination, and sampling.

| Core collection | Valid records | Seen tokens | Trainable tokens |
|---|---:|---:|---:|
| `Nemotron-Cascade-2-SFT-Data` | 13,548,623 | 126.285B | 122.368B |
| `Nemotron-Pretraining-Specialized-v1` | 60,655,840 | 297.379B | 297.319B |
| `Nemotron-Terminal-Corpus` | 366,154 | 7.035B | 4.928B |
| `UltraData-Stack-v3-L2` | 127,133,027 | 229.687B | 229.560B |
| `stack-v3-train` | 6,326,354 | 165.062B | 165.056B |
| **Total** | **208,029,998** | **825.448B** | **819.230B** |

All 46 rejected records are Cascade chat rows with invalid role strings containing embedded prompt/template
text; they are valid JSON with invalid conversation schemas. Their byte offsets and reasons are retained.
Specialized Wiki Rewrite was measured (8.036B tokens) and excluded from inclusion; the five retained Specialized
subsets provide 289.344B tokens. Cascade `terminal_agent` is excluded to avoid duplicating Terminal-Corpus;
AgentTrove/RTS/TMax remain reserved for downstream SFT.

**Repository selection.** Require at least 8,192 eligible rendered tokens, at least one exact UltraData-L2
`(repository, content_id)` match, and L2 file density of at least 20%. Deduplicate L2 evidence before joining;
rank by density × mean L2 quality, then stars and stable repository/commit/record IDs. Preserve original file
paths, content IDs, commits, and deterministic `(file_path, content_id)` order.

| License policy | Selected repositories | Selected tokens |
|---|---:|---:|
| Permissive only | 3,439 | 263,167,023 |
| **Permissive + no_license (chosen)** | **128,122** | **5,858,963,541** |

The chosen pool contains 4,432,962 files, including 1,754,875 L2-matched files. Its disjoint categories are
93,498 code repositories / 1,479,068,404 tokens and 34,624 long-document repositories / 4,379,895,137 tokens.
Including both license groups adds 5,595,796,518 selected tokens in this comparison. Density is recomputed within
each eligible license pool, so the selected repository sets need not be nested.

Selected-repository L2 density has median 32.97% and p90 66.67%; mean L2 quality has median 5.33 and p90 6.78.
Leading file languages are Java (1,525,507), C++ (767,008), Python (748,143), and JavaScript (313,312).
These are file-language counts, not whole-repository token-language shares. Full score/language distributions,
license/vendor impact, every selection gate's counts, length percentiles, and six length bins are saved for both
policies. The largest selected repository is 26,989,841 tokens; overlength content still needs review/chunking.
UltraData remains an eligible file pool across all 11 languages: 212,664,823,421 nonvendor tokens versus
11,876,129,637 under permissive-only eligibility. Overlap with selected repositories was resolved in step 3.

**Replay and long PDFs.** The native Leonardo/LUMI blend is unavailable here. Freeze an explicitly bounded local
proxy: 16 SHA256-ranked DCLM files × first 100,000 rows, eight high-quality original Nemotron-CC files × first
100,000 rows, one full English FinePDFs replay file, and the first 50,000 rows of one FinePDFs file in each of
23 EU languages. Replay keeps documents of at most 4,096 rendered tokens. This is a census of that cut, not of
the complete upstream datasets.

| Proxy family | Eligible tokens | Within-replay weight |
|---|---:|---:|
| DCLM English | 1,269,645,422 | 39.6851% |
| Nemotron-CC English, original/high | 437,133,357 | 29.2857% |
| FinePDFs English | 478,396,232 | 13.0292% |
| FinePDFs, 23 non-English EU languages | 997,098,220 | 18% |

Within the multilingual family, language weights follow eligible token counts. The proxy supports
1,492,650,487 tokens before any family repeats, before deduplication/decontamination. It replaces HPLT with
PDFs, restricts Nemotron-CC to high-quality original documents, and omits the native StarCoder/OLMo/math/
translation components. A local StarCoder aggregate was excluded because file-level provenance is insufficient.

A separate full English FinePDFs file supplies **1,883,780,632 long-document tokens from 18,322 documents** at
least 32K tokens long. English replay and long-PDF inputs use different files; semantic duplicates still need
removal. No arXiv download is part of this cut. Exact files, row limits, revisions where available, whole-file
SHA256 hashes, and language weights are in the replay manifest.

**Artifacts and implementation.** All paths below are relative to `/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step1/`:

- `source_selection.json`: final inclusion decision, license comparison, available pools, and artifact hashes.
- `census_v2/verified_input_manifest.json`: core files, checksums, schemas, tokenizer/template, runtime versions,
  source revisions, policy, and frozen adapter identity.
- `census_v2/census_summary.json`: source/subset totals, teacher/language/license distributions, percentiles, and bins.
- `census_v2/repo_selection/{all,permissive}.{json,parquet}`: selected IDs/commits, scores, filter funnels,
  language counts, and disjoint categories; `*_evidence.parquet` preserves repository-level match evidence.
- `census_v2/{metrics,stack_files,ultra_files}/`: keyed record/file metadata for subsequent materialization.
- `replay_v1/{verified_input_manifest,summary,length_summary}.json`: bounded replay and PDF pins/counts.
- `lf_count_validation.json`: guarded exact LF counting conditions, implementation hashes, native-ID comparisons,
  and a complete work unit with byte-identical statistics and file metadata.

Scripts: [census](../scripts/data/prelude_midtrain_census.py),
[CPU entrypoint](../sft/prelude/census_midtrain.sbatch),
[repository selection](../scripts/data/select_prelude_midtrain_repos.py),
[replay pinning](../scripts/data/pin_prelude_replay.py), and
[finalization](../scripts/data/finish_prelude_midtrain_step1.py).
[Bounded dispatch](../scripts/data/run_prelude_census_subset.py) and
[cache adoption](../scripts/data/merge_prelude_census_cache.py) share completed units with build/count checks.
The [LF counter](../scripts/data/prelude_lf_counting.py) uses only mathematically unmergeable LF boundaries
verified against the frozen tokenizer; it refuses conversation offsets. The original census adapter and rendered
representation are unchanged. Frozen copies and job settings remain under the output tree.

Initial scans hit the 512 GiB memory limit on oversized repositories; all completed work was reused. Ordinary CPU
scans handled bounded groups; the final two groups completed with guarded equivalent LF counting. The original
workers were released after all units were present. Aggregation `15717847`, both-policy selection `15717713`,
and finalization `15717714` all completed successfully. Ten regression tests pass, including exact native IDs on
108 Unicode/markup/boundary fixtures and selection-density/license checks. The 10.7M-token full-work-unit parity
check also passed. Reference counts must be recomputed after production transforms.

Terminal-Corpus uses later `user` turns for terminal output and parser feedback. The census maps these Terminus-2
environment responses to tool turns while retaining the initial task; step 2 must preserve this source rule.

- [x] Freeze exact input files, revisions/checksums, schemas, tokenizer/template, and adapter versions.
- [x] Complete streaming schema/record/token census, rejection accounting, teacher/language distributions,
      percentiles, and before/after filter bins.
- [x] Select repository-level code with L2 quality/density signals and preserved repository/commit/file provenance.
- [x] Resolve both code license groups, record retained-token impact, and preserve keyed license/teacher metadata
      for production normalization.
- [x] Pin replay sources and document the native-blend approximation, source weights, and language weights.
- [x] Pin additional long PDFs and measure eligible availability separately from replay.

## 2. Normalize sources and implement mid-training loss masks

Documents remain raw text with loss on all non-padding target tokens. Conversations use Prelude ChatML with
`<think>` and loss on all non-padding target tokens except tool/environment response spans. Preserve system/user
prompts, assistant reasoning, assistant tool calls, and role boundaries.

The current [renderer](../scripts/sft_gen/prerender_levanter.py) only trains assistant spans. The current
[SFT normalizer](../scripts/data/materialize_prelude_sft_v1.py) also drops `reasoning_content` on assistant turns
before the last user message. Mid-training needs a separate mode or adapter that preserves those trained turns.
Keep existing SFT behavior as the default.

- [x] Add source-specific adapters with stable IDs and provenance. Read Specialized text from the actual schema
      (`text` in the inspected InfiniByte shard), and construct repository documents from Stack's nested files.
- [x] Normalize tool/environment responses semantically. In sampled Cascade math-tool, SWE-agentic, and
      conversational-agent records, responses are **`role=user` with `<tool_response>...</tool_response>` content**.
      A `role=tool` check alone is insufficient. Recognize documented response wrappers at message boundaries,
      preserve genuine user turns, and reject or explicitly account for ambiguous/malformed cases.
- [x] Preserve existing reasoning/tool markup without nesting duplicate wrappers. Keep earlier assistant reasoning
      when its turn contributes training loss. Record counts of every normalization/rejection rule.
- [x] Add an explicit mid-training mask policy to rendering: ordinary system/user and assistant targets receive
      loss; complete tool/environment response spans receive no loss. Specify delimiter/header treatment and verify
      causal target shifting through the actual loader. Padding and packed example boundaries must be handled.
- [x] Freeze the tokenizer and template from `rl/sft/prelude/` and record their hashes/token IDs. Define initialization
      of the six added ChatML/think/response token embedding rows, verify model vocabulary compatibility, and include
      the tokenizer in checkpoint export.
- [x] Add focused fixtures for genuine user turns, tool-role responses, user-role response wrappers, assistant tool
      calls, multiple reasoning turns, empty content, and packing/length boundaries. Check exact token IDs and masks
      through the training loader, while retaining the existing assistant-only SFT behavior.

### Step 2 implementation and verification (2026-10-17)

Implementation: [source adapters/masks](../scripts/data/prelude_midtrain.py),
[resumable materializer](../scripts/data/materialize_prelude_midtrain.py),
[tokenizer and model preparation](../sft/prelude/prepare_midtrain_tokenizer.py), and
[explicit loader format](../sft/prelude/midtrain_format.py).
The [operating notes](../sft/prelude/MIDTRAIN.md) specify the canonical schema, every mask boundary,
normalization/rejection policy, initialization/export behavior and reproduction commands.
[Machine-readable evidence](../sft/prelude/midtrain_step2_evidence.json) pins the implementation,
tokenizer/model contract, actual Levanter source/runtime, artifact hashes, counts and jobs.

Artifacts: `/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step2`.
Build ID: `8d499b5955ec902885e8e2ea1f14b667a83f9f0bcc8c24422fa6e34d54a370aa`.

- The frozen adapter inventory contains 965 selected source files and 6,740 resumable work units.
  Source IDs retain physical input SHA256, revision, relative path and byte-offset/global-row locator.
  Stack uses the selected `all` repository list and both permissive and no-license non-vendor files.
- The source pilot covers 102 groups: all nine Cascade files, five Specialized categories, 27 Terminal
  subset groups, 11 UltraData languages, selected Stack repositories, and all 49 replay/PDF files.
  It inspected 306 rows, normalized 303 and rendered 300 (2,616,608 tokens; 1,974,521 target tokens).
  Two boundary-ambiguous tool-response rows and one vendor file were rejected with locators/reasons.
  Three overlength rows remain normalized and are explicitly deferred for chunking. These are bounded
  verification counts, not corpus-wide production yields. Pilot job `15718676` completed successfully.
- Ordinary system/user/assistant targets train in full. Whole tool groups, including shared user header,
  wrappers, payload, closing delimiter and trailing LF, are masked. BOS is untrained. The actual loader
  shifts the target mask left by one, zero-pads, blocks cross-document attention and masks packed BOS
  boundaries. The dedicated format rejects implicit truncation; existing SFT defaults are unchanged.
- `tokenizer.json` and the ChatML template retain their step 1 hashes. The derived bundle corrects stale
  native-EOS metadata in `special_tokens_map.json` to match ChatML EOS in `tokenizer_config.json`;
  both original and derived hashes are recorded. Documents still end with native EOS 2; exported
  generation configuration stops on IDs 262146 and 2.
- `base_hf` initializes rows 262145–262150 in **both** `[262272,4096]` embedding/head matrices to their
  respective means over native rows `[0,262144)`, excluding pad. Initialization job `15718661` completed;
  every other model byte is verified unchanged. Unchanged shards are symlinks to the frozen base.
  Current-code tests independently recompute and verify all twelve initialized rows and the output shard
  hash. The trainer preserves padded model vocabulary during HF import and copies the frozen tokenizer
  plus contract into every local HF checkpoint export; resume/export never reinitializes token rows.
- **29 tests pass:** 23 source/census/selection/normalization/initialization tests (job `15718683`) and six
  real Levanter loader/checkpoint tests in `levanter_jax_26.09-py3_torch.sif`. These cover exact IDs/masks,
  genuine users and tool wrappers, earlier reasoning, structured calls, empty turns, resume integrity,
  causal shifting, padding, packing/attention boundaries, exactly 65,536 versus 65,537 tokens, SFT defaults,
  and real tiny Qwen3 import/export with the full Prelude vocabulary. No model training has run.

## 3. Deduplicate and decontaminate before sampling

Step 3 completed on JURECA under `synthlaion` (2026-10-17). The operating notes and exact identity,
benchmark-review, split and execution policies are in
[CURATE_MIDTRAIN.md](../sft/prelude/CURATE_MIDTRAIN.md).
Artifacts: `/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step3`.
Scan build: `a7152c86f2b2ccff1d398c4b35ca5ff653319eb2db075498b94e1e18c66df476`.
The full SFT reference scan (`15718734`) completed all 126 parts and verified exactly 4,412,306 retained
SFT v1 rows, including input byte checksums. The full source scan (`15718735`) completed all 8,932
units on four CPU nodes in 26m21s. Independent coverage checks reconcile every unit against frozen
footers, selected repository locators, JSONL census ranges, and replay row cuts: **199,371,691 input
rows, 197,537,512 adapter-accepted rows**. Stack coverage is the selected 128,122 repositories.

The scanner evaluated 281,182,600,496 13-word windows. It saved raw matches in 529,041 records and
at least one diversity-eligible match in 456,016. Global review (`15718807`) completed. The
[reviewed exclusions](../sft/prelude/benchmark_exclusions_v1.json) list 290 exact boilerplate
13-grams with reasons; all other eligible matches remain removal evidence. Frequency alone never
excludes a hash. The sorted native index view preserves all 5,185,587 original hashes.
After review, 401,079 records retain benchmark evidence; 63 also match reserved SFT tasks and are counted
under SFT removal. Another 54,937 records have only reviewed boilerplate among their diversity-eligible
hits; this does not guarantee retention because other filters still apply.

The global decisions are frozen as `d4a08456c0b8c953a79700392de5887e4c4381bcd2fdc620a873f21c30662401`.
All 288,476 decision files were checked and independently recounted by parallel completion job
`15718847` in 2m05s. Its manifest matches the completed original reducer (`15718814`) byte-for-byte
on the full corpus, SHA256 `d73a451d11eb7184781cfd3ac0f9653e2a4d38f5858a255ff61f38a7190e0325`.
Full canonical materialization (`15718817`, four nodes) completed all 8,932 parts in 8m57s.
Final verification (`15718887`) completed successfully in 52m28s. All 8,932 canonical parts passed
checksum and row-count checks; global checks verified unique retained IDs, no train/validation group
overlap, agreement with frozen keep decisions, canonical schema, and consistent curation versions.
Curation build: `128263cb0e97dcc66720515095d5c6a4b5f724511ed8cc081ba60aa9e58bb6d1`.
Completion manifest: `curation_manifest.json`, status `step3-complete`, SHA256
`852dfef3a912d2270a97c9ed0319c2f5049d1ea8e5800c24ee6fff0ccc3d463d`.
The [execution evidence](../sft/prelude/step3_evidence.json) records policy/runtime/tool hashes and jobs.

Exclusive removal reasons are applied in this order: SFT overlap, benchmark evidence, Stack file
coverage, then exact duplicate. Their counts reconcile exactly to all 197,537,512 accepted source rows:

| Decision | Rows |
|---|---:|
| Reserved SFT task | 247,332 |
| Benchmark match | 401,016 |
| UltraData file already covered by retained Stack | 1,642,127 |
| Exact task/document duplicate | 25,931,369 |
| **Retained** | **169,315,668** |

The frozen split assigns 167,639,429 retained IDs to training and 1,676,239 to validation, using 1%
by task component, repository, or document identity. Full payload/ID/split verification passed.
The 73,025 low-diversity-only raw matches remain recorded without causing removal.
The content-ID conflict audit contains 168,990 keys with different file bytes; ID equality alone never
causes removal. Eight regression checks pass, including Unicode/native equivalence, uncapped hits,
SFT aliases, exact code coverage, quality representatives, frequent true benchmark matches, input
manifest tampering, and identical parallel-completion manifest bytes.

Adapter rejects include 704,569 ambiguous tool-response rows (478,860 conversational-agent and
225,709 math), 46 unknown-role rows, seven malformed inline-reasoning rows, 1,129,556 ineligible
UltraData files and one file missing provenance. The strict boundary policy is unchanged from step 2;
every rejection retains its source locator and reason. Source payloads and the SFT v1 release are
unchanged. No near-duplicate pass or model training has run. The counts above describe canonical
candidates; step 4 below records the retained-token recount, replay/PDF eligibility cuts and weighted
parent selections. Actual token-ID chunk export remains step 5.

- [x] Deduplicate conversation tasks against SFT v1, especially Cascade math versus Nemotron-SFT-Math-v4. Freeze
      prompt canonicalization and hashes, excluding tool responses from task identity; retain stable task IDs where
      available. Save removed IDs and overlap counts by source.
- [x] Reserve AgentTrove balanced traces and our RTS/TMax teacher traces for downstream agentic SFT in this draft.
      Resolve the older plan's conflicting listing of AgentTrove among mid-training sources before freezing inputs.
      Exclude Cascade `terminal_agent` to avoid duplicating Terminal-Corpus.
- [x] Deduplicate UltraData files against files included in selected Stack repositories using content IDs/hashes.
      Assign each retained item one mixture category so code and long-document allocations do not double count it.
- [x] Reuse MV's hash-only deduplication pattern with explicit content/prompt keys. Check that distinct code
      differing in case, indentation, or string contents survives, and retain the full structured record by stable ID.
- [x] Freeze a held-out validation partition by task/repository identity, keeping related chunks and duplicate
      prompts together. Persist split assignments before packing.
- [x] Apply `/p/data1/mmlaion/marianna/decontam/index_prelude_v1` to documents and all conversation content,
      including masked tool outputs and reasoning. Reuse the existing 13-gram hashing and record index hashes.
- [x] Review boilerplate handling at this corpus scale. The SFT conversation filter excludes low-diversity n-grams
      and hashes matching more than 20 rows; do not assume that absolute frequency cutoff is suitable here.
      Persist raw/filtered hits, inspect true benchmark matches, and version any exclusions before dropping rows.

## 4. Freeze the mixture by retained token counts

Completed on JURECA under `synthlaion` (2026-10-17). All **169,315,668 retained canonical rows**
were counted again with the frozen tokenizer/template and production masks. The full pool contains
529,343,825,430 seen / 525,737,061,846 trainable tokens. Replay >4K and long-PDF <32K cuts exclude
536,439 and 519,339 candidates respectively. The eligible training pool has **166,594,212 rows /
512,064,060,367 seen / 508,494,258,155 trainable tokens**; held-out validation has 1,665,678 rows /
5,118,310,528 seen / 5,082,404,934 trainable tokens. Source/split coverage, all 8,932 checksums/footers,
worker/SQL totals, mask accounting and stage accounting passed.

Outputs are under `/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step4`.
`retained_census.json` completion build is
`481bb408ccca3596fc317c7d1e83c1b9ebf60ba2ce209cb6c3606a8c0a3fe60b`.
The frozen **`mixture_v3/mixture_manifest.json`** has status `step4-complete`, build
`7239c4ca51352ad5bafc4565cb0675b1ebdb73549fcf92c37401c4f71f493f1b`, SHA256
`a03d76a897805785add51d8f93cfd3eb1c38dbd95353a8acdf30ce5931beddc6`.
The [policy and reproduction commands](../sft/prelude/MIXTURE_MIDTRAIN.md) and
[execution evidence](../sft/prelude/step4_evidence.json) pin every dependency and result.

| Context | Selected parents | Seen tokens | Trainable tokens |
|---|---:|---:|---:|
| 16K | 186,595 | 499,918,595 | 489,184,884 |
| 32K | 347,764 | 999,867,143 | 973,966,082 |
| 64K | 673,473 | 1,999,735,426 | 1,935,346,645 |
| **Total** | **1,207,832** | **3,499,521,164** | **3,398,497,611** |

The 3.5B seen-token upper envelope uses 0.5B / 1B / 2B stage budgets. Whole-parent selection has a
0.0137% total undershoot. Actual category shares differ by at most 0.004 percentage point per stage;
document seen-token shares are 64.9952–64.9979%, with trained shares 66.2029–67.0572%.
All 52 source weights are fixed across stages, with token quotas and no parent replacement. The maximum
effective epoch is 0.5332 over a full eligible training pool and 1.0 over a context-qualified pool.
`source_exposure_report.json` records both denominators for every source/stage.

**Explicit length decisions.** Only 38,613,004 clean training chat tokens are in originals at least
32K long, and IF has none. A strict 32K minimum cannot supply the 64K stage's 200M-token chat/IF
allocation without repetition. The frozen recipe therefore exempts chat/IF and <=4K replay from
the preferred 8K / 16K / 32K original-length minima. At least 67.5% of seen exposure comes from
parents meeting those minima. Scientific-Coding, Cascade conversational-agent and selected short
code repositories have no 32K-qualified originals, so their cross-stage weights are zero; long
Stack repositories retain 6.25%. Unavailable buckets are explicit in the manifest. Ultra language
ratios and the replay proxy's family/language ratios are fixed to measured clean availability.

**Execution and validation.** Pilot `15719041` checked 440 rows from 55 source/subset groups.
Initial census `15719043` preserved 8,929 completed parts before bounded completion `15719065`
finished the final 224,811 Terminal rows in 221 disjoint ranges. The equivalent linear renderer
checks identical rendered text per row, preserves semantic tool masks, and passed complete ID/mask
fixtures plus byte-identical full-unit parity (`15719064`, 2,767 conversations / 39,242,872 tokens).
Finalization `15719066`, balanced mixture `15719080`, and independent full selection audit `15719081`
all completed successfully. Sixteen regression tests pass. The audit verifies checksums, seeded order,
train-only IDs, global parent uniqueness, chunk accounting, length eligibility, actual seen/trained
source/category/format shares, effective epochs and code-language ratios.
These artifacts are virtual chunk counts and selected parent IDs; model training, final token-ID
export and loader smoke testing remain steps 5–6.

Frozen category targets (seen-token exposure):

| Category | Target seen-token share | Candidate sources |
|---|---:|---|
| Reasoning / math / science | 30% | Cascade math/science; Specialized InfiniByte, RQA, STEM-SFT, textbooks |
| Code + agent | 25% | Selected Stack repos, UltraData-L2, Scientific-Coding, Terminal-Corpus, Cascade SWE/conversational-agent |
| Chat / instruction following | 10% | Cascade chat part 1 and IF |
| Long documents | 12.5% | Disjoint selected long repos and pinned long PDF/arXiv documents |
| Short-sequence replay | 22.5% | Original no-change components or a documented approximation |

These category weights sum to 100%. Separately target **65% raw documents / 35% conversations** (the original
60-70% / 30-40% ranges). Category and format are separate dimensions of the same tokens, not extra allocations.

- [x] Choose per-source weights after the clean token census so category and format targets are jointly feasible.
      Calculate sampling by token exposure rather than row count; report both seen and trained token shares.
- [x] Limit unintended repetition of small sources, record effective epochs, and freeze sampler seed/exhaustion
      behavior. Check actual sampled token shares against the manifest before training.
- [x] Keep category/format shares and replay allocation consistent across context stages. Change long-example
      length selection by stage and retain replay as an explicit exception; report any unavailable length buckets
      rather than silently changing source weights.

## 5. Build a versioned release and loader smoke run

Output root: `/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step5`.
The [export and handoff guide](../sft/prelude/EXPORT_MIDTRAIN.md) records the exact
CPU build, continuation policy, Levanter import and MN5 transfer commands.
The portable release contains **1,207,832 unique parents / 1,224,141 chunks**,
**3,499,521,164 seen / 3,398,497,611 trainable tokens**, matching every frozen
source/stage count. All original tokens survive; continuation prefixes add 4,191,413
masked context tokens. There are 26,103 data shards and 5,570,423,420 artifact bytes.
Release build: `e4e594c1190b2c02b576875e94484467d2a7983218fc98fe860ec581fa875819`.
Manifest SHA256: `1acbb7b946512f3a25708e05ca1f84178d04e2e3c3c980ea99c4af276aaa7e1a`.

Preparation `15719227` verified all 8,932 canonical checksums and located every
selected parent. Pilot `15719240` and independent pilot audit `15719243` passed.
Full export `15719244` completed on four `synthlaion` CPU nodes in 2m35s;
independent read-back `15719245` verified every ID/mask, reconstructed parent digests,
continuation intervals/overlaps/EOS, metadata and counts in 4m00s. There are no
zero-target chunks in this particular selection; synthetic tests cover their retention.

CPU loader qualification covers all 49 active source pools at every context, exact
packed IDs/masks, causal shift, padding, segment boundaries, finite shuffle and
cache relocation. Tiny Qwen3 with the full vocabulary passed real Adam updates,
nonzero gradients on all six added embedding/head rows, exact checkpoint resume,
HF export/reimport and tokenizer checks. Full trainer fresh/resume checks exposed
and fixed a resume-exemplar vocabulary mismatch: keep the model axis at 262272
while preserving the tokenizer's 262151 entries. MN5 has a byte-identical Levanter
image, and all portable data/runtime/model destination checksums passed. The initial
transfer used `jsc-vm`; subsequent bulk copies should run on Jupiter using its
existing rclone `mn5:` remote (SSH into Jupiter from the VM). Target cache imports and the short
9B GPU canary are recorded in [step5 evidence](../sft/prelude/step5_evidence.json).
All three full native caches and loader smokes passed on MN5 in job `47456439`
(4m38s). Native finite packed lengths are 38,791 / 38,825 / 41,554 at 16K / 32K /
64K, with exact release row/seen/trainable counts and identical target/JURECA
loader reports. Use `cache_v3`; earlier interrupted cache roots remain preserved.
The actual 9B canary passed on four MN5 H100s in job `47456121` (34m34s): two
fresh updates, full model/optimizer restore from native checkpoint, then one
resumed update from step 2. Recorded fresh/resumed losses were 5.7836275 /
5.2237182. Independent read-back `47456853` passed in 36s. Both BF16 HF exports
have 399 tensors / 9,094,607,872 parameters with every weight finite, all base tensor
names/shapes preserved, vocabulary 262272, EOS IDs 2/262146 and all eight frozen
tokenizer hashes intact. All six added embedding/head rows changed during training.
This is a 128-token synthetic correctness check; production context/attention
settings, finite final-batch scheduling and evaluation gates remain step 6.

- [x] Add a resumable CPU build entrypoint under `rl/sft/prelude/` and adapters under `rl/scripts/data/`.
      Keep normalized data, clean documents, rendered conversations, exclusion/split records, manifests, and reports
      in separate subdirectories. Never overwrite source snapshots or the SFT v1 release.
- [x] Wire the reusable MV drivers through the compatibility adapters above. Use the bounded CPU metadata/global-hash
      reduction implemented in step 3 and independent CPU shards for tokenization/census. Freeze all dependencies, explicit paths,
      resource settings, and the source manifest; make the default recipe independent of a VM deployment.
- [x] Include source identity, adapter/script hashes, schema, dedup/filter settings, mask policy, tokenizer/template,
      benchmark-index hashes, split seed, and category assignments in rebuild keys. Verify that changing one of
      these invalidates the appropriate output even when input file metadata is unchanged.
- [x] Preserve original document text/structured conversations in step 3; export both as portable token/mask
      Parquet with stable parent/chunk IDs and provenance, as agreed for JURECA preprocessing and MN5 import.
      Use the frozen stage selections and finite Levanter stage-dataset factory; production final-batch handling remains step 6.
- [x] Preserve complete examples where possible. Define handling of overlength repos/conversations, continuation
      context, document separators, and loss on chunk boundaries; report dropped/chunked tokens. Do not count a
      window filled with unrelated short examples as evidence of long-range training.
- [x] Emit before/after row/token counts for every filter, stage length distributions, trained-token shares,
      rejected-record reasons, overlap/decontamination results, and artifact checksums.
- [x] Exercise one small CPU training/checkpoint-resume/export cycle with both formats. Verify finite loss, correct mask
      shifting, nonzero gradients on added chat-token rows, and matching tokenizer IDs after reload/export.
- [x] Register the resulting dataset under `rl/inventory/datasets/` with exact paths, source revisions, tokenizer,
      build command/version, counts, and manifest hash.
- [x] Verify transfer checksums, all three native cache imports/loader smokes, and the short 9B fresh/resume/export
      canary on MN5. This uses synthetic mixed-format examples, not the production allocations or context curriculum.

## 6. Draft context-extension recipe and evaluation gates

The unchanged dominant2 base's 4K evaluation is already complete. Reuse the
[evaluation overview](../evals/README.md),
[2026-10-16 run record](../evals/runs/20261016-prelude-anneal300b-ml1-dominant2-base-fewshot.yaml)
(job `15715430`) and [base-few-shot protocol](../scripts/eval_jureca/base_fewshot/README.md)
as the short-context regression reference. The record identifies the raw results
and per-benchmark metrics; the selected model's
[inventory](../inventory/models/prelude-9b-anneal300b-ml1-dominant2-iter0989075.yaml)
pins revision `9d252ade085f74176eb2a3fa321a59f3b5af6deb`.

| MATH (4-shot) | GSM8K (8-shot) | MMLU-Pro (5-shot) | HumanEval+ (0-shot) | MBPP+ (0-shot) | MultiPL-E (0-shot) |
|---:|---:|---:|---:|---:|---:|
| 47.8% | 80.5% | 42.2% | 54.3% | 38.1% | 24.7% |

Preserve the recorded 4096-token window, greedy plain-text prompts and BOS handling,
lm-eval 0.4.13 / vLLM 0.26, few-shot counts, generation budgets and task versions
for stage comparisons. In particular, use MATH's `math_verify` score, GSM8K's
flexible extraction, `mbpp_plus_base` with the full plus tests, and the same eight
MultiPL-E languages. The ChatML/thinking baseline and long-context retrieval
checks remain separate gates. Register subsequent runs through the existing
`rl/evals/record_base_fewshot.py` workflow and regenerate its overview as documented
in `rl/evals/README.md`.

The selected dominant2 checkpoint now has a qualified MN5 path at
`/gpfs/scratch/ehpc532/ot/models/prelude-anneal300b-ml1-dominant2-iter0989075`.
The finite-loader runtime candidate is staged at
`/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/gates_20261018/runtime_production_v1`;
its draft launcher is `rl/sft/prelude/prelude_midtrain_mn5.sbatch`. The measured recipe used RoPE theta `6000000`, a `65536` maximum model context, sliding-window
attention disabled, Qwen3 `32:8` FA4, packed cross-document causal masks, peak LR
`1e-5` and 100 cumulative warmup steps. The reconciled finite draft uses one cosine schedule over `4419` cumulative steps,
with `1048576` padded sequence positions per optimizer step. The theta6M quality gates failed;
the launcher requires an explicit theta for the next preregistered pilot. The finite one-pass allocation
uses actual packed-cache counts; seen tokens and padding are recorded separately in
[`budget.json`](../evals/prelude_midtrain/budget.json):

| Stage | Context | Real seen tokens | Packed sequences | Steps | Cumulative end step |
|---|---:|---:|---:|---:|---:|
| 1 | 16K | 499,918,595 | 38,791 | 607 | 607 |
| 2 | 32K | 999,867,143 | 38,825 | 1,214 | 1,821 |
| 3 | 64K | 1,999,735,426 | 41,554 | 2,598 | 4,419 |

Total: 3,499,521,164 seen / 3,398,497,611 trained tokens and 4,633,657,344 padded
sequence positions. Each cache uses the seed-20261017 bijective shuffle exactly once;
final batch padding is 57 / 23 / 14 zero-loss sequences. The former 477/1431/2862
step draft counted nominal positions as data tokens and underallocated the frozen 64K data.

The eight-node mesh uses `context=4` within each four-H100 node and `data_dcn=8`
across nodes. Batch size is 64 / 32 / 16 sequences at 16K / 32K / 64K, with one
sequence per data shard and gradient accumulation. `embed` is sharded over `context`;
`mlp`, `heads`, `kv_head`, and `vocab` use `data_dcn,data`. Per-rank logs are retained.
Stage 2 and 3 require a native optimizer checkpoint; loader resume subtracts the
stage's global start step for both fresh-stage and same-stage continuation.

The FA4 numerical probe passed in job `47457873` for 32 query heads / 8 KV heads /
head dimension 128 and packed causal segment masks, with maximum absolute error
`0.0007448196` and finite outputs and gradients. The one-node four-H100 synthetic
9B canary `47458490` completed fresh training, native full-state resume, and
TensorStore/HF export in 35m32s. Chained packed production-cache smoke jobs then
passed at 16K (`47459354`, 2m37s), 16K-to-32K (`47459402`, 2m34s), corrected
state-sharded 32K (`47460001`, 2m43s), and a real 64K optimizer update
(`47460202`, 2m40s job / 41.6s step). The 64K run produced
`checkpoints/step-3` and `hf/step-3` under
`/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/runs/prelude-longcontext-continuation-64k-cp4-step4`.

For the long-context smoke, XLA could not reduce a 58.38 GiB lower-bound computation
below roughly 58.5-61.3 GiB per process. The 16K/32K/64K tasks peaked at
27,031,578K / 29,371,601K / 28,594,904K `MaxRSS`. Native checkpoint writes give
each process 25.41 GiB of a 101.64 GiB state and stage 15.94 GiB peak under the
16 GiB checkpoint budget. These numbers qualify the one-node smoke path; production
eight-node throughput and sustained scheduling remain to be measured in the run.

- [x] Establish the unchanged base's 4K results: reuse the recorded dominant2 MATH, GSM8K, MMLU-Pro,
      HumanEval+, MBPP+, and MultiPL-E baseline above.
- [x] Establish the initialized step-2 model's small ChatML/thinking baseline, distinct from the unchanged base's plain-text results: [scores](../evals/prelude_midtrain/initialized_chatml_scores.json). Thinking-on
      GSM8K/MATH500/HumanEval+ = 70/32/71%; thinking-off = 16/4/0%. Nonthinking code loops on
      `<tool_call>` (100/100 generation limits); thinking-on MATH has 57/100 unclosed reasoning outputs.
      This establishes a baseline; it does not qualify chat formatting or later stage promotion.
- [x] Freeze metric-specific regression tolerances before scaling: [gate policy](../evals/prelude_midtrain/gate_policy.json),
      frozen at 2026-10-18 12:58:16 UTC and committed as `0382322`. Maximum drops are 2 pp for MATH, GSM8K,
      HumanEval+, MBPP+, MultiPL-E and 1.5 pp for MMLU-Pro. Require paired, familywise-adjusted one-sided
      bootstrap lower bounds above the margins; uncertainty can yield inconclusive, never automatic pass.
- [ ] Evaluate after each stage at the same 4K settings, plus RULER/NIAH at the trained context length and the
      small ChatML/thinking suite. Investigate meaningful regressions before proceeding to the next stage.
- [x] Record the smoke runtime/memory, FA4 numerical check, context curriculum, mesh, checkpoint and HF export
      evidence across the real 16K/32K/64K optimizer steps.
- [x] Record eight-node pilot throughput/memory, native and HF checkpoint/config/tokenizer evidence, and raw
      initialized/smoke quality results in the existing evaluation registry.
- [ ] Qualify the context-extension recipe against the failed long-context targets before production promotion.
- [ ] Record production checkpoint/config/tokenizer revisions, sampled token shares, and matched post-stage results. The full 3-3.5B-token run and downstream SFT results remain future work; this issue
      delivers a verified dataset and a draft recipe with measured smoke/pilot evidence; the recipe is not yet quality-qualified.

## Remaining-gate results (2026-10-18; context quality failed)

Evidence root: `/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/gates_20261018` (MN5),
with JURECA preparation/results under
`/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/gates_20261018`.
The production training run has not been submitted.

- Eight-node attempt `47487732` failed before training because the original mesh put
  all 32 devices in the within-node axes. Corrected cross-node attempt `47488020`
  reached the optimizer compile but exhausted GPU memory. Retry `47488201` keeps the
  global batch fixed and uses one sequence per data shard with gradient accumulation;
  completed all 20 updates and saved native/HF `step-19`. Training step `47488201.0`
  and its parent job completed `0:0` (1h06m21s). Across 18 warm updates (global steps 2–19),
  mean duration was 128.887s and throughput 8,135.6 padded positions/s; MFU averaged 1.99%.
  All 32 GPU memory samples are retained. Independent full readback verified all 399 tensors /
  9,094,607,872 parameters finite, the frozen tokenizer contract and all six added rows preserved
  and updated: [evidence](../evals/prelude_midtrain/eight_node_evidence.json).
- Actual finite-tail/resume loader tests `15720206` passed. End-to-end CPU test
  `15720223` then exposed a pinned Haliax backward-pass bug: zero-weight microbatches
  returned zero loss but differentiated a masked 0/0. The scoped finite-stage fix guards
  the denominator; tests `15720307` passed, and corrected train/resume `15720306`
  completed all four updates with finite exported parameters after both tail batches.
- ChatML baseline inputs are fixed SHA256-ranked 100-example subsets of GSM8K,
  MATH500 and HumanEval+, both thinking modes, 4096 window / 1024 generation / greedy.
  Raw output and special tokens are retained; unfinished reasoning scores as empty.
  JURECA evaluation `15720292`, no-op continuation `15720329` and grader `15720353` completed; it uses the same vLLM 0.26 installation as the original
  plain-text baseline, with its own distinct ChatML prompts.
- RULER uses all 13 lm-eval 0.4.13 tasks, upstream generation and first
  100 of 500 examples per task. Exact token IDs, raw documents, input and source hashes
  are frozen before inference. 4K initialization evaluation `15720293` completed: RULER **80.923%**, below the 85% target;
  repeated-passkey NIAH **100%**, above the 95% target. SQuAD QA ended immediately on EOS2
  for 99/100 prompts, and Hotpot QA for 76/100. Retain this failed mean target.
  [Per-task scores](../evals/prelude_midtrain/initialized_ruler_4096_scores.json). The 16K smoke completed in `47489495` (9m45s): **RULER 4.323%, passkey NIAH 0%**,
  both targets failed. [Matched 4K controls](../evals/prelude_midtrain/rope_diagnostic_scores.json),
  job `47490675` (8m26s, exit `0:0`), measured initialized/smoke weights at both theta values:
  initialized theta100K=100%, initialized theta6M=81%, smoke theta6M=81%, smoke theta100K=100%.
  The unchanged initialized control also reproduces its vLLM 0.26 result under MN5 vLLM 0.27.
  Changing theta alone therefore reduces even 4K retrieval; restoring theta recovers this metric
  on the smoke weights. This identifies a recipe problem; it is not full six-benchmark parity proof.
  The 32K smoke `47489496` also failed: **RULER 3.641%, passkey NIAH 0%**.
  The 64K smoke `47489497` completed (29m21s): **RULER 0%, passkey NIAH 0%**, both failed.
  These smoke measurements characterize infrastructure and current weights;
  they do not substitute for post-stage quality gates after substantial training.
- During preparation, upstream FWE sizing exceeded the 16K window including generation.
  Regenerate that entire task before freezing its inputs, with measured headroom, and retain its actual
  generator window. No input is truncated or selectively omitted. Upstream sampling also uses global RNG state;
  future checkpoints must replay the frozen IDs/prompts, rather than regenerate them. Targets remain mean
  RULER >=85% and repeated-passkey NIAH >=95%; failed targets will remain failed.

## Next qualification work

Infrastructure, frozen tolerances, initialized ChatML measurements and all 4K/16K/32K/64K
retrieval measurements are complete. The current theta6M recipe **does not pass quality gates**.
Keep the issue in progress and production unsubmitted. The launcher now requires an
explicit `MIDTRAIN_ROPE_THETA`; the failed 6M setting is no longer an implicit launch default. Next, freeze a revised RoPE/context
curriculum and a bounded adaptation pilot, replay the same benchmark inputs, and require both
long-context targets plus matched 4K noninferiority before promotion. Do not lower thresholds
or use the few-update smoke as evidence that adaptation training has succeeded.

### Revised RoPE recipe and stage-1 pilot (2026-10-20)

Decision (user): RoPE theta **9M** for all three stages, then **YaRN factor 2** at inference for 128K
(no 128K training stage). 9M is the critical-dimension estimate for 64K from a 4K / theta-100K base
(critical dimension 74 of 128; 16K needs ~0.8M, 32K ~2.7M, 64K ~8.9M, 128K ~30M). One theta for every stage
avoids re-adapting at each stage; YaRN x2 changes only the slow dimensions, so <=64K behaviour is kept.

Pilot: stage 1 only (16K, steps 0-607 of the 4419-step cosine), Megatron-Bridge on MN5 (TP4, distributed
optimizer, selective recompute, fp32 grad reduce, 8 nodes), run `stage1-16k-theta9m`, job `47617036`,
checkpoints every 150 steps. Promotion to stage 2 requires, on the same frozen inputs: matched 4K
noninferiority vs the initialized model at theta 100K (4K passkey back to ~100%), plus 16K RULER and
repeated-passkey NIAH targets. 128K is checked later with YaRN x2 on the 64K checkpoint.

### Stage-1 results (step 607; 2026-10-21)

Training: `47617036` timed out while saving step 450; resume `47627559` loaded step 300 and completed
301-607 (losses match the first job to ~4 decimals). 100-step mean loss 1.275, 0.953, 0.930, 0.931, 0.924,
0.920. 11.7 s/step on nodes spread over 2-3 racks vs 22.9 s/step over 5 racks; later jobs request
`--switches=2@01:00:00` and save before the time limit (`EXIT_MINS`).

Export: Bridge HF export of step 607 matches the Megatron checkpoint on 16 stage-1 sequences (mean loss
0.93416 vs 0.93419, max per-sequence difference 0.00104; base 2.01169), job `47627813`. Copied with
SHA256 to Jupiter `/p/scratch/ccstdl/dc-agent-shared/checkpoints/prelude-9b-midtrain-v1-stage1-16k-theta9m-step607`.

RULER (frozen inputs, 13 tasks x 100) and repeated-passkey NIAH. Records on JUWELS/JURECA (vLLM 0.26):
[4K](../evals/runs/20261020-prelude-midtrain-s1-16k-t9m-s607-ruler4k.yaml),
[16K](../evals/runs/20261020-prelude-midtrain-s1-16k-t9m-s607-ruler16k.yaml),
[32K](../evals/runs/20261020-prelude-midtrain-s1-16k-t9m-s607-ruler32k.yaml),
[64K](../evals/runs/20261020-prelude-midtrain-s1-16k-t9m-s607-ruler64k.yaml).

| Context | RULER mean | Passkey | Gate | Reference |
| ---: | ---: | ---: | --- | --- |
| 4K | 87.4% | 100% | pass | initialized 80.9% / 100% (theta 100K) |
| 16K | 74.1% | 100% | fail (mean) | smoke theta6M 4.3% / 0% |
| 32K | 60.2% | 100% | fail (mean) | smoke theta6M 3.6% / 0% |
| 64K | 12.6% | 82% | fail | smoke theta6M 0% / 0%; not yet trained at 64K |

The MN5 replay (vLLM 0.27, same frozen inputs) agrees: 4K 87.83% / 100%, 16K 73.87% / 100%.
Theta 9M plus 16K training fixes the theta-6M failure: 4K passkey is back to 100% (it was 81% at theta 6M), and
passkey holds out to 32K. The 16K mean misses 85%. Weakest 16K tasks: multikey-3 37%, CWE 31%, QA 36/39%.
At 4K, FWE dropped from 89% to 50%.

Short-context few-shot ([record](../evals/runs/20261020-prelude-midtrain-s1-16k-t9m-s607-base-fewshot.yaml)),
4K window, greedy, vs the dominant2 parent:

| | MATH | GSM8K | MMLU-Pro | HumanEval+ | MBPP+ | MultiPL-E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dominant2 parent | 47.8 | 80.5 | 42.2 | 54.3 | 38.1 | 24.7 |
| stage 1 step 607 | 47.4 | 79.4 | 43.0 | 54.3 | 47.6 | 33.6 |
| change (pp) | -0.4 | -1.1 | +0.8 | 0.0 | +9.5 | +8.9 |
| margin (pp) | -2 | -2 | -1.5 | -2 | -2 | -2 |
| Qwen3.5-9B-Base (reference) | 56.2 | 84.5 | 58.1 | 50.0 | 64.0 | 54.1 |
| stage 1 vs Qwen (pp) | -8.8 | -5.1 | -15.1 | +4.3 | -16.4 | -20.5 |

The Qwen3.5-9B-Base row is the same suite and settings
([record](../evals/runs/20261015-qwen35-9b-base-base-fewshot.yaml)); it is a reference, not a gate.
There is no Qwen3.5-9B RULER or ChatML measurement on the frozen inputs yet, so the RULER and ChatML tables
have no Qwen row. Against the dominant2 parent, every point estimate is within its frozen margin. The paired, familywise-adjusted bootstrap required by the
[gate policy](../evals/prelude_midtrain/gate_policy.json) has not been run yet, so this is not a formal pass.

ChatML ([record](../evals/runs/20261020-prelude-midtrain-s1-16k-t9m-s607-chatml.yaml)), 100 examples each,
1024 generation tokens:

| | GSM8K | MATH500 | HumanEval+ |
| --- | ---: | ---: | ---: |
| initialized, thinking off | 16 | 4 | 0 |
| stage 1, thinking off | 87 | 57 | 23 |
| initialized, thinking on | 70 | 32 | 71 |
| stage 1, thinking on | 79 | 38 | 31 |

Thinking-off is much better. Thinking-on HumanEval+ dropped from 71 to 31, but 65/100 of those outputs hit the
1024-token generation limit with reasoning still open. Most of the drop is longer reasoning that does not
finish in the budget, not necessarily lost coding ability; check this with a larger budget before treating it
as a regression. Thinking-off HumanEval+ still has 42/100 generation-limit hits and 23 open reasoning blocks.

Decision (user): proceed to stage 2 and recheck 16K there. This departs from the preregistered rule above,
which required the 16K targets before stage 2.

### Stage 2 (32K; 2026-10-21)

Run `stage2-32k-theta9m`: 1.0B real tokens (974M trained), GBS 32 x 32K, steps 607-1821 of the same
4419-step cosine. Data offset `STAGE_START_SAMPLE=38848` = samples consumed by stage 1, and the LR schedule
was patched to count optimizer steps so the GBS change keeps it continuous (step 608 LR 9.696e-6 at both GBS).
Job `47652129` completed all 1214 steps in 4h25m (~13 s/step, one rack, 33 GB peak); checkpoints every 150 steps
plus `iter_0001821`. The backup resume `47652134` exited with "no samples left to consume" (expected: nothing
left to train).

Export: job `47693179`, HF vs Megatron mean loss 0.92165 vs 0.92168 on the 16 stage-1 sequences (max per-sequence
difference 0.00074; base 2.01169). Copied with SHA256 (16/16 OK) to Jupiter
`/p/scratch/ccstdl/dc-agent-shared/checkpoints/prelude-9b-midtrain-v1-stage2-32k-theta9m-step1821` for JUWELS evals.

RULER on MN5 (vLLM 0.27, same frozen inputs, 1300/1300 examples per context), job `47691164`:

| Context | Stage 1 mean | Stage 2 mean | Stage 2 passkey | Gate |
| ---: | ---: | ---: | ---: | --- |
| 4K | 87.8% | 88.0% | 100% | pass |
| 16K | 73.9% | 78.7% | 100% | fail (mean) |
| 32K | 60.2% (JUWELS) | 71.2% | 100% | fail (mean) |

Per-task, stage 1 -> stage 2:

| Task | 4K | 16K | 32K (stage 2) |
| --- | ---: | ---: | ---: |
| multikey-2 | 100 -> 100 | 75 -> 91 | 68 |
| multikey-3 | 100 -> 100 | 37 -> 72 | 33 |
| multiquery | 100 -> 100 | 87 -> 94.5 | 91 |
| CWE | 71.4 -> 87.2 | 30.8 -> 55.4 | 16.2 |
| FWE | 49.7 -> 74.0 | 64.3 -> 66.7 | 84.7 |
| QA SQuAD | 70.4 -> 40.7 | 38.8 -> 34.4 | 26.0 |
| QA Hotpot | 56.0 -> 47.0 | 36.0 -> 15.0 | 20.0 |

Other tasks are 96-100% at every context.

Stage 2 improves retrieval and aggregation at 16K and restores 4K FWE (50 -> 74), but QA gets worse. Much of the QA
loss is a format failure, not a retrieval failure: on these plain-text prompts the model increasingly opens a ChatML
assistant turn with `<think>` instead of answering. Outputs containing `<|im_start|>` or `<think>`:

| | 4K | 16K | 32K |
| --- | ---: | ---: | ---: |
| stage 1 (all tasks) | 0 | 80 | - |
| stage 2 (all tasks) | 8 | 120 | 156 |
| stage 2, of the 200 QA prompts | 8 | 107 | 156 |

Most of these hit the generation limit (16K: 90/100 SQuAD and 88/100 Hotpot finish on length). This is consistent
with the ~30-40% ChatML conversation share in the mixture being triggered by long plain-text documents. Options: score
QA from the first answer line as well as the frozen scorer (diagnostic only, not a gate change), run RULER QA with the
ChatML template, or lower the conversation share in stage 3. The preregistered plain-text gate remains failed.

QA split on the plain-text outputs: lm-eval's QA scorer already accepts the reference anywhere in the output, so
first-line truncation cannot raise scores. Instead, accuracy on outputs that stayed plain text vs drifted to ChatML:
Hotpot 16K stage 1 50% (48) vs 23% (52), stage 2 47% (19) vs 7% (81); SQuAD 16K stage 2 47% vs 0%; Hotpot 32K
45% vs 9%. The separate 4K SQuAD drop (70 -> 41) has no drift: in 28 of the 35 lost answers stage 2 emits
`\nAnswer:` and then EOS, the same early-EOS failure the initialized model showed.

ChatML diagnostic (not a gate): the same frozen documents wrapped as a user turn, the assistant turn opened with the
empty think block and the task's `gen_prefix`, window context + 64 tokens, same decoding and scorer
(`gates_20261018/infer_ruler_chatml.py`, job `47708014`). Plain-text re-encoding of the documents matches the
frozen IDs for all 1300 prompts.

| Checkpoint | Context | Plain mean | ChatML mean | SQuAD plain -> ChatML | Hotpot plain -> ChatML |
| --- | ---: | ---: | ---: | ---: | ---: |
| stage 1 | 4K | 87.8 | 89.4 | 70 -> 71 | 56 -> 43 |
| stage 1 | 16K | 73.9 | 74.2 | 39 -> 54 | 36 -> 41 |
| stage 2 | 4K | 88.0 | 89.8 | 41 -> 62 | 47 -> 41 |
| stage 2 | 16K | 78.7 | 79.7 | 34 -> 57 | 15 -> 37 |
| stage 2 | 32K | 71.2 | 70.9 | 26 -> 58 | 20 -> 40 |

ChatML recovers QA to stage-1 levels, so the plain-text QA loss is a format effect. The means barely move because
ChatML costs multi-key retrieval at long context (stage 2 32K: multikey-2 68 -> 37, multikey-3 33 -> 25,
multivalue 93 -> 78; 16K multikey-2 91 -> 64), often answering with a distractor key. Multi-key retrieval at 32K is
the remaining weakness in both formats. Passkey is 100% everywhere.

JUWELS/JURECA stage-2 evals (vLLM 0.26; records `../rl/evals/runs/20261021-prelude-midtrain-s2-32k-t9m-s1821-*`,
summary in the [gates README](../evals/prelude_midtrain/README.md)):

| RULER mean | Stage 1 | Stage 2 | Stage 2 passkey |
| ---: | ---: | ---: | ---: |
| 4K | 87.4 | 87.8 | 100 |
| 16K | 74.1 | 78.9 | 100 |
| 32K | 60.2 | 70.8 | 100 |
| 64K | 12.6 | 21.1 | 90 |

These agree with the MN5 replay within 0.4 pp at 4K-32K. 64K is still untrained (stage 3).

ChatML GSM8K / MATH500 / HumanEval+, thinking off / on, 100 examples each
([4K](../evals/runs/20261021-prelude-midtrain-s2-32k-t9m-s1821-chatml.yaml),
[16K window, 14336-token budget](../evals/runs/20261021-prelude-midtrain-s2-32k-t9m-s1821-chatml16k.yaml)):

| Task | Stage 1 4K | Stage 2 4K | Stage 1 16K | Stage 2 16K |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 87 / 79 | 84 / 87 | 87 / 81 | 86 / 90 |
| MATH500 | 57 / 38 | 55 / 41 | 70 / 49 | 73 / 61 |
| HumanEval+ | 23 / 31 | 49 / 55 | 23 / 37 | 50 / 61 |

Stage 2 improves chat formatting: thinking HumanEval+ generation-limit hits fall from 65 to 34 (4K) and 25 (16K).
Thinking-on HumanEval+ is still below the initialized model's 71 at 4K.

Plain-text few-shot, 4K reference protocol
([record](../evals/runs/20261021-prelude-midtrain-s2-32k-t9m-s1821-base-fewshot.yaml)):

| | MATH | GSM8K | MMLU-Pro | HumanEval+ | MBPP+ | MultiPL-E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dominant2 parent | 47.8 | 80.5 | 42.2 | 54.3 | 38.1 | 24.7 |
| stage 1 | 47.4 | 79.4 | 43.0 | 54.3 | 47.6 | 33.6 |
| stage 2 | 47.0 | 80.1 | 43.9 | 51.8 | 35.7 | 33.8 |
| stage 2 vs parent (pp) | -0.8 | -0.4 | +1.7 | **-2.5** | **-2.4** | +9.1 |
| margin (pp) | -2 | -2 | -1.5 | -2 | -2 | -2 |

Stage 2's HumanEval+ and MBPP+ point estimates fall outside the frozen 2 pp margins. Both are noisy (standard error
about 3.9 and 2.6 pp), so the paired procedure (`compare_regression.py`) may call them inconclusive rather than
failed; it has not been run. MBPP+ moving 38 -> 48 -> 36 across stages is large enough to check for a
format/stop effect before reading it as lost skill. The 16K-window / 4096-token rerun
([record](../evals/runs/20261021-prelude-midtrain-s2-32k-t9m-s1821-base-fewshot-ctx16k.yaml)) is within about
1 point of the 4K numbers.

### Stage 3 (64K; 2026-10-21)

Decision (user): keep the current stage-3 mixture (the ChatML diagnostic shows the QA loss is format), and proceed
although the 16K/32K mean targets are failed. Smoke `47708082` (12 steps from step 1821, debug queue): TP4 CP1 full
recompute fits at 48.6 GB peak, ~13.3 s/step, LR continuous (6.9111e-6 at step 1821), data from sample 77696;
its checkpoint was deleted. Run `stage3-64k-theta9m`: 41,554 rows (2.0B tokens), GBS 16 x 64K, steps 1821-4419,
single 12h job `47711650` (checkpoints every 150 steps, save before the limit). Export + RULER 4K/16K/32K/64K
gates `47711651` start after it ends (`afterany`).

Training `47711650` completed all 2598 steps in 9h58m (~12.5-13.5 s/step); LR ended at the 1e-6 floor. 400-step
mean loss stayed flat at 0.861-0.876. Consumed samples ended at 119264, 14 past the 119250 rows (41,554 does not
divide by 16). Export check `47711651`: HF 0.91635 vs Megatron 0.91631 (max per-sequence 0.00061; base 2.01169).
Copied with SHA256 (16/16 OK) to Jupiter
`/p/scratch/ccstdl/dc-agent-shared/checkpoints/prelude-9b-midtrain-v1-stage3-64k-theta9m-step4419`.

RULER on MN5 (vLLM 0.27, frozen inputs, 1300/1300 per context); ChatML is the diagnostic wrapper above
(job `47778137`, 64K window 65536 + 64 with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`):

| Context | Stage 2 plain | Stage 3 plain | Stage 3 passkey | Stage 3 ChatML | ChatML passkey |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 88.0 | 83.9 | 100 | **90.2** | 100 |
| 16K | 78.7 | 80.5 | 100 | 83.2 | 100 |
| 32K | 71.2 | 75.0 | 100 | 75.9 | 100 |
| 64K | 21.1 (JUWELS) | 56.3 | 100 | 54.8 | 88 |

Plain-text gate: fails the mean at every context, including 4K, which passed after stages 1 and 2. Passkey passes
everywhere.

Stage 3 gains on multi-key retrieval: 16K multikey-3 72 -> 93; 32K multikey-2 68 -> 92, multikey-3 33 -> 65. At 64K the
weak tasks are multikey-3 8, CWE 12, Hotpot 7, SQuAD 29 and variable tracking 31.

The 4K plain-text drop is format, not retrieval:
- SQuAD falls 41 -> 24. 57 of its 78 misses are `\nAnswer:` followed immediately by EOS.
- FWE falls 74 -> 46. All 62 misses fill the token budget with newlines.
- 4K ChatML recovers both: SQuAD 63, FWE 89.

As at stage 2, ChatML lowers multi-key retrieval at 16-32K:
- 32K multikey-2 92 -> 69, multikey-3 65 -> 50.
- 64K single-needle 100 -> 88 and 97 -> 80.

ChatML outputs counted in the plain-text runs: 18 (4K), 74 (16K), 99 (32K), 98 (64K).

Summary: the three-stage run delivers 64K retrieval (passkey 100%, single-needle 82-100%), with the strongest
multi-key retrieval seen so far at 16-32K, but not the preregistered 85% means. Short-context plain-text answers
degrade in termination/format, not knowledge.

JUWELS/JURECA stage-3 evals (records `../rl/evals/runs/20261022-prelude-midtrain-s3-64k-t9m-s4419-*`, details in the
[gates README](../evals/prelude_midtrain/README.md)):
- RULER: 84.2 / 80.2 / 74.5 / 56.6 at 4K-64K, within 0.5 pp of MN5.
- ChatML at 4K, thinking off / on: GSM8K 88/85, MATH500 50/37, HumanEval+ 67/57.
  - Thinking-off HumanEval+ budget hits fall 34 -> 0.
  - At 16K, thinking-on MATH500 reaches 69.

Paired non-inferiority vs the dominant2 parent (`compare_regression.py`, 20,000 replicates, Bonferroni): no stage
passes all six metrics and none fails any. Stage 3:
- Pass: MATH (-0.2), MMLU-Pro (+1.3), MultiPL-E (+8.8).
- Inconclusive: GSM8K (-1.3), HumanEval+ (-5.5 [-13.4, +2.4]), MBPP+ (-5.3 [-11.6, +1.1]).

MBPP+ is a stopping artifact: completions run on past the function (no `\ndef` stop) and are cut mid-line.
Re-grading on the longest parseable prefix (diagnostic only) gives parent 39.9, stage 1 54.8, stage 2 54.8,
stage 3 53.4. HumanEval+ is not affected by that artifact. Its stage-3 point drop (54.3 -> 48.8) is unexplained
and remains inconclusive.

The [evaluation registry](../evals/README.md) contains all completed measurements. The
[paired historical reference](../evals/prelude_midtrain/base_fewshot_paired_reference.json)
exports the original full six-benchmark run and reproduces every registered headline score;
stage 1 is the first trained stage measured under the matched few-shot protocol (above).

### 128K with YaRN x2 (inference only, 2026-10-22)

The stage-3 HF export with `rope_type=yarn`, factor 2, `original_max_position_embeddings=65536`,
`max_position_embeddings=131072` (theta 9M unchanged; weights symlinked,
`hf/iter_0004419-yarn2-128k` on MN5). The 128K inputs were generated with the same frozen generator, 13 tasks x 100
(`gates_20261018/inputs_ruler_131072`). MN5 job 47801975, 3h42m on one node.

| RULER mean (passkey) | 4K | 64K | 128K |
|---|---|---|---|
| Stage 3, no YaRN, plain | 83.9 (100) | 56.3 (100) | - |
| Stage 3 + YaRN x2, plain | 83.5 (100) | 61.4 (100) | 43.7 (86) |
| Stage 3 + YaRN x2, ChatML | 89.0 (100) | 57.6 (100) | 30.6 (3) |

- YaRN keeps 4K (-0.4) and raises 64K plain by 5.1, from VT 31 -> 76, multi-value 59 -> 87, multi-query
  72 -> 86 and single-3 82 -> 99. Multikey-2 drops 74 -> 28.
- 128K plain-text by task: single-needle 86/96/80, multi-key 65/25/0, multi-query 54, multi-value 52, VT 36,
  FWE 64, CWE 1, SQuAD 5, Hotpot 5.
- 128K ChatML collapses on format: most single-1 answers are `:` or a truncated number followed by EOS. It is not
  a retrieval measure.

Summary: YaRN x2 is safe to ship on the 64K checkpoint (no loss at 4K, gain at 64K) and gives usable 128K
single-needle retrieval, but 128K multi-hop/aggregation is weak. A short 128K training stage would be needed to
claim 128K quality.

### Stage 4 (128K, YaRN x2 trained; 2026-10-23)

Short 128K stage from the stage-3 checkpoint with YaRN x2 (theta 9M, `original_max_position_embeddings=65536`)
applied in training. Selection: [select_prelude_stage4_128k.py](../scripts/data/select_prelude_stage4_128k.py)
(8,012 rows, 999.46M seen tokens, fill 95.2%), exported with `export_prelude_midtrain.py --mixture` and packed
with [pack_prelude_stage_megatron.py](../scripts/data/pack_prelude_stage_megatron.py). LR 5e-6, 1,001 steps,
MN5 job 47820790 COMPLETED in 5h57m (~21 s/step, no OOM).

- Export: the `yarn_*` fields are not Megatron provider dataclass fields, so `run_config.yaml` kept only
  `position_embedding_type: yarn` and reload failed (`AttributeError: yarn_rotary_scaling_factor`). Fixed by
  setting YaRN class defaults on the provider when `YARN_FACTOR` is set and writing YaRN `rope_parameters` in the
  HF config. Parity (post job 47858217): HF 0.91617 vs Megatron 0.91602 mean, max per-sequence diff 0.00079.
- HF export: `runs/stage4-128k-yarn2/hf/iter_0001001` on MN5; Jupiter
  `/p/scratch/ccstdl/dc-agent-shared/checkpoints/prelude-9b-midtrain-v1-stage4-128k-yarn2-step1001`
  (sha256 verified, 16 files).

| RULER mean (passkey) | 4K | 64K | 128K |
|---|---|---|---|
| Stage 3 + YaRN x2 (inference), plain | 83.5 (100) | 61.4 (100) | 43.7 (86) |
| Stage 4, plain | 85.0 (100) | 65.8 (100) | 50.4 (99) |
| Stage 3 + YaRN x2 (inference), ChatML | 89.0 (100) | 57.6 (100) | 30.6 (3) |
| Stage 4, ChatML | 90.9 (100) | 64.4 (100) | 45.0 (39) |

Stage 4 improves every cell, passes the 85% 4K gate and fixes 128K passkey (86 -> 99 plain). ChatML 128K is
still format-limited (passkey 39). Recommendation: stage 4 is the base for the full SFT v2 run (0058); the SFT
smoke continues from stage 3. Chat/agentic evals of stage 4 are pending.

## Acceptance

- [ ] A fresh invocation can reproduce the clean release and reports from the pinned inputs without manual edits.
- [x] Semantic tool-response masks, earlier reasoning turns, source overlaps, and packed-loader behavior have
      explicit checks; SFT v1 behavior remains covered.
- [x] Every token allocation is backed by available clean data, including replay, with a documented split and
      no unaccounted duplicates between allocations or the reserved downstream SFT data.
- [ ] The inventory, release manifest, pilot evidence, and evaluation thresholds identify the exact artifacts to use
      for a subsequent staged training run.

## Related

- [Prelude base selection and original mid-training plan](../evals/analysis/prelude-sft-base-selection/README.md).
- [Prelude SFT v1 inventory](../inventory/datasets/prelude-sft-v1.yaml) and
  [build entrypoint](../sft/prelude/build_sft_v1.sbatch).
- [Conversation decontamination](../scripts/data/decontam/decontam_conversations.py) and
  [benchmark index specification](../scripts/data/decontam/benchmark_index_prelude.json).
- 0048: Grug/Snowball SFT data work; 0051: RTS/TMax teacher trace releases reserved for agentic SFT.
