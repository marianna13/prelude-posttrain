# Prelude 9B training data: mid-training (stages 1-4) and SFT v2

One place for the data that went into the Prelude 9B context-extension stages and SFT v2: every source, its weight
and the processing. Details and job IDs are in issues
[0054](../../../issues/0054-build-prelude-9b-midtraining-data.md) (mid-training) and
[0058](../../../issues/0058-build-prelude-sft-v2-intellect-3-reasoning-chat-to.md) (SFT v2).

## Mid-training, stages 1-3 (16K / 32K / 64K, 3.5B seen tokens)

Inventory: [prelude-midtrain-v1.yaml](../../inventory/datasets/prelude-midtrain-v1.yaml). Exact files: on JURECA,
`/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step1/source_selection.json` (972 pinned files
with SHA256) and `step4/mixture_v3/mixture_manifest.json` (frozen recipe, SHA256 `a03d76a8...`). Epochs per
source are in `step4/.../source_exposure_report.json`.

### Upstream sources

| Collection | Revision | Used for |
|---|---|---|
| `nvidia/Nemotron-Cascade-2-SFT-Data` | `9f36020d` | math, science, SWE, chat, IF conversations (`terminal_agent` excluded) |
| `nvidia/Nemotron-Pretraining-Specialized-v1` | `9ed3718b` | InfiniByte-Reasoning, STEM-SFT, RQA, Math-Textbooks (Wiki Rewrite excluded) |
| `nvidia/Nemotron-Terminal-Corpus` | `a1667c4f` | terminal agent conversations |
| `openbmb/UltraData-Stack-v3-L2` | `d6bdb3e7` | code files, 11 languages |
| `HuggingFaceCode/stack-v3-train` | `1f61b735` | whole repositories (code and long documents) |
| DCLM baseline 1.0 | local, pinned | replay: 16 files x first 100K rows |
| Nemotron-CC, original / high quality | local, pinned | replay: 8 files x first 100K rows |
| FinePDFs English | local, pinned | replay: 1 file; long documents: a separate file, originals >= 32K tokens |
| FinePDFs, 23 EU languages | local, pinned | replay: first 50K rows of one file per language |

AgentTrove, RTS and TMax were kept out of mid-training (reserved for SFT and RL).

### Category targets (seen tokens; the same in every stage)

| Category | Share | Documents | Conversations |
|---|---:|---:|---:|
| Reasoning / math / science | 30% | 14.53% | 15.47% |
| Code / agent | 25% | 15.47% | 9.53% |
| Chat / instruction following | 10% | 0% | 10% |
| Long documents | 12.5% | 12.5% | 0% |
| Short replay (<= 4K tokens) | 22.5% | 22.5% | 0% |
| **Total** | **100%** | **65%** | **35%** |

### Per-stage totals

| Stage | Context | Parents | Seen tokens | Trained tokens |
|---|---|---:|---:|---:|
| 1 | 16K | 186,595 | 499,918,595 | 489,184,884 |
| 2 | 32K | 347,764 | 999,867,143 | 973,966,082 |
| 3 | 64K | 673,473 | 1,999,735,426 | 1,935,346,645 |
| **1-3** | | **1,207,832** | **3,499,521,164** | **3,398,497,611** |
| 4 | 128K (YaRN x2) | 8,012 rows | 999,460,000 | - |

Stage 4 is a separate selection from the same curated corpus,
[select_prelude_stage4_128k.py](../../scripts/data/select_prelude_stage4_128k.py) (fill 95.2%), and is not in
the inventory file.

### Weights for all 52 pools (seen-token share, stages 1-3)

From [step4_evidence.json](step4_evidence.json) `source_weights`. A weight of 0 means the pool has no originals
of at least 32K tokens, so it could not keep a fixed weight across stages.

| Collection | Pool | Weight |
|---|---|---:|
| `Nemotron-Cascade-2-SFT-Data` | `math` | 13.411% |
| `Nemotron-Cascade-2-SFT-Data` | `science` | 2.063% |
| `Nemotron-Cascade-2-SFT-Data` | `swe` | 4.763% |
| `Nemotron-Cascade-2-SFT-Data` | `chat` | 8.000% |
| `Nemotron-Cascade-2-SFT-Data` | `instruction_following` | 2.000% |
| `Nemotron-Cascade-2-SFT-Data` | `conversational_agent` | 0.000% |
| `Nemotron-Pretraining-Specialized-v1` | `InfiniByte-Reasoning` | 9.361% |
| `Nemotron-Pretraining-Specialized-v1` | `STEM-SFT` | 4.680% |
| `Nemotron-Pretraining-Specialized-v1` | `RQA` | 0.481% |
| `Nemotron-Pretraining-Specialized-v1` | `Math-Textbooks` | 0.004% |
| `Nemotron-Pretraining-Specialized-v1` | `Scientific-Coding` | 0.000% |
| `Nemotron-Terminal-Corpus` | - | 4.763% |
| `UltraData-Stack-v3-L2` | `php` | 5.098% |
| `UltraData-Stack-v3-L2` | `java` | 2.396% |
| `UltraData-Stack-v3-L2` | `cpp` | 2.257% |
| `UltraData-Stack-v3-L2` | `javascript` | 1.568% |
| `UltraData-Stack-v3-L2` | `python` | 1.453% |
| `UltraData-Stack-v3-L2` | `csharp` | 0.923% |
| `UltraData-Stack-v3-L2` | `shell` | 0.842% |
| `UltraData-Stack-v3-L2` | `go` | 0.369% |
| `UltraData-Stack-v3-L2` | `ruby` | 0.286% |
| `UltraData-Stack-v3-L2` | `rust` | 0.151% |
| `UltraData-Stack-v3-L2` | `r` | 0.132% |
| `stack-v3-train` | `data/long_documents` | 6.250% |
| `stack-v3-train` | `data/code` | 0.000% |
| `FinePDFs-English-long` | `long_documents` | 6.250% |
| `dclm-baseline-1.0` | `dclm_english` | 8.929% |
| `Nemotron-CC-original-high` | `nemotron_english` | 6.589% |
| `FinePDFs-English-replay` | `pdf_english` | 2.932% |
| `FinePDFs-EU-replay` | `bul_Cyrl` | 0.191% |
| `FinePDFs-EU-replay` | `ces_Latn` | 0.162% |
| `FinePDFs-EU-replay` | `dan_Latn` | 0.179% |
| `FinePDFs-EU-replay` | `deu_Latn` | 0.176% |
| `FinePDFs-EU-replay` | `ekk_Latn` | 0.178% |
| `FinePDFs-EU-replay` | `ell_Grek` | 0.189% |
| `FinePDFs-EU-replay` | `fin_Latn` | 0.172% |
| `FinePDFs-EU-replay` | `fra_Latn` | 0.184% |
| `FinePDFs-EU-replay` | `gle_Latn` | 0.119% |
| `FinePDFs-EU-replay` | `hrv_Latn` | 0.168% |
| `FinePDFs-EU-replay` | `hun_Latn` | 0.168% |
| `FinePDFs-EU-replay` | `ita_Latn` | 0.192% |
| `FinePDFs-EU-replay` | `lit_Latn` | 0.172% |
| `FinePDFs-EU-replay` | `lvs_Latn` | 0.174% |
| `FinePDFs-EU-replay` | `mlt_Latn` | 0.217% |
| `FinePDFs-EU-replay` | `nld_Latn` | 0.190% |
| `FinePDFs-EU-replay` | `pol_Latn` | 0.164% |
| `FinePDFs-EU-replay` | `por_Latn` | 0.168% |
| `FinePDFs-EU-replay` | `ron_Latn` | 0.179% |
| `FinePDFs-EU-replay` | `slk_Latn` | 0.184% |
| `FinePDFs-EU-replay` | `slv_Latn` | 0.174% |
| `FinePDFs-EU-replay` | `spa_Latn` | 0.179% |
| `FinePDFs-EU-replay` | `swe_Latn` | 0.172% |

### Processing

| Step | What was done | Docs |
|---|---|---|
| 1. Pin and count | Exact files and revisions with SHA256; every row counted with the Prelude tokenizer and ChatML template | 0054 step 1 |
| Code selection | Whole stack-v3 repositories with >= 8K tokens, >= 20% of files in UltraData-L2, ranked by L2 share x L2 quality; permissive + no-license, vendored files excluded (128,122 repositories, 5.86B tokens) | 0054 step 1 |
| 2. Format and masks | Documents: raw text, BOS/EOS, all tokens trained except BOS. Conversations: ChatML + `<think>`, earlier reasoning kept, everything trained except whole tool-response groups; Terminal-Corpus output in `user` turns treated as tool turns. Six new chat-token embeddings set to the mean of the existing rows | [MIDTRAIN.md](MIDTRAIN.md) |
| 3. Dedup and decontamination | See the removal table below; 1% validation split held out by task / repository / document. No near-duplicate pass | [CURATE_MIDTRAIN.md](CURATE_MIDTRAIN.md) |
| 4. Mixture | Fixed category and format weights; whole parents only, each used once across stages; preferred original length >= 8K / 16K / 32K per stage (replay and chat/IF exempt); deterministic seeded order | [MIXTURE_MIDTRAIN.md](MIXTURE_MIDTRAIN.md) |
| 5. Export and chunking | Parents longer than the context are split; each continuation gets a masked BOS + 256 masked tokens of prior context; nothing truncated, original EOS kept | [EXPORT_MIDTRAIN.md](EXPORT_MIDTRAIN.md) |

Step-3 removals, applied in this order (from 197,537,512 accepted rows):

| Removal | Rows |
|---|---:|
| Prompt overlap with SFT v1 tasks | 247,332 |
| Benchmark 13-gram match (`index_prelude_v1`, 290 reviewed boilerplate n-grams whitelisted) | 401,016 |
| UltraData file already in a selected Stack repository | 1,642,127 |
| Exact task / document duplicate | 25,931,369 |
| **Retained** | **169,315,668** |

Rejected before that: 704,569 rows with ambiguous tool-response boundaries, 46 rows with unknown roles, 7 with
malformed inline reasoning, 1,129,556 ineligible UltraData files.

## SFT v2 (37.7B seen / 30.1B trained tokens)

Data on JURECA: `/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v2/{census,decontam,selection,normalized,rendered}`.
Scripts: `rl/scripts/data/sft_v2/`. Trained in one pass at 64K from stage 4 (9,020 steps).

### Upstream sources

| Source | Revision | Licence |
|---|---|---|
| `PrimeIntellect/INTELLECT-3-SFT` | `ff48f664` | card empty; check the upstream datasets |
| `open-thoughts/AgentTrove` | `b395a430` | apache-2.0 |
| `nebius/SWE-rebench-openhands-trajectories` | `35455389` | cc-by-4.0 |
| SFT v1 `nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` rows | `be3b3e04` | see [prelude-sft-v1.yaml](../../inventory/datasets/prelude-sft-v1.yaml) |

Used only for overlap checks: `PrimeIntellect/Scale-SWE-Verified` `2c29b67e` (the SWE RL pool) and
`princeton-nlp/SWE-bench_Verified` `c104f840`.

### Mix

| Mix source | Selection | Rows | Seen tokens | Trained tokens |
|---|---|---:|---:|---:|
| `i3_math` (`openreasoning_math`) | <= 4 traces per prompt | 634,354 | 11.97B | |
| `i3_code` (`openreasoning_code`) | <= 12 traces per prompt | 406,380 | 6.91B | |
| `i3_science` (`openreasoning_science`) | 53% sample | 847,922 | 5.48B | |
| `i3_am_chat` + `i3_am_if` (thinking on) | all | 1,005,318 | 2.37B | |
| `i3_tool` + `i3_toucan` (native tool calls) | all | 422,438 | 1.59B | |
| `i3_swe_swiss` (issue to patch) | all | 9,934 | 0.17B | |
| AgentTrove (Terminus-2) | behavioural cut, longest episode per (trial, run), (source, teacher) capped at 284M tokens | 595,509 | 7.29B | 2.48B |
| nebius SWE-rebench-OpenHands (native tools) | resolved, ended with submit, <= 64K rendered | 20,256 | 1.04B | 0.26B |
| `v1_if_chat_nothink` + `v1_if_instruction_following_nothink` | SFT v1 IF/chat, reasoning stripped, empty think block | 398,253 | 0.89B | |
| **Total** (after rendering) | | **4,340,196** | **37.69B** | **30.09B** |

### Decontamination drops (`decontam_drop`)

| Source | Rows | Dropped rows | Dropped tokens | Main reason |
|---|---:|---:|---:|---|
| nebius | 67,074 | 3,761 | 241M | SWE overlap with Scale-SWE / SWE-bench Verified 3,126; benchmark 674 |
| AgentTrove | 1,696,847 | 3,435 | 62M | 11 TBLite tasks verbatim (2,012 rows); benchmark 3,363 |
| `openreasoning_math` | 2,044,407 | 5,878 | 94M | benchmark |
| `openreasoning_code` | 1,896,395 | 3,101 | 40M | benchmark |
| `openreasoning_science` | 1,600,812 | 312 | 3.1M | benchmark |
| `am_chat` | 952,046 | 1,344 | 7.4M | benchmark |
| `am_if` | 54,720 | 104 | 0.2M | benchmark |
| `swe_swiss` | 10,254 | 63 | 1.7M | benchmark (SWE-bench issue text) |
| `openreasoning_tool` | 310,051 | 16 | 0.1M | benchmark |
| `toucan_tool` | 115,724 | 5 | 0.03M | benchmark |

### Processing

| Step | What was done |
|---|---|
| 1. Census | Every row tokenized with the Prelude chat tokenizer; SWE overlap matched on repo + PR and on >= 20 shared 13-grams of issue text |
| 2. Decontamination | 13-gram match against `index_prelude_v1` (whole Prelude eval suite incl. SWE-bench Verified, TB2.1, TBLite); n-grams matched by > 20 distinct prompts treated as boilerplate. Instructions matched against TMax-1k, RTS-1k, CalibForge, TBLite, TB2.1. Flagged rows dropped at mix time (0.48B of 104B tokens). Exact prompt duplicates vs Cascade-2 / SFT v1 flagged but **not** dropped |
| 3. Normalize | Reasoning kept only after the last user turn; tool calls rendered hermes-style with tools in the system prompt; only assistant turns trained, tool/environment turns masked; AgentTrove turns trained only if valid Terminus JSON or a handoff reply |
| 4. Render | Prelude ChatML + `<think>`, 64K cap (842 nebius rows over 64K and 168 unrenderable rows dropped) |
| 5. Loop scan | 570 rows with repetition loops or garbage tails in reasoning (0.01%) excluded at packing |
| 6. Pack | Megatron packing at 64K (`megatron_packed/full_v2/65536`), one pass |

## Known gaps

- No near-duplicate dedup in either pipeline (exact only).
- INTELLECT-3 licences are not resolved (the dataset card is empty).
- No inventory file for SFT v2 or for the stage-4 selection.
