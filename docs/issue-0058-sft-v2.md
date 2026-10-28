---
title: "Build Prelude SFT v2: INTELLECT-3 reasoning/chat/tools + AgentTrove, multi-harness"
labels: [agent-generated, data, sft]
status: open
gh_number: 62
created: 2026-10-22
---

## Goal

Build the SFT mix for the mid-trained Prelude 9B, then run RLVR (math/code) and agentic RL. Start from
`prelude-9b-midtrain-v1-stage3-64k-theta9m-step4419`, the 64K, theta-9M result of
[0054](0054-build-prelude-9b-midtraining-data.md). The model must work in two agent harnesses:

- **Terminus-2:** inline JSON actions (`analysis` / `plan` / `commands` / `task_complete`) in assistant text,
  with terminal output in `user` turns.
- **Native tool calls:** `tools` schema in the prompt, structured `tool_calls` in assistant turns and `tool`-role
  results, rendered Qwen3/hermes style (`<tool_call>` JSON) by the Prelude chat template, as SFT v1 does.

It should also support thinking on (default) and thinking off.

## Sources

| Source | Use | Rows | Thinking | Tool format | Licence |
| --- | --- | ---: | --- | --- | --- |
| [PrimeIntellect/INTELLECT-3-SFT](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-3-SFT) `openreasoning_math` / `_code` / `_science` | reasoning, RLVR prior | 2.04M / 1.9M / 1.6M | yes | none | card empty; check upstream |
| INTELLECT-3 `am_chat`, `am_if` | chat / IF (thinking on) | 952k / 54.7k | yes | none | card empty; `am_chat` is from a-m-team/AM-DeepSeek-R1-0528-Distilled |
| INTELLECT-3 `openreasoning_tool` | native tool calls, multi-turn | 310k | yes | `tool_calls` + `tools` | check upstream |
| INTELLECT-3 `toucan_tool` | native tool calls | 116k | **no** | `tool_calls` + `tools` | check upstream |
| INTELLECT-3 `swe_swiss` | single-turn issue to patch | 10.3k | yes | none | check upstream |
| [open-thoughts/AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove) @ `b395a43` | Terminus-2 agentic | 1.70M published, 1.24M unique usable | mixed teachers | Terminus-2 inline JSON | apache-2.0 |
| [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | native-tool SWE agent (OpenHands) | 67,074 (32,161 resolved) | **no** (Qwen3-Coder-480B-A35B-Instruct) | OpenHands v0.54 function calling, `tool` role | cc-by-4.0 |
| ~~[nvidia/SWE-Zero-openhands-trajectories](https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories)~~ (not used) | native-tool SWE agent, volume | 318,115 over 118,092 issues; **no success label** | not stated (Qwen3-Coder-480B) | OpenHands, `tool` role | cc-by-4.0 (MIT/Apache/BSD repos only) |
| Non-thinking IF/chat (see below) | thinking-off mode | TBD | **no** | none | - |

Sampled rows (20 per split, 2026-10-22): `am_chat`, `am_if`, `openreasoning_tool` and `swe_swiss` all carry
`<think>` in the completion; `toucan_tool` has none. Schema: `source`, `prompt`, `completion` (lists of
`{role, content, tool_calls}`), plus `tools` on the tool splits.

AgentTrove facts, from the 2026-09-25 assessment in [0048](0048-experiment-sft-data-snowball-grug-push.md) and the
[inventory](../inventory/datasets/agenttrove-sft-traces-v1.yaml):
- The column is `conversations`. There is no `reward` column, and identity is `trial_name` + `run_id` + `episode`.
- 7.6% of rows are null, 12.7% are exact duplicates and 7.0% are variants. `result` is on only 12.9% of rows and is
  mostly timeouts, so do not filter on reward.
- Filter on behaviour instead (the rule adopted in 0048):
  - the last turn is from the assistant;
  - it contains `task_complete`, or `result`/`judgment` reports success;
  - its JSON is brace-balanced (not cut mid-object);
  - it is non-empty.
- Many teachers (GLM-4.6/4.7/5.0, GPT-5 family, Kimi, MiniMax, Qwen3, Gemini). Balance by teacher and source dataset.
- It is already materialized at `/e/data1/datasets/playground/ot/agenttrove-sft-v1/`.

## Decisions (2026-10-22, user)

- **Terminal agent data:** AgentTrove, not Nemotron-Terminal-Corpus. Terminal-Corpus is already in the mid-training
  mix (0054), so it would be a second exposure.
- **TMax-1k / RTS-1k:** not used in SFT. They stay as RL task pools. Other RL datasets would be needed before
  spending them on SFT.
- **Native-tool SWE SFT:** nebius SWE-rebench-OpenHands only (not SWE-Zero).
- **SWE agentic RL pool:** Scale-SWE-Verified (not SWE-rebench / SWE-Lego). AgentTrove's R2E-Gym, SWE-smith and
  SWE-Gym traces stay in SFT; AgentTrove has no TMax/RTS.
- **Non-thinking IF/chat:** add some.
- **Multi-harness agent training:** Terminus-2 and native tool calls.

## Plan

1. **Pin and census.** Download INTELLECT-3-SFT at a pinned revision to JURECA `/p/data1`. Record rows, tokens
   (Prelude tokenizer), turn counts, thinking share, tool-call shape and length distribution per split. Resolve
   licences from the upstream datasets before any publication.
2. **Normalize** to the SFT v1 schema with `rl/scripts/data/materialize_prelude_sft_v1.py` conventions:
   - Keep reasoning on assistant turns after the last user message.
   - Map `tool_calls` to the template's hermes rendering, with `tools` in the system prompt.
   - Mask `tool` / environment turns from the loss.
   - AgentTrove: map terminal output in `user` turns to non-trained context (the Terminal-Corpus census rule in
     0054). Train every assistant turn.
3. **Thinking-off data.** Two candidates:
   - Nemotron IF-Chat v3 rows rendered with the empty think block (`<think>\n\n</think>`). Already normalized for
     SFT v1, so the source is ready.
   - Or `am_chat` / `am_if` with the reasoning stripped and the final answer kept.
   Target about 10-15% of chat/IF rows. Also keep `toucan_tool` as the non-thinking native-tool source.
4. **Multi-harness agent data.**
   - Terminus-2: the AgentTrove behavioural cut, balanced by teacher and source.
   - Native tools: `openreasoning_tool` + `toucan_tool`. These are general API tools, not a terminal.
   - Gap: no native-tool *terminal/SWE agent* trajectories. Option: mechanically convert a share of AgentTrove
     Terminus-2 trajectories to native calls (each `commands` batch becomes a `bash`/`execute` tool call, each
     terminal output a `tool` turn, `task_complete` a finish tool). The same behaviour would then appear under
     both formats.
   - The convertor and target tool schema must match the native-tool harness used for agentic RL. **Decide which
     harness** (for example a Harbor native-tool agent, mini-swe-agent or OpenHands) before converting.
5. **Dedup and decontamination.**
   - Prompt-hash dedup of `openreasoning_*` against SFT v1 (Nemotron Math-v4, OpenCodeReasoning) and against the
     mid-training Cascade-2 conversations. These likely share prompts.
   - Decontaminate with `index_prelude_v1` (`rl/scripts/data/decontam/`): the 35-benchmark index plus AIME25,
     GPQA-D, IFEval, HumanEval+/MBPP+, MultiPL-E, SWE-bench Verified, TB2.1 and TBLite.
   - Also exclude AgentTrove tasks that overlap the **RL task pools** (TMax-1k, RTS-1k, CalibForge) and the agentic
     eval tasks (TB2.1, TBLite, SWE-bench Verified). Match by task id/instruction hash and 13-gram instruction overlap.
6. **Mix and render.**
   - Choose per-source weights under the **~40B-token** budget (draft split below). Weight the reasoning rows
     toward verifiable and hard problems.
   - Cap at 64K tokens. Render with the Prelude ChatML + `<think>` template.
   - Every assistant turn ends in `<|im_end|>`. 0054 found early-EOS and run-on termination failures; check the
     rendered ends.
7. **Smoke SFT and evals.**
   - Short SFT on MN5 from stage 3.
   - Then the 0054 ChatML suite (thinking on/off), plain few-shot, RULER at 4K/64K, and agentic evals in both
     harnesses (TBLite/TB2.1 with Terminus-2, plus a native-tool harness).

## Token budget (user, 2026-10-22): about 40B tokens

Draft split, to be replaced by real per-source token counts after the step-1 census:

| Bucket | Sources | Draft tokens | Share |
| --- | --- | ---: | ---: |
| Reasoning: math / code / science | INTELLECT-3 `openreasoning_*` (deduped vs SFT v1 / mid-training) | ~22B | 55% |
| Terminus-2 agentic | AgentTrove behavioural cut | ~8B | 20% |
| Native tool calls | SWE-rebench-OpenHands (resolved, minus Scale-SWE / SWE-bench Verified overlap), `openreasoning_tool`, `toucan_tool`, optional converted AgentTrove | ~4B | 10% |
| Chat / IF, thinking on | `am_chat`, `am_if` | ~3B | 7.5% |
| Chat / IF, thinking off | Nemotron IF-Chat v3 empty-think, or stripped `am_*` | ~1.5B | 3.75% |
| SWE (non-agentic) | `swe_swiss`, SFT v1 SWE | ~1.5B | 3.75% |

AgentTrove is Terminus-2 only (inline JSON, no `tool` role). Native-tool agentic data comes from
**SWE-rebench-OpenHands** (user, 2026-10-22):
- **Shape:** real OpenHands v0.54 function-calling runs on SWE-rebench GitHub issues, 1,823 repositories. Columns:
  `trajectory`, `tools`, `resolved`, `exit_status`, `model_patch`, `gen_tests_correct`.
- **Cut:** keep `resolved == 1` (32,161) and `exit_status == 'submit'`. Parse the tool-call `arguments` JSON strings
  before rendering, and render `tools` into the system prompt.
- **Size:** about 64 turns on average (max 100). Measure tokens against the 64K cap. Rough estimate 1-2B tokens;
  replace with the census figure.
- **No reasoning traces:** Qwen3-Coder is non-thinking, so this is a thinking-off agentic source. Either render with
  the empty think block, or generate reasoning later.
- **Contamination:** exclude instances overlapping SWE-bench Verified. Instance ids and issue/patch 13-grams must
  both be checked, and same-repo overlap reported separately.
- **Harness:** its tool schema (OpenHands bash / file-editor / finish) is a natural choice for the native-tool RL
  harness. Converting AgentTrove to native calls becomes optional, to keep terminal-task coverage in both formats.

**SWE-Zero-OpenHands**: considered and **not used** (user, 2026-10-22). Its 318K trajectories are execution-free
(unverified). They come from the same Qwen3-Coder/OpenHands setup as nebius, and draw on SWE-Gym/R2E-Gym issues
that overlap the RL pools.

### Agentic RL SWE pool and SFT exclusions (user, 2026-10-22)

The SWE agentic RL pool is
[PrimeIntellect/Scale-SWE-Verified](https://huggingface.co/datasets/PrimeIntellect/Scale-SWE-Verified):
- 17,202 Python issue-resolving tasks, about 5,200 GitHub repos, cc-by-4.0, verifiers `scaleswe_v1`.
- Each task has a pre-built Docker image (`image_url` + `pre_commands`), `FAIL_TO_PASS` / `PASS_TO_PASS` and a gold
  `patch`.
- "Verified" is Prime Intellect's gold-patch validation (2,979 of 20,181 upstream rows removed), **not** SWE-bench
  Verified.

SWE-rebench-based SWE-Lego is not used for RL, so the nebius SFT trajectories do not need a SWE-Lego exclusion.
AgentTrove contains R2E-Gym, SWE-smith and SWE-Gym traces and no TMax/RTS (user), so it needs no RL-pool exclusion
for those. Remaining checks:
- **Scale-SWE vs SFT:** Scale-SWE is mined from GitHub PRs at large scale, so it may share PRs with SWE-rebench
  (nebius) and with AgentTrove's SWE-smith / SWE-Gym / R2E-Gym tasks. Match on normalized `repo` + PR number
  (from `github_url` / instance ids) and on gold-patch / issue-text hashes. Drop the matching SFT trajectories,
  or drop those tasks from the RL pool. Report both counts.
- **Scale-SWE vs SWE-bench Verified:** the card does not mention decontamination. Remove any Scale-SWE task whose
  repo + PR (or issue) is a SWE-bench Verified instance, so RL does not train on the agentic eval. Same-repo
  overlap is expected (django, sympy, ...); report it but do not drop on repo alone.
- **All SFT sources:** exclude SWE-bench Verified instances, plus the TB2.1 / TBLite agentic eval tasks.


## Step-1 census (2026-10-22, JURECA job 15731519)

Every row was tokenized with the Prelude chat tokenizer: message text plus 5 tokens of template overhead per message;
"trained" means assistant tokens. Scripts are in `rl/scripts/data/sft_v2/` (`census_sft_v2.py`, `swe_overlap.py`,
`census_sft_v2.sbatch`). Sources are pinned under `/p/data1/mmlaion/marianna/sft_sources/` (`PINNED_REVISION` files).
Output is in `/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v2/census/`: per-row parquet plus `*.summary.json`.

| Source | Rows | Tokens | Trained | p50 / p90 tokens | > 64K | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| I3 `openreasoning_math` (Nemotron-Post-Training-v1) | 2,044,407 | 35.86B | 35.68B | 15.9K / 30.7K | 0 | **169,912 prompts, ~15 responses each** |
| I3 `openreasoning_code` (OpenCodeReasoning-2) | 1,896,395 | 29.24B | 28.20B | 14.2K / 29.7K | 0 | **34,150 prompts, ~60 responses each** |
| I3 `openreasoning_science` (OpenScienceReasoning-2) | 1,600,812 | 10.34B | 9.97B | 4.4K / 14.6K | 1 | 1 response per prompt |
| I3 `am_chat` (thinking) | 952,046 | 2.27B | 2.09B | 1.6K / 4.7K | 0 | |
| I3 `am_if` (thinking) | 54,720 | 0.11B | 0.09B | 1.1K / 4.9K | 0 | |
| I3 `openreasoning_tool` (thinking, native) | 310,051 | 1.59B | 1.11B | 4.5K / 9.0K | 0 | 289,718 prompts |
| I3 `toucan_tool` (non-thinking, native) | 115,724 | 0.36B | 0.09B | 1.7K / 7.0K | 201 | tool calls in 73,306 rows |
| I3 `swe_swiss` | 10,254 | 0.19B | 0.05B | 13.5K / 39.7K | 258 | |
| AgentTrove, all | 1,696,847 | 20.01B | 7.64B | 8.7K / 26.1K | 5,261 | 128,435 null rows; 1,235,208 unique keys |
| **AgentTrove, behavioural cut** (first per key, complete, <= 64K) | **1,104,046** | **13.94B** | **5.39B** | | | 21% thinking |
| nebius SWE-rebench-OpenHands, all | 67,074 | 4.31B | 1.07B | **61.0K** / 89.6K | **27,249** | 6,306 instances |
| **nebius, resolved + submit + <= 64K** | **22,517** | **1.14B** | **0.27B** | | | 75% of tokens are tool output |

Findings:
- **INTELLECT-3 total: 79.9B tokens.** Most of it is repeated sampling: math has about 15 responses per prompt, code
  about 60. Cap responses per prompt; the full code split is 34K problems.
- **AgentTrove's kept 13.9B is dominated by two teachers:** gpt-5-nano 7.15B and GLM-4.6/GLM 8.5B across
  endpoints. Largest sources: freelancer 3.79B, exp_tas 2.73B, stack_exchange 1.51B, swesmith 1.18B, neulab 0.76B,
  r2egym 0.57B. Balance by teacher and source.
- **nebius trajectories are long:** the median is 61K tokens, and 41% of rows exceed 64K. At a 64K cap, 22.5K of
  32K resolved runs remain (1.14B tokens, only 0.27B trained). A 128K SFT stage would keep more.

SWE overlap (`swe_overlap.json`; matching on repo + PR, and on 20 or more shared non-template 13-grams of issue text):

| Query | vs Scale-SWE (RL) | vs SWE-bench Verified (eval) |
| --- | ---: | ---: |
| Scale-SWE, 17,202 tasks | - | **0** (3 tasks share a Verified repo) |
| nebius, 6,306 instances | **291** (286 by id) | **10** (by text) |
| AgentTrove SWE tasks (swesmith / r2egym / SWEGym / Inferred Bugs), 57,835 | 3 | **61** (by text) |

Scale-SWE-Verified is clean against SWE-bench Verified. Drop from SFT the 291 + 10 nebius instances and the 3 + 61
AgentTrove tasks.

Proposed 40B mix (seen tokens), with caps from the census:

| Bucket | Selection | Tokens |
| --- | --- | ---: |
| Math | <= 4 responses per prompt (about 680K rows) | ~11.5B |
| Code | <= 12 responses per prompt (about 410K rows) | ~6.3B |
| Science | sample of 1.6M unique | ~5.5B |
| Terminus-2 agentic | AgentTrove cut, teacher/source-balanced, minus overlaps | ~8B |
| Native tools | nebius cut minus overlaps (1.1B) + `openreasoning_tool` (1.6B) + `toucan_tool` (0.36B) | ~3.0B |
| Chat / IF, thinking | `am_chat` + `am_if`, all | ~2.4B |
| Chat / IF, thinking off | Nemotron IF-Chat v3 empty-think, or stripped `am_*` | ~1.5B |
| SWE patch-writing | `swe_swiss` | ~0.2B |
| **Total** | | **~38.4B** |

Trained tokens are lower than seen tokens: about 35% of agent tokens are trained, versus about 97% for reasoning.

## Step-5 dedup and decontamination flags (2026-10-23, JURECA jobs 15733802 / 15733817 / 15733903)

Script: `rl/scripts/data/sft_v2/decontam_sft_v2.{py,sbatch}`. It runs in three steps: `refs` and `flags` run in
parallel, then `finalize`. Output is in `/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v2/decontam/`:
- `flags/<source>.parquet`, row-aligned with `census/<source>.parquet`;
- `summary.json`.

Rows are flagged, not dropped. The mix step filters on `decontam_drop` and decides on the prompt-duplicate flags.

The checks:
- **Benchmarks:** 13-gram matches against `index_prelude_v1`, with hashing as in `decontam_conversations.py`. The
  index covers every benchmark of the Prelude eval suite (evalchemy, few-shot, SWE-bench Verified, TB2.1, TBLite).
  - Prompt-side and answer-side matches are counted separately.
  - An n-gram matched by more than 20 *distinct prompts* is treated as boilerplate. The count is per prompt, not
    per row, because INTELLECT-3 repeats prompts 15-60 times.
- **RL pools and agentic evals:** the AgentTrove `instruction` (or the nebius first user turn) is matched by
  13-grams against TMax-1k (1,000), RTS-1k (984), CalibForge (5,317), TBLite (100) and TB2.1 (89) instructions. A
  match needs at least 20 shared 13-grams, or at least half of a short target's 13-grams.
- **SWE:** the step-1 `swe_overlap` matches.
- **Prompt duplicates:** the exact first user message (lowercased, whitespace collapsed) is checked against SFT v1
  (0.80M distinct prompts), Nemotron-Terminal-Corpus (366K), the other v2 sources, and all of Cascade-2 (6.07M
  distinct prompts over 13.5M rows). The whole Cascade-2 pool is used, a superset of what mid-training saw.

`decontam_drop` = benchmark, RL-pool, agentic-eval or SWE overlap:

| Source | Rows | Drop rows | Drop tokens | Main reason | Prompt in Cascade-2 / SFT v1 |
| --- | ---: | ---: | ---: | --- | --- |
| nebius | 67,074 | 3,761 | 241M | SWE overlap 3,126; benchmark 674 | **59,990** / 0 |
| AgentTrove | 1,696,847 | 3,435 | 62M | **TBLite tasks 2,012**; benchmark 3,363 | 128,685 / 0 |
| `openreasoning_math` | 2,044,407 | 5,878 | 94M | benchmark (2,917 answer-side only) | 0 / 0 |
| `openreasoning_code` | 1,896,395 | 3,101 | 40M | benchmark (2,235 answer-side only) | 0 / 0 |
| `openreasoning_science` | 1,600,812 | 312 | 3.1M | benchmark | 0 / 0 |
| `am_chat` | 952,046 | 1,344 | 7.4M | benchmark | 69,675 / 440 |
| `am_if` | 54,720 | 104 | 0.2M | benchmark | 16,220 / 3,529 |
| `swe_swiss` | 10,254 | 63 | 1.7M | benchmark (SWE-bench issue text) | 317 / 0 |
| `openreasoning_tool` | 310,051 | 16 | 0.1M | benchmark | 18 / 1 |
| `toucan_tool` | 115,724 | 5 | 0.03M | benchmark | 2 / 0 |

In total, 0.48B of 104B tokens are dropped.

Findings:
- **AgentTrove contains 11 TBLite eval tasks verbatim, in 2,012 rows** (`code_contests` 1,387, `taskmaster` 625).
  - The tasks keep their TBLite names: `vimscript-vim-quine`, `broken-python`, `jq-data-processing`,
    `schedule-vacation`, `amuse-install`, `hydra-debug-slurm-mode`, `acl-permissions-inheritance`, `log-summary`,
    `jsonl-aggregator`, `mlflow-register`, `pandas-etl`.
  - A task-name check finds the same 2,012 rows.
  - Earlier models trained on AgentTrove have probably seen these tasks, so their TBLite scores need a re-check.
  - No AgentTrove row matches TMax, RTS or CalibForge.
- **89% of nebius rows have a first user turn that is also in Cascade-2.** Cascade-2's SWE subset appears to
  include the same SWE-rebench OpenHands runs.
  - Mid-training selected only about 6.2K of Cascade-2's 440K SWE rows (16K-128K selections), so little of this
    was seen.
  - The duplicates are flagged, not dropped.
- **INTELLECT-3 math/code has no exact prompt overlap with SFT v1 or with Cascade-2 math.** The check is exact, so
  reworded prompts or a different instruction prefix would be missed. A near-duplicate (13-gram) pass is not done
  yet.
- **IFEval constraint phrasing.** The boilerplate cut matters most for `am_if`: 45% of its prompts share an IFEval
  13-gram, from only 226 distinct n-grams. After the cut, 104 rows remain, and most of those still look like
  constraint templates. Per-row hit counts are kept, so the mix can choose a threshold above one hit.

## Mix build (2026-10-23, JURECA job 15734010)

Scripts: `rl/scripts/data/sft_v2/materialize_prelude_sft_v2.py` (`select`, `normalize`) and `build_sft_v2.sbatch`
(select + normalize, then render with `prerender_levanter.py --enable-thinking unset --max-tokens 65536
--skip-unrenderable`). Output in `/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v2/{selection,normalized,rendered}`.
Rows with `decontam_drop` are excluded.

The approved defaults (user, 2026-10-23):
- INTELLECT-3 replaces the SFT v1 reasoning sources.
- Native-tool data is nebius + `openreasoning_tool` + toucan; AgentTrove is not converted.
- Thinking-off data is the SFT v1 IF/chat rows with reasoning stripped. Their assistant turns after the last user
  get the empty think prefix `<think>\n\n</think>\n\n` (Qwen3 hybrid convention), the same block the chat template
  prefills for `enable_thinking=false`.

Selection rules:
- INTELLECT-3 math keeps at most 4 traces per prompt; code at most 12; science a 53% sample.
- AgentTrove keeps the longest episode per (trial, run), with the behavioural cut. Each (source, teacher) is capped
  at 284M tokens by water-filling. Reasoning is kept only after the last user turn (SFT v1 rule). Turns are trained
  only if they are valid Terminus JSON or a handoff reply.
- nebius: 842 rows over 64K once rendered are skipped.

| Mix source | Rows | Tokens |
| --- | ---: | ---: |
| `i3_math` | 634,354 | 11.97B |
| `i3_code` | 406,380 | 6.91B |
| `i3_science` | 847,922 | 5.48B |
| `i3_am_chat` + `i3_am_if` | 1,005,318 | 2.37B |
| `i3_tool` + `i3_toucan` | 422,438 | 1.59B |
| `i3_swe_swiss` | 9,934 | 0.17B |
| AgentTrove | 595,509 | 7.29B (2.48B trained) |
| nebius | 20,256 | 1.04B (0.26B trained) |
| `v1_if_chat_nothink` + `v1_if_instruction_following_nothink` | 398,253 | 0.89B |
| **Total** | **4,340,196** | **37.69B (30.09B trained)** |

Per-source rows are normalized counts; the total is after rendering, which rejects 168 more.

- Thinking-off IF/chat is 0.89B, below the 1.5B target, because stripping reasoning shortens the rows. Raising
  `NOTHINK_FRACTION` to 1.0 would give about 1.2B.
- 16.2% of the tokens (6.15B) are in rows over 32K: math 19.5%, code 17.9%, AgentTrove 16%, nebius 98.6%. A
  32K-only SFT is ruled out; the run uses 64K.

## Smoke SFT from stage 3 (2026-10-23/24)

The smoke pack is a seeded 2.03B-token subset of the rendered mix (`pack_prelude_stage_megatron.py --inputs
--budget-tokens 2e9 --max-docs 128`, JURECA job 15734089):
- 31,111 rows, 1.60B trained tokens, fill 99.6%;
- `megatron_packed/smoke2b/65536`, copied to MN5 with a verified sha256.

Training run:
- MN5 job 47871773: stage-3 base (`iter_0004419`, theta 9M), 64K, global batch 16 (1M tokens), TP4 with full
  recompute on 8 nodes, LR 1e-5 cosine to 1e-6.
- 1,944 steps (one pass) in 6h36m at about 12.0 s/step (about 290 TFLOP/s/GPU), with no errors.
- Loss went from 0.52 to about 0.44 by step 100, then stayed flat. Grad norm was about 0.17.

Export: MN5 job 47894647; HF loss 0.96176 vs Megatron 0.96166, max per-sequence difference 0.00079. The model is
at `/p/scratch/ccstdl/dc-agent-shared/checkpoints/prelude-9b-sft-v2-smoke2b-stage3-step1944` on Jupiter (16 files
sha256-verified), inventory id `prelude-9b-sft-v2-smoke2b-stage3-step1944`.

Evals (records `rl/evals/runs/20261024-prelude-sft-v2-smoke-s1944-*`; evalchemy with a 32K budget, agentic with
terminus-2, thinking on, 3x timeouts). Scores are thinking on / thinking off:

| Benchmark | Stage 3 | SFT v2 smoke |
| --- | ---: | ---: |
| AIME24 | 30.0 / 20.7 | **36.0 / 28.0** |
| AIME25 | 21.3 / 20.3 | **29.0 / 26.3** |
| MATH500 | 76.2 / 65.2 | **81.2 / 70.0** |
| GPQA-Diamond | 33.5 / 25.4 | **44.1 / 42.8** |
| IFEval (prompt, loose) | 63.6 / 66.4 | **69.4 / 68.4** |
| HumanEval+ | **68.3 / 62.8** | 45.1 / 61.6 |
| MultiPL-E | pending / 33.2 | pending / **45.4** |
| SWE-bench random 100 | 10 | **12** |
| TBLite 100 | 6 | 6 |
| TB2.1 | - | 3.4 |

On SWE-bench the agent timed out in 14 trials (82 of 98 for stage 3) and marked the task complete in 92. On
TBLite it still timed out in 40.

### HumanEval+ thinking-on drop: reasoning that never ends

Source: `evalchemy_15735853.log`. HumanEval+ in evalchemy decodes greedily (`do_sample: False` in
`HumanEvalPlus/eval_instruct.py`), while the math tasks sample at T=0.7.
- In thinking-on mode, 77 of 164 outputs fail code extraction. None of the 77 ever closes `</think>`:
  - 61 are repetition loops (zlib ratio of the last 4,000 chars below 0.15). For example, HumanEval/0 rereads the
    prompt's example list for 73K characters.
  - 16 are long non-repeating reasoning that hits the 32K budget.
- Of the 87 outputs that finish, 74 pass (85%). The model's code is fine; it fails to stop.
- In thinking-off mode only 7 outputs fail extraction.

### Loop scan of the training mix (JURECA job 15736625)

Script: `rl/scripts/data/sft_v2/loop_scan_sft_v2.py` over `normalized/`. It checks every trained reasoning turn
for:
- a repetition loop: the last 4,000 chars compress to below 0.15 of their size;
- a garbage tail: CJK junk such as `极` in the last 50 chars of an otherwise non-CJK trace.

| Source | Reasoning turns | Loops | Garbage tails | Median / p99 chars |
| --- | ---: | ---: | ---: | ---: |
| `i3_code` | 406,380 | 191 | 209 | 45.6K / 121K |
| `i3_math` | 634,354 | 25 | 183 | 36.3K / 91K |
| `i3_science` | 847,922 | 3 | 55 | 15.5K / 78K |
| `i3_am_chat` | 950,702 | 6 | 20 | 2.2K / 39K |
| AgentTrove | 183,819 | 5 | 23 | 0.9K / 3.5K |
| others (`am_if`, `swe_swiss`, tool) | 754,556 | 3 | 5 | |

- 570 rows are flagged in total (about 0.01%). Every reasoning turn in the mix is followed by an answer.
- The training data does not teach looping. The likely cause is the reasoning length: the code median is about
  12K tokens. A 2B-token smoke learns that style before it learns to reason well at that length, and greedy
  decoding then makes the loops much worse.
- The flagged rows are in `loop_scan/full/flagged_turns.jsonl`, with uuid plus part and row.
  `pack_prelude_stage_megatron.py --exclude-rows` drops them at pack time. It was tested on
  `i3_math/part-133-000`: 7 of 7 dropped, uuids match.

Plan:
- The full SFT has about 15x the data and is expected to reduce the loops.
- A HumanEval+ rerun with thinking on and sampling (T=0.6, top-p 0.95) separates the decoding effect.
- During the full run, track the share of HumanEval+ thinking-on outputs (greedy) that never close `</think>`; it
  is 77/164 on the smoke. If it does not fall, cap the longest math/code traces in the mix.

## Full SFT from stage 4 (2026-10-24/25)

Training run:
- MN5 job 47941695 (64 nodes, 19h31m) plus resume 48005759 (64 nodes, 34 min), `megatron_midtrain.sbatch`.
- Stage-4 base (`stage4-128k-yarn2`, iter 1001, YaRN x2 over 64K), full rendered mix packed at 64K
  (`megatron_packed/full_v2/65536`), global batch 64 x 64K (4M tokens), LR 2e-5 cosine, TP4, full recompute.
- 9,020 steps, one pass over 37.7B tokens. HF exports at iters 500 / 1000 / 2500 / 5000 / 9020 (jobs 48005770-73,
  48009719). Inventory: `rl/inventory/models/prelude-9b-sft-v2-full-stage4-step9020.yaml`.

Evals of the final step: records `rl/evals/runs/20261025-prelude-sft-v2-full-s9020-*`, script
`$S/eval_moves/prelude_20261025_sft_v2_full_s9020.sh`. Thinking on is sampled at T=0.6 / top-p 0.95; thinking off
and the extra HumanEval+ row are greedy. The full comparison, with Qwen3.5-9B-Base, Qwen3.5-9B (SFT + RL, reference
only) and Grug 09.21, is [`sft_v2_full_vs_qwen35.md`](../evals/prelude_midtrain/sft_v2_full_vs_qwen35.md).

| Benchmark | SFT smoke (2B) | SFT full (37.7B) | Qwen3.5-9B-Base |
| --- | ---: | ---: | ---: |
| AIME24 (think / nothink) | 36.0 / 28.0 | **54.3 / 50.3** | 80.3 / 73.3 |
| AIME25 | 29.0 / 26.3 | **43.3 / 37.0** | 66.3 / 66.7 |
| MATH500 | 81.2 / 70.0 | **84.6 / 78.2** | 90.8 / 89.0 |
| GPQA-Diamond | 44.1 / 42.8 | **51.2 / 45.6** | 73.6 / 56.7 |
| IFEval (prompt, loose) | 69.4 / 68.4 | **77.3 / 74.9** | 70.6 / 58.8 |
| HumanEval+ (think sampled / nothink) | - / 61.6 | **80.8 / 65.6** | 86.6 / 79.3 |
| HumanEval+, think greedy | 45.1 | **58.6** | 86.6 |
| MultiPL-E (think / nothink) | - / 45.4 | 46.9 / 46.2 | 63.3 / 62.3 |
| SWE-bench random 100 (3 runs) | 12 (1 run) | **17.3** (19, 14, 19) | 4 (1 run) |
| TB2.1 (3 runs) | 3.4 (1 run) | 3.0 (3, 4, 1) | - |
| TBLite 100 (3 runs) | 6 (1 run) | **7.7** (11, 7, 5) | 7 (1 run) |

Where both clusters ran a job, the full-SFT value is the mean of the two runs (they differ by at most 4 points).
Qwen-Base thinking-on runs are greedy.

- **Unclosed thinking falls but does not vanish.** Greedy thinking-on HumanEval+: 50 of 164 outputs have no code
  block (smoke 77 of 164); sampled: 4 of 164. The sampled score (80.8) is what the model's code is worth; greedy
  decoding still loops.
- **Thinking off does not improve with data.** MultiPL-E thinking off is 46.2 at the end and 46.8 at step 500.

### Data scaling (steps 500-5000)

Same evals per intermediate export (`$S/eval_moves/prelude_20261025_sft_v2_full_steps.sh`, JUWELS), all finished
2026-10-26; full table in [`sft_v2_full_vs_qwen35.md`](../evals/prelude_midtrain/sft_v2_full_vs_qwen35.md#data-scaling-sft-v2-full-intermediate-exports).

| Benchmark | s500 (2.1B) | s1000 (4.2B) | s2500 (10.5B) | s5000 (21.0B) | s9020 (37.7B) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AIME24 think | 36.3 | 41.0 | 46.0 | 48.0 | 54.3 |
| GPQA-D think | 44.6 | 46.8 | 45.3 | 45.1 | 51.2 |
| IFEval think | 71.0 | 70.0 | 73.4 | 75.8 | 77.3 |
| HumanEval+ think T0.6 / greedy | 68.3 / 42.1 | 70.1 / 47.6 | 72.0 / 43.9 | 78.0 / 51.8 | 80.8 / 58.6 |
| MultiPL-E think / nothink | 38.4 / 46.8 | 42.0 / 47.3 | 44.1 / 46.3 | 47.4 / 46.3 | 46.9 / 46.2 |
| SWE / TB2.1 / TBLite (3 runs) | 14.0 / 1.9 / 6.0 | 12.3 / 1.9 / 6.7 | 14.7 / 1.9 / 7.7 | 16.0 / 3.4 / 8.3 | 17.3 / 3.0 / 7.7 |

Thinking-on reasoning and code improve steadily with data (AIME24 +18, HumanEval+ +12.5, greedy +16.5), with no
plateau on AIME/GPQA; thinking-off code is flat from 2B tokens on. Agentic barely moves (+3 SWE, +1 TB2.1, +2 TBLite
from 2B to 37.7B tokens): the SFT v2 data does not teach agent behaviour.

### Agentic failure modes

Plot: [`rl/evals/plots/prelude_sft_failure_modes.png`](../evals/plots/prelude_sft_failure_modes.png)
(`rl/evals/plot_prelude_failure_modes.py`; Qwen3.5-9B, Grug SFT465, SFT smoke, SFT full; runs pooled).

- **SWE ends in wrong "done".** 59% of the full SFT's trials declare completion with a wrong patch (Qwen 35%), after
  a median of 215 turns (Qwen 54).
- **Format decays with trajectory length on TB.** 21-22% of agent turns are rejected format errors, growing from
  4% in the first 25 turns to 33% after turn 150; Qwen stays at 1-2% throughout. The full SFT decays faster than
  the smoke SFT on long TB runs.
- **Shell traps.** 18-20% of TB trials end stuck at the shell continuation prompt (an unclosed heredoc or quote
  swallows the following commands); Qwen 0-2%.
- **Short reasoning per turn.** Median hidden reasoning is 117 / 90 / 103 tokens per agent turn (SWE / TB2.1 /
  TBLite) vs Qwen's 185 / 276 / 276. Trajectories store only the text after `</think>`, so this is estimated as
  completion tokens minus visible-message tokens. A likely cause: the AgentTrove rows keep reasoning only after the
  last user turn (selection rules above), so the agentic data trains at most one reasoning block per trajectory.

## Recovery SFT on top of the full SFT (2026-10-25, MN5)

Question (user): does the Grug "recovery SFT" mix lift Prelude's agentic scores the way it lifted Grug 09.21
(Grug 09.21 -> v4-think s300: TBLite 12.5 -> 23.0; the same mix on Antidoom, SFT465: SWE 40.0, TBLite 26.7)?

- **Mix:** `grug-sft-mix-v4-40M-think` ([inventory](../inventory/datasets/grug-sft-mix-v4-40m-think.yaml), 0048):
  7,453 conversations, 55% AgentTrove agentic, 25% summarization handoff, 12% synthetic structural recovery, 8%
  Nemotron non-agentic; teacher reasoning on every agentic turn. No row mentions any of the 11 TBLite task names
  found in AgentTrove.
- **Render:** `prerender_levanter.py --tokenizer prelude_chat_tokenizer --enable-thinking unset --max-tokens 65536`:
  46.9M tokens, 14.5M trained, no row refused. Output and packs:
  `/p/data1/mmlaion/marianna/sft_mixes/prelude_grug_v4_recovery/`.
- **Pack:** two passes packed separately (seeds 20261026 / 20261027, 718 rows of 64K each, fill 99.7%) and
  concatenated: 1,436 rows, copied to MN5 `prelude_sft_v2/megatron_packed/grug_v4_recovery_2ep/65536` (sha256 verified).
- **Train:** from the s9020 HF export, global batch 8 x 64K, LR 1e-5 cosine, 179 steps (two passes, close to
  Grug's two-epoch SFT465), warmup 9 steps (5%, as Levanter for Grug), checkpoints at 90 and 179; 8 nodes.
  `RUN_NAME=sft-v2-full-s9020-grug-v4-recovery`.
  - First try (job 48037611, 4 nodes, GBS 4) ran out of memory at the first loss: with 4 data-parallel ranks the
    distributed optimizer state per GPU is too large. 8 nodes fit (as the smoke did).
  - `megatron_midtrain.py` had the warmup hard-coded at 100 steps; it now reads `WARMUP` (default 100, so earlier
    stages and resumes are unchanged; backups `*.bak-20261025-prewarmup` on MN5). The trainer has no epoch wrap,
    which is why the passes are packed explicitly.
  - Job 48039684 died at start on a bad node ("CUDA-capable device busy or unavailable", as01r1b03; excluded).
  - Job 48041737 trained 37 steps, then ran out of memory at step 38 from fragmentation (the TE cross-entropy
    allocates a 16 GiB fp32 logits buffer per 64K sequence). `megatron_midtrain.sbatch` now passes
    `PYTORCH_CUDA_ALLOC_CONF` into the container.
  - **Job 48044063 completed** all 179 steps with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (~5 s/step,
    loss 0.37 early -> 0.25-0.33).
- **Export:** MN5 jobs 48045032 / 48045033; HF vs mcore loss on the check set 1.00763 vs 1.00773 (iter 90) and
  1.01494 vs 1.01481 (iter 179), per sequence within 0.0008 (s9020 start: 2.01); YaRN config kept. Copied to JSC
  (sha256 verified): `prelude-9b-sft-v2-full-s9020-grug-v4-recovery-step{90,179}` (inventory registered).
- **Evals (2026-10-26):** `$S/eval_moves/prelude_20261026_sft_recovery.sh ACCOUNT "179 90"`, same 9 evalchemy jobs +
  SWE/TB2.1/TBLite reps 1-3 as s9020; tags `prelude-sft-rec-s{179,90}`; evalchemy on JUWELS, agentic twins on both.
- **Caveat:** earlier turns' reasoning stays in the training context (as for Grug), whereas at inference terminus-2
  sends only the visible messages.

## Open questions

- Per-source token counts (census) to firm up the split above.
- Which native-tool harness the agentic RL will use, which fixes the tool schema for the AgentTrove conversion.
- Whether to keep SFT v1 sources (Nemotron math/science/IF/SWE) alongside INTELLECT-3, or replace them.
- RLVR task pool (math/code verifiable) and its decontamination against the SFT prompts.

## Status

- [x] Pin INTELLECT-3-SFT, census per split (tokens, thinking, tool shape, lengths); licences still open
- [x] Normalizers: INTELLECT-3 (`prompt`/`completion`/`tools`), AgentTrove behavioural cut
- [x] Thinking-off IF/chat source chosen and rendered (SFT v1 IF/chat, empty think block; 0.89B)
- [ ] Native-tool harness chosen; AgentTrove Terminus-2 to native-tool convertor (if adopted)
- [x] Dedup vs SFT v1 / mid-training; decontam incl. RL task pools and agentic evals (flags; exact prompt keys)
- [ ] Near-duplicate prompt pass for INTELLECT-3 math/code vs SFT v1 / Cascade-2 (optional)
- [ ] Mix weights frozen under the token budget; rendered release + inventory YAML
- [x] Loop scan of the mix; flagged rows excluded at pack time (`--exclude-rows`)
- [ ] Smoke SFT from stage 3 + evals in both harnesses (terminus-2 done; native-tool harness pending)
- [x] HumanEval+ thinking-on rerun with sampling (T=0.6, top-p 0.95)
- [x] Full SFT from stage 4 (9,020 steps) + evals; unclosed-think rate tracked (greedy HumanEval+ no-code 50/164)
- [x] Data-scaling evals for steps 500 / 1000 / 2500 / 5000
- [x] Recovery SFT (Grug v4 mix) on top of s9020 (MN5 job 48044063), exported and copied
- [ ] Recovery SFT evals (steps 90 / 179, running)

## Related

- [0054](0054-build-prelude-9b-midtraining-data.md): mid-training and the stage-3 starting checkpoint.
- [Prelude SFT v1](../inventory/datasets/prelude-sft-v1.yaml) and its
  [build](../sft/prelude/build_sft_v1.sbatch).
- [0048](0048-experiment-sft-data-snowball-grug-push.md): AgentTrove assessment.
- [0048](0048-experiment-sft-data-snowball-grug-push.md) / [0036](0036-experiment-sft-grug-datakit-terminus.md): the Grug recovery-SFT mix (v4 think) reused above.
- 0051 / [0057](0057-experiment-antidoom-s465-rl-on-tmax-rts-and-calibf.md): TMax/RTS/CalibForge, kept for RL.
