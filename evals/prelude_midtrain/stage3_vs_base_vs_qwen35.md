# Prelude 9B: SFT base vs mid-train stage 3 (64K) vs SFT v2 smoke vs Qwen3.5-9B-Base (+ Birger 256K)

Updated 2026-10-24. Only the SFT v2 smoke has chat/agentic SFT (it includes AgentTrove terminus-2 traces).

| column | inventory id |
|---|---|
| Base | `prelude-9b-anneal300b-ml1-dominant2-iter0989075` (SFT base chosen 2026-10-16; mid-train start, 4K context) |
| 64K | `prelude-9b-midtrain-v1-stage3-64k-theta9m-step4419` (+3.4B trained tokens at 16K/32K/64K, RoPE theta 9M) |
| SFT smoke | `prelude-9b-sft-v2-smoke2b-stage3-step1944`: 64K + 2.03B tokens (1.60B trained) of the SFT v2 mix, one pass, LR 1e-5 (issue 0058) |
| Qwen | `qwen3.5-9b-base` (`Qwen/Qwen3.5-9B-Base`, 262K context) |
| 256K (Birger) | `prelude-9b-anneal300b-256k-theta64m` (`birgermoell/oellm-9b-256k-theta64m-prelude-anneal300b` @ `4aad6daf`): plain anneal300b (not dominant2) + 8.0B tokens of long-context CPT, 16K→256K, RoPE theta 500K→64M |

## Base few-shot (4K window, greedy, plain-text prompts)

Suite: `rl/scripts/eval_jureca/base_fewshot/`. Shots in brackets.

| Benchmark | Base | 64K | Qwen | 256K (Birger) |
|---|---:|---:|---:|---:|
| MATH (4) | 47.8 | 47.6 | **56.2** | 13.2 |
| GSM8K (8) | 80.5 | 79.2 | **84.5** | 40.8 |
| MMLU-Pro (5) | 42.2 | 43.5 | **58.1** | 27.1 |
| HumanEval+ (0) | **54.3** | 48.8 | 50.0 | 9.8 |
| MBPP+ (0), raw | 38.1 | 32.8 | **64.0** | 0.3 |
| MBPP+ (0), cut to parseable prefix | 39.9 | 53.4 | **64.0** | - |
| MultiPL-E (0) | 24.7 | 33.4 | **54.1** | 3.8 |

The 256K model starts from plain anneal300b, which scores far below dominant2 itself (MATH 15.3, GSM8K 54.2, MMLU-Pro
31.1, HumanEval+ 17.1, MBPP+ 20.9, MultiPL-E 13.9; `../analysis/prelude-sft-base-selection/`), so compare it with that
parent: the extension costs every benchmark, code most (MBPP+ −20.6, MultiPL-E −10.1; nested indentation collapses
into SyntaxErrors). Our 64K stage started from dominant2 and keeps its scores. The 256K model has no ChatML gates or agentic
runs; its RULER and evalchemy runs are in progress.

64K vs Base, paired non-inferiority (20,000 bootstrap replicates, Bonferroni over six metrics, margin 2 points,
1.5 for MMLU-Pro): MATH, MMLU-Pro (+1.3) and MultiPL-E (+8.8) pass; GSM8K (−1.3), HumanEval+ (−5.5) and MBPP+
(−5.3) are inconclusive; none fails.

The raw MBPP+ scores of both Prelude models suffer from run-on generation: the completion defines the function and
then keeps writing until the 1,024-token limit cuts it mid-line (MBPP+ stops lack `\ndef`). Cutting each completion
back to its longest parseable prefix (`regrade_mbpp_sanitized.py`, diagnostic only) cuts 116 / 136 / 8 completions
for Base / 64K / Qwen. Qwen's score does not change, so its MBPP+ lead is real.

## Chat (ChatML, frozen Prelude-tokenized inputs; thinking off / on)

| Benchmark | Base, 4K | 64K, 4K | 64K, 16K | Qwen |
|---|---:|---:|---:|---:|
| GSM8K | 16 / 70 | 88 / 85 | 87 / 89 | - |
| MATH500 | 4 / 32 | 50 / 37 | 67 / 69 | - |
| HumanEval+ | 0 / 71 | 67 / 57 | 67 / 60 | - |

4K = 4,096 window with a 1,024-token budget; 16K = 16,384 window with a 14,336-token budget. Base has no chat
template of its own; it was served with the Prelude ChatML template. These inputs are token IDs from the Prelude
tokenizer, so they cannot be run on Qwen; the evalchemy runs below are the like-for-like chat comparison.

## Long context: RULER mean (13 tasks)

| Context | Base | 64K | 64K + YaRN x2 (inference) | Stage 4 (128K, YaRN x2 trained) | 256K (Birger) | Qwen |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 80.9 | 84.2 | 83.5 | 85.0 | **86.4** | - |
| 16K | - | **80.2** | - | - | 77.3 | - |
| 32K | - | **74.5** | - | - | 74.2 | - |
| 64K | - | 56.6 | 61.4 | 65.8 | **67.0** | - |
| 128K | - | - | 43.7 | 50.4 | **61.0** | - |
| 256K | - | - | - | - | running | - |

Plain prompts. Stage-4 and YaRN-inference cells are from issue 0054 (MN5); the others are JSC runs of the same gate
inputs (`gates_20261018/inputs_ruler_*`). At 64K the 256K model leads on QA (hotpot 41 vs 9, SQuAD 55 vs 30 for
stage 3), multivalue/multiquery NIAH and multikey-3 (33 vs 8); stage 3 is ahead on FWE (74 vs 63), CWE (12 vs 2) and
multikey-2 (73 vs 65).

Base supports only 4K positions. Same tokenizer limitation as the chat gates for Qwen.

## Chat: evalchemy (thinking on / off)

64K, SFT smoke and Qwen: 32,768-token budget, 40,960 window, each model's own chat template. Base: Prelude ChatML template,
3,072-token budget, 6,144 window (4K-trained; positions raised with `--hf-overrides` for one 2.6K-token GPQA prompt).

| Benchmark | Base | 64K | SFT smoke | Qwen |
|---|---:|---:|---:|---:|
| AIME24 | 0 / 0 | 30.0 / 20.7 | 36.0 / 28.0 | **80.3 / 73.3** |
| AIME25 | 0 / 0 | 21.3 / 20.3 | 29.0 / 26.3 | **66.3 / 66.7** |
| MATH500 | 0 / 0 | 76.2 / 65.2 | 81.2 / 70.0 | **90.8 / 89.0** |
| GPQA-Diamond | 27.8 / 13.0 | 33.5 / 25.4 | 44.1 / 42.8 | **73.6 / 56.7** |
| IFEval (prompt, loose) | 27.2 / 11.6 | 63.6 / 66.4 | 69.4 / **68.4** | **70.6** / 58.8 |
| HumanEval+ | 6.7 / 0.0 | 68.3 / 62.8 | 45.1 / 61.6 | **86.6 / 79.3** |
| MultiPL-E | 24.5 / 10.1 | pending / 33.2 | running / 45.4 | **63.3 / 62.3** |

Base has no chat SFT: on chat prompts it loops on template tokens or echoes the problem and writes new ones, so
it never gives a boxed answer (MATH/AIME 0) and GPQA sits near the 25% chance level. Its column measures chat-format
following, not ability; the few-shot table above is the fair base comparison. Duplicate submissions of the 64K and
Qwen math jobs agree within 4 points (records list both). 64K think code: the full group hit the 12 h limit in
MultiPL-E; the split rerun finished IFEval/HumanEval+, MultiPL-E still running (`eval_moves/juwels_20261023_s3_think_code_split.sh`).

SFT smoke vs 64K: thinking on/off gains on every math and science benchmark (GPQA +10.6 / +17.4, AIME +6 to +8),
IFEval and MultiPL-E thinking-off (+12.2). HumanEval+ thinking-on drops 23.2 points: HumanEval+ decodes greedily,
and 77 of 164 outputs never close `</think>` (61 repetition loops, 16 runs to the 32K budget). 74 of the 87 that
close pass (85%). The training mix itself has almost no loops (570 flagged rows of 4.3M; issue 0058). A sampled
rerun (T=0.6, top-p 0.95) and the thinking-on MultiPL-E split (3 shards) are running.

## Agentic (terminus-2, thinking on, 3x timeouts, 1 run)

| Benchmark | 64K | SFT smoke | Qwen |
|---|---:|---:|---:|
| SWE-bench random 100 | 10 | **12** | 4 |
| TBLite 100 | 6 | 6 | **7** |
| Terminal-Bench 2.1 | - | 3.4 | - |

Isolated harbor with the `</think>` parser fix. 64K and Qwen have no agentic SFT; 82 of the 64K model's 98 scored SWE
trials ended in AgentTimeoutError (it hits the output limit and never declares done). The SFT smoke times out in
14 of 100 SWE trials (marks complete in 92) but still in 40 of 100 TBLite trials; TB2.1 is 3 of 89. For scale: Antidoom s12
(agentic RL, 67B-A2B) scores 23.3 / 21.7.

## Sources

Records in `../runs/`: `20261016-prelude-anneal300b-ml1-dominant2-base-fewshot`, `20261018-prelude-midtrain-initialized-*`,
`20261022-prelude-midtrain-s3-64k-t9m-s4419-*`, `20261023-*` (evalchemy and agentic for all three), `20261024-prelude-sft-v2-smoke-s1944-*` (SFT smoke), `20261015-qwen35-9b-base-base-fewshot`, `20261015-prelude-256k-base-fewshot{,-code}`. Non-inferiority and MBPP+
re-grade outputs (`noninferiority_s4419.json`, `mbpp_sanitized_regrade_{s4419,qwen35_9b_base}.json`) are under
`/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/gates_20261018/`. Stage-by-stage details: `README.md`.
