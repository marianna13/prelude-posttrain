# Prelude 9B SFT v2 full (37.7B tokens) vs Qwen3.5-9B-Base vs Qwen3.5-9B (+ Grug 09.21, Nemotron 3.5 Lightning)

| Column | Model |
|---|---|
| Prelude SFT | `prelude-9b-sft-v2-full-stage4-step9020` (SFT v2 on the stage-4 128K mid-train, final step, ~37.7B tokens) |
| Qwen-Base | `Qwen/Qwen3.5-9B-Base` |
| Qwen (reference) | `Qwen/Qwen3.5-9B`, post-trained with agentic SFT **and RL**: an upper reference, not a like-for-like comparison |
| Grug 09.21 | `grug-67b-a2b-datakit-sft-262k-2026-09-21` (Marin general/datakit SFT, before our agentic SFT mixes), 67B-A2B MoE |
| Nemotron | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, hybrid Mamba MoE, post-trained (external reference, like Qwen) |

Scores in %. Evalchemy: 32,768-token budget, each model's own chat template. AIME is the mean of 10 samples, GPQA-D of 3;
IFEval is prompt-level loose accuracy; MultiPL-E is the mean pass@1 over 10 languages.

Decoding with thinking on: Prelude SFT, Qwen and Nemotron sample at T=0.6 / top-p 0.95; Qwen-Base and Grug use the task defaults
(greedy for code and IFEval). The "HumanEval+, greedy" row is greedy for every model. Where Prelude ran on both JURECA and
JUWELS the value is the mean of the two runs (they differ by at most 4 points).

| Benchmark | Prelude SFT | Qwen-Base | Qwen (SFT+RL ref.) | Grug 09.21 | Nemotron |
|---|---:|---:|---:|---:|---:|
| **Thinking on** | | | | | |
| AIME24 | 54.3 | **80.3** | 76.0 | 59.0 | 76.7 |
| AIME25 | 43.3 | 66.3 | 60.7 | 43.3 | **67.7** |
| MATH500 | 84.6 | 90.8 | 84.8 | 87.0 | **91.0** |
| GPQA-Diamond | 51.2 | 73.6 | **77.8** | 60.4 | 68.7 |
| IFEval | 77.3 | 70.6 | 80.6 | 32.6 | **93.8** |
| HumanEval+ | 80.8 | 86.6 | 89.0 | 69.5 | **90.2** |
| HumanEval+, greedy | 58.6 | 86.6 | 86.6 | 69.5 | **89.0** |
| MultiPL-E | 46.9 | 63.3 | 68.4 | 55.4 | **72.4** |
| **Thinking off** | | | | | |
| AIME24 | 50.3 | **73.3** | 72.7 | 10.7 | 65.7 |
| AIME25 | 37.0 | **66.7** | 57.7 | 6.0 | 45.7 |
| MATH500 | 78.2 | 89.0 | **91.2** | 64.8 | 86.2 |
| GPQA-Diamond | 45.6 | 56.7 | **78.6** | 49.3 | 64.5 |
| IFEval | 74.9 | 58.8 | **88.8** | 63.6 | 78.0 |
| HumanEval+ | 65.6 | 79.3 | 82.3 | 40.2 | **87.2** |
| MultiPL-E | 46.2 | 62.3 | 61.3 | **67.0** | 65.8 |
| **Agentic (terminus-2, thinking on, 3x timeouts)** | | | | | |
| SWE-bench Verified, random 100 | 17.3 | 4 | **53.7** | 19.2 | 48.3 |
| Terminal-Bench 2.1 | 3.0 | - | 23.2 | 5.9 | **24.0** |
| TBLite 100 | 7.7 | 7 | 40.0 | 12.5 | **47.0** |

Agentic: mean of 3 runs (Qwen-Base: 1 run, no TB2.1; Grug 09.21: 4 runs; Nemotron at TP4, server-default T1.0 like the others), isolated harbor for Prelude and Qwen-Base.
Grug 09.21 agentic runs are from 2026-09-24..10-04, before harbor isolation; SWE is the fixed-observations run, TBLite
predates the 50k output-cap fix (likely biased low). For scale, Grug's agentic SFT465 scores 40.0 / 13.1 / 26.7.
Qwen's 38.6 on C# with thinking off lowers its MultiPL-E mean (61.3); a C#-only rerun reproduced it.

Prelude SFT vs Qwen-Base, the fair comparison of the two bases (neither has agentic RL): Prelude leads on IFEval
(+6.7 thinking on, +16.1 off) and SWE (17 vs 4) and is level on TBLite. It trails by 20-25 points on AIME and
thinking-on GPQA, by 16 on MultiPL-E, by 6 on MATH500 and sampled HumanEval+, and by 28 on greedy HumanEval+, where
its reasoning often never closes (50 of 164 greedy outputs have no code block vs 4 of 164 sampled).


Nemotron 3.5 Lightning (30B-A3B, vLLM 0.26.0, the card recommends 0.27.1 and T1.0; evalchemy thinking-on used T0.6 for
comparability) is level with Qwen3.5-9B on agentic (SWE 48.3 vs 53.7, TB2.1 24.0 vs 23.2, TBLite 47.0 vs 40.0) and ahead on
IFEval, code and AIME25 with thinking on; it trails on GPQA-D and on thinking-off math.

## Data scaling: SFT v2 full intermediate exports

Same evals at Megatron iters 500 / 1000 / 2500 / 5000 / 9020 (GBS 64 x 64K, ~4.2M tokens per step). Agentic: mean of 3 runs.

| Benchmark | s500 (2.1B) | s1000 (4.2B) | s2500 (10.5B) | s5000 (21.0B) | s9020 (37.7B) |
|---|---:|---:|---:|---:|---:|
| AIME24, think | 36.3 | 41.0 | 46.0 | 48.0 | **54.3** |
| AIME25, think | 32.3 | 31.0 | 34.3 | 42.3 | **43.3** |
| MATH500, think | 82.4 | 81.2 | 82.8 | **85.4** | 84.6 |
| GPQA-Diamond, think | 44.6 | 46.8 | 45.3 | 45.1 | **51.2** |
| IFEval, think | 71.0 | 70.0 | 73.4 | 75.8 | **77.3** |
| HumanEval+, think T0.6 | 68.3 | 70.1 | 72.0 | 78.0 | **80.8** |
| HumanEval+, think greedy | 42.1 | 47.6 | 43.9 | 51.8 | **58.6** |
| MultiPL-E, think | 38.4 | 42.0 | 44.1 | **47.4** | 46.9 |
| AIME24, nothink | 35.7 | 28.3 | 44.0 | 47.7 | **50.3** |
| MATH500, nothink | 74.4 | 79.8 | **80.8** | **81.2** | 78.2 |
| IFEval, nothink | 71.4 | 70.8 | 74.4 | **75.4** | 74.9 |
| HumanEval+, nothink | 64.0 | 65.2 | 65.9 | 64.6 | **65.6** |
| MultiPL-E, nothink | 46.8 | **47.3** | 46.3 | 46.3 | 46.2 |
| SWE-bench Verified, random 100 | 14.0 | 12.3 | 14.7 | 16.0 | **17.3** |
| Terminal-Bench 2.1 | 1.9 | 1.9 | 1.9 | **3.4** | 3.0 |
| TBLite 100 | 6.0 | 6.7 | 7.7 | **8.3** | 7.7 |

More SFT data keeps improving thinking-on reasoning and code (AIME24 +18, HumanEval+ +12.5, greedy HumanEval+ +16.5,
MultiPL-E +8.5) with no plateau yet on AIME/GPQA; thinking-off code is flat from step 500. Agentic barely moves
(SWE +3, TBLite +2, TB2.1 within noise), so the SFT v2 data does not teach agent behaviour; the recovery SFT
(`prelude-9b-sft-v2-full-s9020-grug-v4-recovery-step{90,179}`, evals running) tests whether agentic data on top does.
s5000 TBLite rep3 scored 80 of 100 trials.

### Cooldown from s2500

`prelude-9b-sft-v2-cooldown-s2500-step3000`: LR cooldown from the s2500 checkpoint, 500 steps (~2.1B tokens), final
step 3000. Question: is the s2500 → s9020 gain from more data, or would annealing get it? Same evals.

| Benchmark | s2500 | Cooldown s3000 | s9020 |
|---|---:|---:|---:|
| AIME24, think | 46.0 | 46.7 | **54.3** |
| AIME25, think | 34.3 | 32.0 | **43.3** |
| MATH500, think | 82.8 | 83.8 | **84.6** |
| GPQA-Diamond, think | 45.3 | 46.6 | **51.2** |
| IFEval, think | 73.4 | 76.4 | **77.3** |
| HumanEval+, think T0.6 | 72.0 | 73.8 | **80.8** |
| HumanEval+, think greedy | 43.9 | 50.6 | **58.6** |
| MultiPL-E, think | 44.1 | 45.5 | **46.9** |
| AIME24, nothink | 44.0 | 46.3 | **50.3** |
| MATH500, nothink | 80.8 | **81.2** | 78.2 |
| IFEval, nothink | 74.4 | **75.6** | 74.9 |
| HumanEval+, nothink | **65.9** | 62.8 | 65.6 |
| MultiPL-E, nothink | 46.3 | **46.9** | 46.2 |
| SWE-bench Verified, random 100 | 14.7 | 16.7 (19/16/15) | **17.3** |
| Terminal-Bench 2.1 | 1.9 | 2.2 (1/3/2) | **3.0** |
| TBLite 100 | **7.7** | 6.3 (5/6/8) | **7.7** |

The cooldown does not deliver the thinking-on math gain: AIME24 +0.7 and AIME25 -2.3 vs s2500, still 8-11 points below
s9020. It helps instruction following and code formatting a bit (IFEval +3, greedy HumanEval+ +6.7, GPQA +1.3), but
agentic is unchanged within noise. The s9020 reasoning gain comes from more SFT data, not from annealing.

Sources: `../runs/20261025-prelude-sft-v2-full-s9020-*`, `../runs/20261025-qwen35-9b-evalchemy-*-t06p95`, `../runs/20261024-qwen35-9b-*`,
`../runs/20261023-qwen35-9b-base-*`, `../runs/20260926-grug0921-evalchemy-*`, `../runs/20260925-grug0921-evalchemy-nothink-32k-*`,
`../runs/20260925-grug0921-swe100-think-3x-fixobs-*`, `../runs/20261004-grug0921-tb21-think-3x*`,
`../runs/20261002-grug0921-tblite100-think-3x*`, `../runs/20261026-nemotron35-lightning-*`, data scaling
`../runs/20261026-prelude-sft-v2-full-s{500,1000,2500,5000}-*` (submitted with `eval_moves/prelude_20261025_sft_v2_full_steps.sh`).
