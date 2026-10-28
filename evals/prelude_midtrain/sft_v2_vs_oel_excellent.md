# Prelude 9B SFTs: our SFT v2 vs OpenEuroLLM Excellent-Instructions (+ Qwen3.5-9B-Base)

| Column | Model | How it was made |
|---|---|---|
| OEL Excellent | `openeurollm/OpenEuroLLM-Prelude-SFT-Excellent-Instructions` @6551e54c | Birger's `oellm-9b-128k-theta32m-prelude-anneal300b` (128K) + `OpenEuroLLM-Excellent-SFT`, 2 epochs (step 55,836), 32K sequences, assistant-only loss ([Taskboard #416](https://github.com/OpenEuroLLM/Taskboard/issues/416)) |
| Our SFT v2 | `prelude-9b-sft-v2-full-stage4-step9020` ([marianna13/prelude-9b-sft-v2](https://huggingface.co/marianna13/prelude-9b-sft-v2)) | Prelude anneal300b (`iter_0989075`) → our 4-stage long-context mid-train (16K → 128K, YaRN x2) → SFT v2 full, ~37.7B tokens, 64K sequences |
| Qwen3.5-9B-Base | `Qwen/Qwen3.5-9B-Base` | external reference (a base model, no post-training) |

Scores in %. Evalchemy (mlfoundations/evalchemy 6ed6741), 32,768-token generation budget, each model's own chat template,
vLLM 0.26 on 4x A100-40GB. AIME is the mean of 10 samples, GPQA-Diamond of 3. IFEval is prompt-level loose accuracy.
MultiPL-E is the mean pass@1 over 10 languages (cpp, cs, java, js, lua, php, r, rs, sh, ts).

Decoding with thinking on: both Prelude SFTs sample at T=0.6 / top-p 0.95; Qwen3.5-9B-Base uses the task defaults
(greedy for code and IFEval). The "HumanEval+, greedy" row is greedy for every model. Thinking off is greedy.

| Benchmark | OEL Excellent | Our SFT v2 | Qwen3.5-9B-Base |
|---|---:|---:|---:|
| **Thinking on** | | | |
| AIME24 | 32.0 | 54.3 | **80.3** |
| AIME25 | 9.3 | 43.3 | **66.3** |
| MATH500 | 47.0 | 84.6 | **90.8** |
| GPQA-Diamond | 25.9 | 51.2 | **73.6** |
| IFEval | 67.6 | **77.3** | 70.6 |
| HumanEval+ | 53.0 | 80.8 | **86.6** |
| HumanEval+, greedy | 60.4 | 58.6 | **86.6** |
| MultiPL-E | 27.8 | 46.9 | **63.3** |
| **Thinking off** | | | |
| AIME24 | 2.7 | 50.3 | **73.3** |
| AIME25 | 3.7 | 37.0 | **66.7** |
| MATH500 | 44.0 | 78.2 | **89.0** |
| GPQA-Diamond | 25.4 | 45.6 | **56.7** |
| IFEval | 68.6 | **74.9** | 58.8 |
| HumanEval+ | 59.8 | 65.6 | **79.3** |
| MultiPL-E | 33.5 | 46.2 | **62.3** |

Our SFT v2 is ahead of OEL Excellent on every benchmark except greedy HumanEval+ (58.6 vs 60.4): by 22-34 points on AIME,
38 on MATH500 (thinking on), 25 on GPQA-Diamond, 10 on IFEval and 13-19 on MultiPL-E. Both are behind Qwen3.5-9B-Base
except on IFEval.

Notes for reading the OEL column:

- OEL Excellent does not reason in practice. Its template supports `<think>`, but with thinking enabled its answers carry no
  `</think>` and stay short (AIME / MATH500 answers ~1-2K characters), so "thinking on" vs "off" mostly changes the prompt,
  not the behaviour. It is evaluated as an instruct model; a reasoning SFT is not a like-for-like target.
- AIME24 (32.0, all 10 samples between 27 and 37) is far above AIME25 (9.3) and out of line with MATH500 (47.0), and it drops
  to 2.7 with thinking off. Worth checking whether AIME 2024 problems are in `OpenEuroLLM-Excellent-SFT`; not verified.
- The two SFTs differ in base (Birger's 128K extension vs our mid-train), data and sequence length, so this compares
  recipes end to end, not SFT datasets in isolation.
- Our greedy HumanEval+ is low because the reasoning often never closes under greedy decoding (50 of 164 greedy outputs
  have no code block vs 4 of 164 sampled).

Agentic evals (SWE-bench, Terminal-Bench) were not run for OEL Excellent. Our SFT v2: SWE-bench Verified (random 100) 17.3,
Terminal-Bench 2.1 3.0, TBLite 7.7 (terminus-2, mean of 3 runs). OpenEuroLLM's multilingual suite (`oellm-multilingual`)
for our SFT v2, our stage-4 mid-train and Qwen3.5-9B-Base is in [`oellm_multilingual.md`](oellm_multilingual.md).

Sources: `../runs/20261027-oel-prelude-sft-excellent-evalchemy-*`, `../runs/20261025-prelude-sft-v2-full-s9020-*`,
`../runs/20261023-qwen35-9b-base-*`; full comparison with Qwen3.5-9B, Grug and Nemotron in
[`sft_v2_full_vs_qwen35.md`](sft_v2_full_vs_qwen35.md).
