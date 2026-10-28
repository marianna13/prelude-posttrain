# Prelude mid-training qualification

Issue [0054](../../../issues/0054-build-prelude-9b-midtraining-data.md) tracks these gates.
The original six-metric, plain-text 4K reference remains in the
[evaluation registry](../runs/20261016-prelude-anneal300b-ml1-dominant2-base-fewshot.yaml).
ChatML and RULER measurements use separate protocols and are not replacements for it.

**Prelaunch measurements are complete. Infrastructure checks pass; context-quality
gates fail. Production remains held.**

The [policy](gate_policy.json) was frozen before new quality inference, in commit
`0382322`. The [finite budget](budget.json) reconciles the frozen cache with 607,
1,214 and 2,598 updates, or 4,419 cumulative updates. There are 3,499,521,164 real
seen tokens, 3,398,497,611 trained tokens and 4,633,657,344 padded positions.

## Evidence and completed measurements

JURECA root:
`/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/gates_20261018`.
MN5 root: `/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/gates_20261018`.

| Measurement | Checkpoint | Status |
|---|---|---|
| ChatML, thinking on/off, 100 examples each of GSM8K/MATH500/HumanEval+ | Initialized step-2 model | [Complete: both thinking modes](initialized_chatml_scores.json); jobs 15720292/15720329/15720353 |
| RULER, 13 tasks × 100 at 4K | Initialized step-2 model | [80.923% mean; passkey 100%](initialized_ruler_4096_scores.json); mean target failed |
| RULER at 16K | Real-cache smoke `prelude-longcontext-smoke-16k/hf/step-1` | [Complete: mean 4.323%, passkey 0%](smoke_ruler_16384_scores.json), both targets failed |
| RULER at 32K | `prelude-longcontext-continuation-32k-cp4-sharded2/hf/step-2` | [Complete: mean 3.641%, passkey 0%](smoke_ruler_32768_scores.json), both targets failed |
| RULER at 64K | `prelude-longcontext-continuation-64k-cp4-step4/hf/step-3` | [Complete: mean 0%, passkey 0%](smoke_ruler_65536_scores.json), both targets failed |
| Eight nodes / 32 H100, 20 updates at 16K, 1,048,576 positions/update | Initialized model, production cache | [Passed: 20 updates, full HF readback](eight_node_evidence.json) |

All three long-context smokes fail both retrieval targets. The base uses RoPE theta 100000 while
the draft recipe switches to 6000000. The [matched 4K diagnostic](rope_diagnostic_scores.json)
(job 47490675, all 100 frozen repeated-passkey cases, same vLLM 0.27 backend) found:

| Weights | Theta | Passkey % |
|---|---:|---:|
| Initialized | 100000 | 100 |
| Initialized | 6000000 | 81 |
| 16K smoke | 6000000 | 81 |
| 16K smoke | 100000 | 100 |

Changing theta alone harms 4K retrieval; restoring it recovers this metric on the
smoke weights. The source control also matches its vLLM 0.26 result. These controls
identify a RoPE recipe problem; they do not establish full model conversion parity.
A revised curriculum and bounded adaptation pilot remain necessary. The frozen
quality targets are unchanged.

The 4K QA failures are real: 99/100 SQuAD prompts and 76/100 Hotpot prompts ended
immediately on native EOS2. The mean target remains 85%; repeated-passkey NIAH
remains 95%. Smoke weights have only a few updates; their results characterize the
path and checkpoint, and do not qualify the fully trained stages.

Thinking-on GSM8K / MATH500 / HumanEval+ scores are 70 / 32 / 71%; thinking-off
scores are 16 / 4 / 0%. Nonthinking HumanEval+ outputs loop on `<tool_call>` and
all reach the generation limit. Thinking-on MATH has 57/100 unclosed reasoning
outputs (scored as empty answers). These are initialization format/quality findings.

## Post-stage measurements: stage 1 (16K, s607), stage 2 (32K, s1821) and stage 3 (64K, s4419)

Checkpoints `prelude-9b-midtrain-v1-stage1-16k-theta9m-step607` and
`prelude-9b-midtrain-v1-stage2-32k-theta9m-step1821` (RoPE theta 9M, 0.49B and +0.97B trained tokens) and
`prelude-9b-midtrain-v1-stage3-64k-theta9m-step4419` (64K, theta 9M, max_position_embeddings 65536).
Both use the frozen inputs above. Inference ran on JUWELS Booster (vLLM 0.26 host venv) and grading on JURECA
dc-cpu. Records are `../runs/2026102{0,1,2}-prelude-midtrain-s{1,2,3}-*`; the
[overview table](../README.md#prelude-mid-training-qualification) lists every task.

RULER mean, % (target 85; repeated-passkey target 95):

| Context | Initialized | Stage 1 | Stage 2 | Stage 3 |
|---:|---:|---:|---:|---:|
| 4K | 80.9 | **87.4** | **87.8** | 84.2 |
| 16K | - | 74.1 | 78.9 | 80.2 |
| 32K | - | 60.2 | 70.8 | 74.5 |
| 64K | - | 12.6 | 21.1 | 56.6 |

Repeated passkey passes through 32K in both stages, and fails at 64K (82 and 90). Stage 2 gains most on multi-key
NIAH (32K multikey_2 30 → 67, multikey_3 5 → 31). Single-needle tasks stay at 99–100% through 32K. RULER QA is weak
in both stages, and stage 2 regressed on it at short context (4K SQuAD 70 → 38, 16K Hotpot 39 → 15).

Stage 3 makes 64K usable: repeated passkey (niah_single_1) 100, niah_single_2/3 97/82 (stage 2: 20/54), multikey_1/2
85/73 (14/10), multiquery 73 (8), multivalue 60 (10). multikey_3 stays at 8, and 64K QA reaches only 9 (Hotpot) and
30 (SQuAD). At 32K multikey_3 rises again (31 → 65). The cost is at 4K: the mean drops 87.8 → 84.2 (below the 85
target) from fwe 72 → 46 and SQuAD 38 → 26; the needle tasks stay at 96–100.

ChatML, % thinking off / on. The 4K protocol is `inputs_chat` (4096 window, 1024-token budget). The 16K run uses
`inputs_chat_16384`, built with `prepare_gates.py --context 16384 --max-tokens 14336`; the defaults reproduce
`inputs_chat` byte for byte:

| Task | Initialized 4K | Stage 1 4K | Stage 1 16K | Stage 2 4K | Stage 2 16K | Stage 3 4K | Stage 3 16K |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 16 / 70 | 87 / 79 | 87 / 81 | 84 / 87 | 86 / 90 | 88 / 85 | 87 / 89 |
| MATH500 | 4 / 32 | 57 / 38 | 70 / 49 | 55 / 41 | 73 / 61 | 50 / 37 | 67 / 69 |
| HumanEval+ | 0 / 71 | 23 / 31 | 23 / 37 | 49 / 55 | 50 / 61 | 67 / 57 | 67 / 60 |

Stage 1's thinking HumanEval+ drop is a termination problem, not lost coding ability. Plain-text HumanEval+ is
unchanged (54.3 vs 54.3), and 31/35 of its finished thinking answers pass. With 1024 tokens, 65/100 thinking answers
hit the budget; with 14336, 57 still do. Most thinking-off truncations are degenerate loops (repeated `</think>`,
38 of 42). Stage 2 cuts the thinking HumanEval+ budget hits to 34 (4K) and 25 (16K). MATH500 benefits most from the
larger budget (+13 to +20). Stage 3 fixes thinking-off termination: HumanEval+ budget hits 34 → 0 and its score
49 → 67; at 16K, MATH500 budget hits fall to 14 (off) / 31 (on), and thinking-on MATH500 reaches 69.

Plain-text base_fewshot (4K reference protocol), point scores against the parent
`prelude-9b-anneal300b-ml1-dominant2-iter0989075`:

| Metric | Parent | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|---:|
| MATH 4-shot | 47.8 | 47.4 | 47.0 | 47.6 |
| GSM8K 8-shot | 80.5 | 79.4 | 80.1 | 79.2 |
| MMLU-Pro 5-shot | 42.2 | 43.0 | 43.9 | 43.5 |
| HumanEval+ | 54.3 | 54.3 | 51.8 | 48.8 |
| MBPP+ | 38.1 | 47.6 | 35.7 | 32.8 |
| MultiPL-E | 24.7 | 33.6 | 33.8 | 33.4 |

A 16K-window / 4096-token rerun of math + code (records `*-base-fewshot-ctx16k`) is within about 1 point of the 4K
numbers, so the budget does not limit the plain-text protocol.

Paired non-inferiority against the parent (`compare_regression.py`: 20,000 paired bootstrap replicates, Bonferroni
over the six metrics, margin 2 points, 1.5 for MMLU-Pro). Outputs: `noninferiority_{s607,s1821,s4419}.json` under the JURECA root.

| Metric | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| MATH 4-shot | pass | inconclusive | pass (−0.2 [−1.7, +1.3]) |
| GSM8K 8-shot | inconclusive | inconclusive | inconclusive (−1.3 [−3.5, +0.9]) |
| MMLU-Pro 5-shot | pass | pass | pass (+1.3) |
| HumanEval+ | inconclusive | inconclusive (−2.4 [−9.8, +5.5]) | inconclusive (−5.5 [−13.4, +2.4]) |
| MBPP+ | pass (+9.5) | inconclusive (−2.4 [−9.3, +4.5]) | inconclusive (−5.3 [−11.6, +1.1]) |
| MultiPL-E | pass | pass | pass (+8.8) |

No stage passes overall, and none fails any metric. The code intervals are wide because HumanEval+ has 164
problems and MBPP+ 378.

MBPP+ is a stopping artifact, not a skill change. Its stop sequences lack `\ndef`, so a model that writes the
function and then runs on (numbered copies such as `square_nums_2`, `File: 01_python_basics/...` paths) generates to
1,024 tokens and is cut mid-line, which fails as a syntax error. Syntax errors are 116 / 53 / 116 of 378 for parent /
stage 1 / stage 2. For stage 2, 111 of the 116 have a parseable prefix that defines the target function, and 87 show
the run-on pattern. The parent's syntax errors are a different failure (broken function bodies).
`regrade_mbpp_sanitized.py` cuts each completion back to its longest parseable prefix and re-grades it (diagnostic
only; the frozen gate is unchanged). The raw re-grade reproduces every recorded count:

| | Parent | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|---:|
| Recorded MBPP+ (pass / 378) | 144 (38.1) | 180 (47.6) | 135 (35.7) | 124 (32.8) |
| Truncated to parseable prefix | 151 (39.9) | 207 (54.8) | 207 (54.8) | 202 (53.4) |
| Completions cut | 116 | 53 | 116 | 136 |

After truncation all three stages are 13–15 points above the parent; stage 3 runs on more often (136 cut). Stage 2's apparent MBPP+
regression is entirely its run-on stopping. HumanEval+ stops on `\ndef`, `\n#`, `\nclass`, `\nif` and `\nprint`, so it
is not affected (syntax errors 2 / 2 / 6, 73 wrong for all three).
Outputs under the JURECA root: `mbpp_sanitized_regrade_{s1821,s4419}.json` (stages 2 and 3, with per-problem results) and the job
log `logs_grade/mbpp-sanitize-15726471.out` (parent and stage 1; that job timed out before writing its JSON).

## Stage 3 vs the original Prelude and Qwen3.5-9B-Base

Original = `prelude-9b-anneal300b-iter0989075` (the published Prelude anneal); mid-train start =
`prelude-9b-anneal300b-ml1-dominant2-iter0989075` (the "Initialized" model above). None of the four has agentic SFT.

| Benchmark | Original | Mid-train start | Stage 3 (64K) | Qwen3.5-9B-Base |
|---|---:|---:|---:|---:|
| *Base few-shot, 4K, plain text* | | | | |
| MATH 4-shot | 15.3 | 47.8 | 47.6 | **56.2** |
| GSM8K 8-shot | 54.2 | 80.5 | 79.2 | **84.5** |
| MMLU-Pro 5-shot | 31.1 | 42.2 | 43.5 | **58.1** |
| HumanEval+ | 17.1 | **54.3** | 48.8 | 50.0 |
| MBPP+ (raw / cut to parseable prefix) | 20.9 / - | 38.1 / 39.9 | 32.8 / 53.4 | **64.0 / 64.0** |
| MultiPL-E | 13.9 | 24.7 | 33.4 | **54.1** |
| *ChatML, 4K window, 1024-token budget (thinking off / on)* | | | | |
| GSM8K | - | 16 / 70 | 88 / 85 | - |
| MATH500 | - | 4 / 32 | 50 / 37 | - |
| HumanEval+ | - | 0 / 71 | 67 / 57 | - |
| *ChatML, 16K window, 14336-token budget (thinking off / on)* | | | | |
| GSM8K | - | - | 87 / 89 | - |
| MATH500 | - | - | 67 / 69 | - |
| HumanEval+ | - | - | 67 / 60 | - |
| *RULER mean* | | | | |
| 4K | - | 80.9 | 84.2 | - |
| 64K | - | - | 56.6 | - |

Qwen3.5-9B-Base leads every base few-shot metric except HumanEval+; its MBPP+ has no run-on problem (8 completions
cut, score unchanged; `mbpp_sanitized_regrade_qwen35_9b_base.json`). The ChatML gates and RULER use frozen
Prelude-tokenized inputs and cannot be run on Qwen as they are. Chat (evalchemy: AIME24/25, MATH500, GPQA-Diamond,
IFEval, HumanEval+, MultiPL-E; thinking on/off) and agentic (SWE-bench random 100, TBLite) evals for stage 3 and
Qwen3.5-9B-Base were submitted on 2026-10-22 (`eval_moves/prelude64k_qwen35base_20261022.sh`).

## Replay and grading

Each `inputs_*` directory contains 100 frozen raw documents and exact token-ID
prompts per task, plus `inputs-sha256.json`. Replay those files for every candidate.
Upstream samplers also use global RNG state; regeneration is not a substitute for
matching the frozen prompt hashes. `infer_gates.py` verifies every input checksum,
checks prompt length plus generation allowance, and verifies the backend retained
every prompt token. Truncation is prohibited. Raw generations and token IDs remain
under `results/<run>/outputs.jsonl`.

Use `eval_jureca.sbatch` with `GATE_INPUTS` and `GATE_RUN_ID` for the initialized
model. It uses the existing vLLM 0.26 environment. `eval_mn5.sbatch` accepts
`GATE_MODEL` and `GATE_CONTEXT` and uses the existing vLLM 0.27 image beside the
smoke weights. These versioned measurements are recorded separately. A resume
must keep model, decoding, backend versions and input checksums identical; it skips
only complete retained results.

`grade_jureca.sbatch` accepts `GATE_OUTPUTS`. Chat answers strip through the last
`</think>`; open reasoning is an empty answer. MATH uses lm-eval's `math_verify`;
GSM8K uses its flexible numeric extraction. HumanEval+ uses the full plus test
programs through unmodified `code_eval`, 20-second timeout, four workers. The
canonical positive and deliberately invalid negative controls passed.
RULER uses the installed task scorers, including partial matching for QA.
[Infrastructure adjustments](infrastructure_adjustments.json) retain the sizing
and grader fixes. They do not relax a quality threshold.

`export_regression_scores.py` converts the existing base_fewshot sample and MultiPL-E execution
artifacts into the paired format. The [exported reference](base_fewshot_paired_reference.json)
reproduces all six registered headline scores.

`compare_regression.py` implements the frozen paired noninferiority procedure.
It rejects mismatched protocol fingerprints or sample/language IDs. MultiPL-E
resamples shared problem IDs jointly across languages and retains equal language
weight. Each metric reports pass, fail or inconclusive; all six must pass for
promotion. Repeated identical greedy runs do not increase sample size.

## Finite training correctness

The draft production launcher is
[`prelude_midtrain_mn5.sbatch`](../../sft/prelude/prelude_midtrain_mn5.sbatch).
It requires an explicit `MIDTRAIN_ROPE_THETA`; the failed 6M setting is no longer
an implicit launch default. Its default runtime is a separate snapshot at
`gates_20261018/runtime_production_v1`. It consumes each shuffled cache once,
zero-pads final batches, and maps global resume steps to local stage batches.
Stage 2 and 3 require native checkpoint continuation. Resubmitting an existing stage
uses its own latest valid checkpoint before an external initializer. The normal
queue default is 24 hours (MN5 allows up to 72); long stages can span submissions
without resetting the optimizer, LR schedule or finite loader offset. The 16K pilot
measured 8,135.6 padded positions/s (128.887s/update after the first two compilations), implying about 21 hours for the
607-update first stage before later-stage throughput is qualified.

The actual pinned-runtime finite loader, padding-gradient and resume checks passed
in JURECA jobs 15720307/15720331. End-to-end tiny-model train and full-state stage
continuation passed in 15720306, with every exported parameter finite after both
tail batches. Existing loader/tokenizer/converter regression tests passed in
15720330. The scoped Haliax denominator fix prevents all-padding microbatches from
producing NaN gradients; generic SFT methods are restored on exit.

Full production training has not been submitted. Post-stage matched 4K regression,
long-context quality and ChatML measurements remain promotion requirements.

The full production run remains held because the context-quality gates failed.
