# Prelude: OpenEuroLLM multilingual suite (`oellm-multilingual`)

OpenEuroLLM's [oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) (87d16d5) task group `oellm-multilingual`, run on
MareNostrum 5 with the oellm `eval_env` container (lm-eval-harness + lighteval). 172 evals per model, all completed
(`include_base_44_north_macedonian` dropped: the task does not exist in the container's lm-eval).

| Column | Model |
|---|---|
| dominant2 | `openeurollm/prelude` @ `anneal300b_multilingual1_dominant2_iter_0989075`: the base our mid-train starts from |
| stage4 | `prelude-9b-midtrain-v1-stage4-128k-yarn2-step1001`: our 4-stage long-context mid-train (16K → 128K) of dominant2 |
| SFT v2 | `prelude-9b-sft-v2-full-stage4-step9020`: our SFT v2 on stage4 |
| OEL Excellent | `openeurollm/OpenEuroLLM-Prelude-SFT-Excellent-Instructions` @6551e54c: OpenEuroLLM's SFT on Birger's 128K extension of the same anneal |
| Qwen-Base | `Qwen/Qwen3.5-9B-Base` |

Mean over the languages of each task family, in %. FLORES-200 is chrF++ (lighteval, 4-shot); MGSM is lm-eval
`mgsm_native_cot` flexible-extract; the rest are accuracy. n = number of languages / language pairs. All models are
scored as plain few-shot completion (no chat template; see the note below).

| Family | n | dominant2 | stage4 | SFT v2 | OEL Excellent | Qwen-Base |
|---|---:|---:|---:|---:|---:|---:|
| FLORES-200 x→en | 35 | 57.2 | 56.7 | 56.8 | 51.3 | **57.4** |
| FLORES-200 en→x | 35 | 54.8 | **55.1** | 53.7 | 41.9 | 53.5 |
| Belebele | 24 | 85.6 | 83.9 | 85.4 | 78.2 | **90.7** |
| MultiBLiMP | 32 | 96.8 | **97.1** | **97.1** | 89.0 | 92.3 |
| INCLUDE | 45 | 58.9 | 59.1 | 59.4 | 49.2 | **64.9** |
| Global-MMLU | 16 | 63.2 | 61.7 | 62.1 | 55.0 | **70.5** |
| MGSM | 4 | 72.1 | 72.2 | **79.9** | 75.2 | 79.1 |
| XCOPA | 15 | 61.8 | 62.6 | 61.7 | 56.1 | **64.8** |
| XWinograd | 7 | 77.0 | 76.8 | 76.4 | 67.5 | **79.7** |
| XStoryCloze | 12 | 61.8 | 61.6 | 61.8 | 55.9 | **66.1** |

- Our mid-train and SFT keep dominant2's multilingual ability: stage4 and SFT v2 are within ±1.7 points of dominant2 on
  every family, except MGSM where SFT v2 gains +7.8 (the SFT's math data).
- OEL Excellent loses 5-13 points vs the Prelude base on every family except MGSM (+3): FLORES en→x -13, XWinograd -9.5,
  INCLUDE -10, MultiBLiMP -8. Its tokenizer adds a single `<bos>` like dominant2's, so this is not a BOS artifact.
  Its base (Birger's 128K extension) was not evaluated here, so the drop cannot be split between the extension and the SFT.
- Qwen3.5-9B-Base is ahead on knowledge / reading comprehension (Global-MMLU +7-8, Belebele +5, INCLUDE +5-6); Prelude
  is ahead on MultiBLiMP grammaticality (+5) and on par on FLORES; SFT v2 matches Qwen on MGSM.

Chat-template note: lighteval (FLORES, and XCOPA et/it/tr) applies the tokenizer's chat template whenever one exists,
and all these models ship one. In the first run Qwen3.5-Base's template turned thinking on, so it spent the 300-token
budget inside `<think>` and scored ~28 chrF++. All lighteval rows were rerun with `override_chat_template=False`
(the numbers above); lm-eval tasks never use a chat template by default.

Run settings that matter: Prelude rows use `add_bos_token=True` for lm-eval; Qwen rows use `max_length=8192` (the auto
batch-size probe at 262K overflows conv1d indexing); `--max_batch_size 4`; lighteval bootstrap stderrs disabled.

Sources: MN5 `/gpfs/scratch/ehpc532/ot/oellm/runs/` `full3{,b,c,d}` (jobs 48138255, 48140346, 48148564, 48149167),
`full3e` (FLORES rerun, 48160001), `full3f` (lighteval XCOPA rerun, 48160131), `full4` (dominant2 + OEL Excellent,
48160494); FLORES diagnostic `flores_diag` (48158938). Summary from `oellm/mn5/summarize.py` in
`/p/scratch/synthlaion/dc-agent-shared/oellm` (later run dirs win per task).
