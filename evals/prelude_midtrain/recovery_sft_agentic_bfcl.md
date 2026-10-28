# Prelude recovery SFT: agentic + BFCL

Does a small agentic SFT on top of Prelude SFT v2 (the "recovery" step that helped Grug) teach agent behaviour?

| Column | Model |
|---|---|
| SFT v2 | `prelude-9b-sft-v2-full-stage4-step9020` (SFT v2 full, ~37.7B tokens, on the stage-4 128K mid-train) |
| Recovery s90 / s179 | `prelude-9b-sft-v2-full-s9020-grug-v4-recovery-step{90,179}`: SFT v2 + `grug-sft-mix-v4-40M-think` (7,453 rows: agentic 2,186, handoff 441, recovery 400, non-agentic 4,426), 64K packed, LR 1e-5, GBS 8; step 90 = one pass, 179 = two passes |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B`, post-trained with agentic SFT and RL: upper reference, not like-for-like |

Agentic: terminus-2, thinking on, 3x timeouts, isolated harbor, mean of 3 runs (per-run solved counts in brackets).
BFCL: BFCL v4 harbor adapter (single-turn categories, 3,641 tasks), one run.

| Benchmark | SFT v2 | Recovery s90 | Recovery s179 | Qwen3.5-9B |
|---|---:|---:|---:|---:|
| SWE-bench Verified, random 100 | 17.3 (19/14/19) | 18.7 (18/17/21) | **22.0** (21/21/24) | 53.7 |
| Terminal-Bench 2.1 (89) | 3.0 (3/4/1) | 3.4 (3/2/4) | **3.7** (2/4/4) | 23.2 |
| TBLite 100 | 7.7 (11/7/5) | 5.3 (5/5/6) | **9.7** (7/10/12) | 40.0 |
| BFCL v4 | 45.9 | – | **57.2** | 79.2 |

- Two passes of the recovery mix help: SWE +4.7 (every s179 run beats every SFT v2 run), BFCL +11.3, TBLite +2.0.
  TB2.1 stays at 2-4 solved tasks out of 89, within noise.
- One pass (s90) is not enough: SWE +1.4, TBLite -2.4.
- The gap to Qwen3.5-9B is still large (SWE 22 vs 54, TB2.1 4 vs 23, BFCL 57 vs 79). 7.5K rows, a third of them
  agentic, is small next to Qwen's agentic SFT + RL.
- BFCL was not run for s90.

Sources: `../runs/20261027-prelude-sft-rec-s{90,179}-*`, `../runs/20261025-prelude-sft-v2-full-s9020-{swe100,tb21,tblite100}-*`,
Qwen3.5-9B agentic from [`sft_v2_full_vs_qwen35.md`](sft_v2_full_vs_qwen35.md). BFCL from
`/p/scratch/synthlaion/dc-agent-shared/bfcl_eval/jobs/{prelude-sft-v2-full-s9020,prelude-sft-rec-s179,qwen35-9b}-think-t3-bfcl-shard*`
(merged with `scripts/eval_jureca/merge_sharded_eval.py`; not yet recorded as run YAMLs).
