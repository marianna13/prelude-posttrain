# Prelude 9B on Megatron-Bridge (MN5): mid-training stages 1-4 and SFT v2

Snapshot of the MN5 working directory `/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1/megatron_tp` (2026-10-28),
which trained every Prelude context-extension stage and SFT v2 (issues [0054](../../../../issues/0054-build-prelude-9b-midtraining-data.md),
[0058](../../../../issues/0058-build-prelude-sft-v2-intellect-3-reasoning-chat-to.md)). The sbatch files keep their MN5
absolute paths and run from that directory; the container is `mn5_sft_cu13.sif` (Megatron-Bridge 0.3, TE, cu13).
Prelude jobs now run on account `ehpc533` (pass `--account=ehpc533`; some headers still say `ehpc532`). Data and
mixture docs: [../DATA.md](../DATA.md).

## Training

| File | Role |
|---|---|
| `megatron_midtrain.py` | Trainer. Reads packed rows (`tokens.npy`, `loss_mask.npy`, `cu_seqlens.npy`) as THD packed sequences with per-document attention and per-document position ids; per-token loss over the global batch (`calculate_per_token_loss`); TP + distributed optimizer, optional CP with TE's per-document split; cosine schedule with `SCHEDULE_ITERS` and an optional linear cooldown (`DECAY_START`, `DECAY_ITERS`); YaRN (`YARN_FACTOR`, `YARN_ORIG`); resume from the run's checkpoint dir |
| `megatron_midtrain.sbatch` | Multi-node launcher; all knobs are env vars (see its `ENVS` list). Host RDMA libs are bound into the container so NCCL uses InfiniBand |
| `test_dataset.py`, `test_scheduler.py` | Checks of the packed dataset and the scheduler patch |

Packed inputs come from [`pack_prelude_stage_megatron.py`](../../../scripts/data/pack_prelude_stage_megatron.py)
(mid-training release or rendered SFT parts) or `export_stage.py` (Levanter stage order, stages 1-3).

Example, the SFT v2 full run (job 47941695):

```bash
RUN_NAME=sft-v2-full MODEL=$H/runs/stage4-128k-yarn2/hf/iter_0001001 ROPE_THETA=9000000 YARN_FACTOR=2 YARN_ORIG=65536 \
STAGE_CONTEXT=65536 STAGE_DIR=/gpfs/scratch/ehpc532/ot/prelude_sft_v2/megatron_packed/full_v2/65536 \
GBS=64 MBS=1 TP=4 CP=1 RECOMPUTE=full GRAD_FP32=1 LR=2e-5 END_ITER=9020 SCHEDULE_ITERS=9020 SAVE_INTERVAL=500 \
sbatch --account=ehpc533 --nodes=64 megatron_midtrain.sbatch
```

Resume needs the same node count (the distributed-optimizer checkpoint layout depends on the DP size).

## Export and checks

| File | Role |
|---|---|
| `export_hf.py`, `export_check.sbatch` | Megatron checkpoint -> HF safetensors + tokenizer, then loss parity of the export (HF per document) against the checkpoint (Megatron THD) |
| `sft_full_export.sbatch`, `stage4_post.sbatch` | Export with the YaRN x2 `rope_parameters` written into `config.json`; stage 4 also runs RULER |
| `post_stage_gates.sbatch` | Stages 1-3: export + frozen RULER gates |
| `parity_loss.py/.sbatch`, `yarn_parity.sbatch` | HF vs Megatron loss parity on the same packed batches |
| `ruler_chatml.sbatch`, `yarn_ruler.sbatch` | RULER plain + ChatML diagnostic |

## Data export from Levanter (stages 1-3)

`export_stage.py`, `export_stages.sbatch`, `export_levanter_batch.py`: write Levanter's packed training sequences
in the same order as Megatron arrays, so the Megatron stages train on exactly the Levanter batches.

## Benchmarks and tools

`megatron_throughput.py/.sbatch` (TP/recompute throughput on mock data), `sft_sweep.sbatch` (SFT config sweep),
`attn_bench.py`, `attn_ab.sbatch` (attention backend A/B), `check_imports.py`, `wandb_history.py`.
