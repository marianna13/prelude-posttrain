# Prelude 9B post-training: context extension and SFT

Code and data recipe for taking [OpenEuroLLM Prelude 9B](https://huggingface.co/openeurollm/prelude) (4K context,
branch `anneal300b_multilingual1_dominant2_iter_0989075`) to a 128K-context chat/reasoning/agent model:

| Step | Context | Tokens | Steps | Notes |
|---|---|---:|---:|---|
| Stage 1 | 16K | 0.5B | 607 | RoPE theta 100K -> 9M |
| Stage 2 | 32K | 1.0B | 1,214 | |
| Stage 3 | 64K | 2.0B | 2,598 | stages 1-3: one cosine schedule, peak 1e-5 |
| Stage 4 | 128K | 1.0B | 1,001 | YaRN x2 over 64K trained, LR 5e-6 |
| SFT v2 | 64K | 37.7B (30.1B trained) | 9,020 | from stage 4, LR 2e-5 cosine, GBS 64 x 64K |

The data recipe (every training item as a pointer to its upstream record) is the Hugging Face dataset
[`marianna13/prelude-9b-posttrain-recipe`](https://huggingface.co/datasets/marianna13/prelude-9b-posttrain-recipe).
Sources, weights and processing are in [sft/prelude/DATA.md](sft/prelude/DATA.md); the full build log with every
decision and measurement is [docs/issue-0054-midtraining-data.md](docs/issue-0054-midtraining-data.md) (context
extension) and [docs/issue-0058-sft-v2.md](docs/issue-0058-sft-v2.md) (SFT v2).

## Layout

| Path | Contents |
|---|---|
| `scripts/data/` | Mid-training data pipeline (census, repository selection, adapters/masks, dedup + decontamination, mixture, export, stage-4 selection, Megatron packing) |
| `scripts/data/sft_v2/` | SFT v2 census, decontamination, SWE overlap, selection + normalization, loop scan |
| `scripts/data/decontam/` | Benchmark 13-gram index definitions and conversation decontamination |
| `scripts/sft_gen/` | SFT rendering to token IDs + loss masks (`prerender_levanter.py`) and AgentTrove conversion |
| `sft/prelude/` | Tokenizer preparation, loader format, step docs (`MIDTRAIN.md`, `CURATE_MIDTRAIN.md`, `MIXTURE_MIDTRAIN.md`, `EXPORT_MIDTRAIN.md`), evidence JSONs with code/artifact hashes |
| `sft/prelude/megatron/` | Megatron-Bridge trainer used for every stage and SFT, HF export with loss parity, RULER gates ([README](sft/prelude/megatron/README.md)) |
| `sft/levanter/` | Levanter chat loader/trainer used for loader verification and the early pilot runs |
| `evals/prelude_midtrain/` | Gate definitions, RULER/ChatML inference and scoring, result tables |
| `inventory/` | Dataset and checkpoint records (revisions, token counts, paths) |
| `recipe/` | Build of the published recipe and the scripts that rebuild training data from it |

The code is a snapshot of the scripts that produced the released data and checkpoints, unchanged, because their
SHA256s are pinned in the evidence/manifest files. They therefore keep the absolute paths of the clusters they ran
on (JURECA `/p/...` for data, MareNostrum 5 `/gpfs/...` for training); set them for your system. The `recipe/`
scripts are path-free.

## Rebuilding the training data

The recipe holds no third-party text. It pins, per training item, the upstream repository, revision, file, SHA256,
row index or byte offset, raw-record hash and the SHA256 of the final token IDs and loss mask. The rebuild scripts
download the upstream files, run the original pipeline code on the selected records and stop on any difference.

```bash
pip install -r requirements-data.txt
hf download marianna13/prelude-9b-posttrain-recipe --repo-type dataset --local-dir recipe_data

# mid-training stage (stage1_16k | stage2_32k | stage3_64k | stage4_128k)
python recipe/rebuild_midtrain.py --recipe recipe_data --stage stage1_16k --tokenizer recipe_data/tokenizer \
    --cache upstream --out midtrain_stage1
python scripts/data/pack_prelude_stage_megatron.py --release midtrain_stage1 --context 16384 \
    --out packed/16384 --pad-multiple 4 --max-docs 128

# SFT v2
python recipe/rebuild_sft_v2.py --recipe recipe_data --tokenizer recipe_data/tokenizer --cache upstream --out sft_v2
python scripts/data/pack_prelude_stage_megatron.py --inputs 'sft_v2/rendered/*/part-*.parquet' --context 65536 \
    --out packed/sft_v2 --pad-multiple 4 --max-docs 128
```

Use `--limit N` (mid-training) or `--limit-parts N` (SFT) for a quick check first. Upstream sources that need
access approval on the Hub (some NVIDIA datasets) require `hf auth login` with an account that accepted their terms.
Nemotron-CC files are fetched from `data.commoncrawl.org`.

Every rebuilt parent is checked token for token against the release. Packing is a separate, seeded step: the
SFT v2 rebuild keeps our part files and row order, so packing it as above (seed 20261022, the default) reproduces
the trained arrays row for row (577,303 rows of 64K, 37.68B tokens). For mid-training, the rebuild writes parts in a
different order than our export, so packing gives the same documents in a different row order. (Stages 1-3 were
trained in Levanter's packing order, exported by `sft/prelude/megatron/export_stage.py`; stage 4 with the packing
above at 131072.)

## Training

Megatron-Bridge 0.3.0rc0 / Megatron-Core 0.16.0rc0, Transformer Engine 2.18.0, PyTorch 2.13.0 + CUDA 13.0,
flash-attn 2.8.3, NCCL 2.29.7, transformers 5.15.1 (MareNostrum 5 container `mn5_sft_cu13.sif`). See
[sft/prelude/megatron/README.md](sft/prelude/megatron/README.md) for the launch commands, the loss masking and
packed-sequence attention, and the export checks. The trainer patches some Megatron-Bridge 0.3 internals (THD
context-parallel split, cooldown schedule), so other versions need checking.

## Licences

Code: Apache-2.0 ([LICENSE](LICENSE)). Data: the recipe only references upstream records; each upstream dataset keeps
its own licence and terms (listed in [sft/prelude/DATA.md](sft/prelude/DATA.md)). The mid-training code pool
includes repositories and files without a licence (`license_type: no_license` in each record's `provenance_json`
or `repo_files_json`); filter on it if you need permissively licensed code only.
