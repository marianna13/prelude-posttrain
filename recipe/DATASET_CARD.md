---
license: other
license_name: per-source
license_link: https://github.com/marianna13/prelude-posttrain/blob/main/sft/prelude/DATA.md
language:
- en
- multilingual
tags:
- data-recipe
- long-context
- sft
- openeurollm
pretty_name: Prelude 9B post-training data recipe
configs:
- config_name: midtrain_stage1_16k
  data_files: midtrain/stage1_16k.parquet
- config_name: midtrain_stage2_32k
  data_files: midtrain/stage2_32k.parquet
- config_name: midtrain_stage3_64k
  data_files: midtrain/stage3_64k.parquet
- config_name: midtrain_stage4_128k
  data_files: midtrain/stage4_128k.parquet
- config_name: sft_v2
  data_files: sft_v2/*.parquet
---

# Prelude 9B post-training data recipe

The exact training data of the [OpenEuroLLM Prelude 9B](https://huggingface.co/openeurollm/prelude) context-extension
stages (16K, 32K, 64K, 128K) and SFT v2, as **pointers to upstream records**. No third-party text is included: each
row names the upstream dataset, revision, file and record, with hashes that let a rebuild check it got the same
bytes and the same tokens. The code that builds the data from these pointers is
[github.com/marianna13/prelude-posttrain](https://github.com/marianna13/prelude-posttrain) (`recipe/`).

| Part | Items | Seen tokens | Trained tokens |
|---|---:|---:|---:|
| `midtrain/stage1_16k` | 186,595 parents | 0.50B | 0.49B |
| `midtrain/stage2_32k` | 347,764 parents | 1.00B | 0.97B |
| `midtrain/stage3_64k` | 673,473 parents | 2.00B | 1.94B |
| `midtrain/stage4_128k` | 334,353 parents | 1.00B | 0.96B |
| `sft_v2` (rows with `trained = true`) | 4,339,626 conversations | 37.68B | 30.08B |

## Files

- `midtrain/stage*.parquet`: one row per parent document or conversation.
  - Selection: `id` (canonical ID), `mix_source`, `pool`, `category`, `kind` (document / conversation),
    `split_group`, `context`, `original_tokens`, `seen_tokens`, `trainable_tokens`, `chunks`.
  - Upstream pointer: `source`, `hf_repo`, `revision`, `relative_path`, `file_sha256`, `locator_kind`
    (`row_index` or `byte_offset`), `locator`, `record_sha256` (SHA256 of the raw record), `family`, `language`.
  - `repo_files_json`: for whole-repository documents (stack-v3-train), the files in order with `content_id` and
    `license_type`. `provenance_json`: the full provenance (upstream metadata, no text).
  - `ids_sha256` / `loss_sha256`: SHA256 of the parent's token IDs (int32 LE) and loss mask (int8) before chunking.
- `sft_v2/<mix_source>.parquet`: one row per selected conversation.
  - `uuid` (`<source>:<row>`), `hf_repo`, `revision`, `relative_path`, `row` (row within that file),
    `census_row`; SFT v1 thinking-off rows give `upstream_uuid` instead of a row.
  - `rendered` (survived rendering), `trained` (not dropped by the loop scan), `part` / `part_row` (position in our
    rendered parts), `ids_sha256` / `loss_sha256`.
- `manifests/`: source selection, curation, mixture and release manifests, stage-4 mixture, benchmark n-gram
  exclusions, the adapter file list (`midtrain/source_files.json`), the tokenizer contract, SFT v2 selection /
  decontamination / normalization reports, SWE overlap and loop-scan lists.
- `tokenizer/`: the Prelude chat tokenizer (Prelude's tokenizer plus ChatML, think and tool tokens, and the chat
  template).

## Upstream sources

Mid-training: nvidia/Nemotron-Cascade-2-SFT-Data, nvidia/Nemotron-Pretraining-Specialized-v1,
nvidia/Nemotron-Terminal-Corpus, openbmb/UltraData-Stack-v3-L2, HuggingFaceCode/stack-v3-train,
mlfoundations/dclm-baseline-1.0, nvidia/Nemotron-CC (from data.commoncrawl.org), HuggingFaceFW/finepdfs.
SFT v2: PrimeIntellect/INTELLECT-3-SFT, open-thoughts/AgentTrove, nebius/SWE-rebench-openhands-trajectories,
nvidia/Nemotron-SFT-Instruction-Following-Chat-v3. Weights and processing:
[DATA.md](https://github.com/marianna13/prelude-posttrain/blob/main/sft/prelude/DATA.md).

## Licence

The recipe files are metadata about upstream records. Using the referenced data is governed by each upstream
dataset's licence and terms. The mid-training code pool includes repositories and files without a licence
(`license_type: no_license`); filter on it if you need permissively licensed code only. The tokenizer derives from
the Apache-2.0 Prelude model.
