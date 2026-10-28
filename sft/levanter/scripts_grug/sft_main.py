# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic Levanter chat SFT from an HF checkpoint, as a plain CLI for Slurm.

Adapted from marin's ``experiments/sft/configs/snowball_h100.py`` / ``experiments/sft/launcher.py``. The launcher builds
a ``TrainLmConfig`` and dispatches it through StepRunner + Fray (``--accelerator`` -> ``ResourceConfig``). Here Slurm
owns the resources: ``train`` builds the same config with the launcher's own helpers (``_levanter_train_config``,
``apply_output_path``) and calls ``levanter.main.train_lm.main`` in every srun rank; JAX's distributed init reads
the Slurm topology. The HF checkpoint is used verbatim (``initialize_from_hf`` + ``use_hf_model_config``), so a Grug
``grug_moe`` export trains as Levanter's ``snowball`` model with no conversion and no snowball_base.json sidecar
(SnowballConfig takes qk_mult / sliding_window from the HF config).

Two steps, so the cache is built once on a CPU node rather than by every training rank:

  prepare  tokenize a dataset into a Levanter cache (marin's tokenize pipeline, the launcher's ``chat_tokenize``)
           --format chat         rows with a messages column; the chat template (default: the tokenizer's own)
                                 must carry {% generation %} blocks -- loss on every assistant span
           --format prerendered  rows with ``ids`` / ``loss`` columns (rl/scripts/sft_gen/prerender_levanter.py):
                                 per-turn loss masks (train_loss), nothing re-tokenized
           --dataset accepts a local parquet / jsonl(.gz) file, a directory, a glob, or ``hf:<repo>[@<revision>]``
           (downloaded with huggingface_hub; run prepare where there is internet, or point HF_HOME at a filled cache)
  train    SFT from the HF checkpoint on the prepared cache. Training length: --steps N or --epochs E
           (E * rows / batch without packing, E * tokens / (seq_len * batch) with packing).

Example (one CPU node, then N GPU nodes):
  python sft_main.py prepare --dataset $MIX/levanter_prerendered/train.parquet --format prerendered \\
      --tokenizer $GRUG --cache-path $CACHE --seq-len 32768
  srun ... python sft_main.py train --model-path $GRUG --model-type snowball --cache-path $CACHE \\
      --format prerendered --output-path $OUT --seq-len 32768 --batch-size 32 --epochs 2 --lr 1e-5 --warmup 0.05
"""

from __future__ import annotations

import dataclasses
import glob
import json
import math
import os
from pathlib import Path

import click

DEFAULT_MESH = '{"axes": {"data": 1, "replica": 1, "model": 1, "context": 4, "expert": -1}, ' \
               '"compute_mapping": {"batch": ["replica_dcn", "data", "expert"], "position": "context"}}'
LLAMA3_CHAT_EOS_TOKEN_IDS = "128001,128009"


# ---- shared ---------------------------------------------------------------------------------------------------


def _read_template(chat_template: str | None, tokenizer_path: str) -> str | None:
    """A template file path, the literal template, or None for the tokenizer's own (chat_template.jinja)."""
    if chat_template is None:
        candidate = Path(tokenizer_path) / "chat_template.jinja"
        return candidate.read_text() if candidate.is_file() else None
    path = Path(chat_template)
    return path.read_text() if path.is_file() else chat_template


def _format(fmt: str, *, messages_field: str, chat_template: str | None, seq_len: int, pack: bool):
    """The dataset format; prepare and train must build the identical one (it is part of the cache identity)."""
    if fmt == "midtrain":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prelude"))
        from midtrain_format import PreludeMidtrainFormat
        return PreludeMidtrainFormat(messages_field="ids", chat_template=None, mask_user_turns=True,
                                     pack=None if pack else False, max_tokens=seq_len)
    if fmt == "prerendered":
        from scripts_grug.grug_datakit_chat import SnowballPrerenderedChatFormat

        return SnowballPrerenderedChatFormat(
            messages_field="ids",
            chat_template=None,
            mask_user_turns=True,
            pack=None if pack else False,
            ids_field="ids",
            loss_field="loss",
            max_tokens=seq_len,
        )
    from levanter.data.text.formats import ChatLmDatasetFormat

    return ChatLmDatasetFormat(
        messages_field=messages_field,
        chat_template=chat_template,
        mask_user_turns=True,
        pack=None if pack else False,
    )


def _resolve_dataset(dataset: str) -> list[str]:
    """Local files for a dataset argument (file, directory, glob or hf:<repo>[@<rev>])."""
    if dataset.startswith("hf:"):
        from huggingface_hub import snapshot_download

        repo, _, revision = dataset[3:].partition("@")
        local = snapshot_download(repo, repo_type="dataset", revision=revision or None,
                                  allow_patterns=["*.parquet", "*.jsonl", "*.jsonl.gz", "*.json"])
        dataset = local
    if os.path.isdir(dataset):
        paths = [p for ext in ("parquet", "jsonl", "jsonl.gz", "json")
                 for p in glob.glob(os.path.join(dataset, "**", f"*.{ext}"), recursive=True)]
    else:
        paths = glob.glob(dataset, recursive=True)
    paths = sorted(p for p in paths if os.path.isfile(p) and not os.path.basename(p).startswith("."))
    if not paths:
        raise click.ClickException(f"no dataset files found for {dataset!r}")
    return paths


def _default_converter_reference(config_cls: type, model_path: str) -> None:
    """Make ``config_cls.hf_checkpoint_converter()`` default to ``model_path``.

    ``levanter.main.train_lm`` first builds the converter with no argument, i.e. on the class's built-in HF hub
    reference (e.g. the public Snowball / Qwen repo), and loads that repo's tokenizer before it applies
    ``initialize_from_hf``. On an offline cluster that fails; with this default every converter, including the one
    used for the final HF export, refers to the checkpoint actually being trained.
    """
    original = config_cls.hf_checkpoint_converter

    def hf_checkpoint_converter(self, ref_checkpoint=None, *args, **kwargs):
        return original(self, ref_checkpoint or model_path, *args, **kwargs)

    config_cls.hf_checkpoint_converter = hf_checkpoint_converter


def _cache_stats(cache_path: str) -> dict:
    stats = Path(cache_path) / "train" / ".stats.json"
    if not stats.is_file():
        raise click.ClickException(f"{stats} not found: run `prepare` for this cache first")
    return json.loads(stats.read_text())


def _model_config_with_overrides(model_cls: type, model_path: str, seq_len: int,
                                model_max_seq_len: int | None, rope_theta: float | None,
                                disable_sliding_window: bool):
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if not hasattr(model_cls, "from_hf_config"):
        raise click.UsageError(f"{model_cls.__name__} does not support explicit HF config overrides")
    model_config = model_cls.from_hf_config(hf_config)
    max_seq_len = model_max_seq_len or max(int(model_config.max_seq_len), seq_len)
    if max_seq_len < seq_len:
        raise click.UsageError(f"--model-max-seq-len {max_seq_len} is shorter than --seq-len {seq_len}")
    rope = model_config.rope
    if rope_theta is not None:
        rope = dataclasses.replace(rope, theta=float(rope_theta))
    replacements = {"max_seq_len": max_seq_len, "rope": rope}
    if disable_sliding_window and hasattr(model_config, "use_sliding_window"):
        replacements["use_sliding_window"] = False
    return dataclasses.replace(model_config, **replacements)


def _has_own_checkpoints(output_path: str) -> bool:
    checkpoint_root = Path(output_path) / "checkpoints"
    return any((path / "manifest.json").is_file() for path in checkpoint_root.glob("step-*") if path.is_dir())


def _scheduled_optimizer_config(AdamConfig, *, schedule_steps: int | None, **kwargs):
    if schedule_steps is None:
        return AdamConfig(**kwargs)
    registered = AdamConfig.get_known_choices().get("prelude_scheduled_adam")
    if registered is not None:
        return registered(schedule_steps=schedule_steps, **kwargs)

    @dataclasses.dataclass(frozen=True)
    class ScheduledAdamConfig(AdamConfig):
        schedule_steps: int | None = None

        def build(self, num_train_steps):
            return super().build(self.schedule_steps or num_train_steps)

    AdamConfig.register_subclass("prelude_scheduled_adam", ScheduledAdamConfig)
    return ScheduledAdamConfig(schedule_steps=schedule_steps, **kwargs)


# ---- CLI ------------------------------------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Levanter chat SFT from an HF checkpoint (prepare a cache, then train)."""


def _common_data_options(f):
    for opt in reversed([
        click.option("--format", "fmt", type=click.Choice(["chat", "prerendered", "midtrain"]), default="chat", show_default=True),
        click.option("--messages-field", default="messages", show_default=True, help="--format chat: turns column"),
        click.option("--chat-template", default=None,
                     help="--format chat: template file or literal; default = the tokenizer's chat_template.jinja"),
        click.option("--seq-len", type=int, default=32768, show_default=True),
        click.option("--pack/--no-pack", default=False, show_default=True,
                     help="pack conversations per sequence (snowball/grug: keep off, no packed-document mask yet)"),
    ]):
        f = opt(f)
    return f


@main.command()
@click.option("--dataset", required=True, help="file, directory, glob, or hf:<repo>[@<revision>]")
@click.option("--tokenizer", required=True, help="tokenizer dir or HF id (normally the model path)")
@click.option("--cache-path", required=True)
@_common_data_options
def prepare(dataset, tokenizer, cache_path, fmt, messages_field, chat_template, seq_len, pack) -> None:
    """Tokenize DATASET into a Levanter chat cache at CACHE_PATH (CPU)."""
    from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize

    paths = _resolve_dataset(dataset)
    click.echo(f"{len(paths)} source file(s):" + "".join(f"\n  {p}" for p in paths))
    template = None if fmt in ("prerendered", "midtrain") else _read_template(chat_template, tokenizer)
    fmt_obj = _format(fmt, messages_field=messages_field, chat_template=template, seq_len=seq_len, pack=pack)
    tokenize(TokenizeConfig(train_paths=paths, validation_paths=[], cache_path=cache_path, tokenizer=tokenizer,
                            format=fmt_obj, tags=[Path(cache_path).name]))
    stats = _cache_stats(cache_path)
    click.echo(f"cache {cache_path}: {stats.get('total_elements')} rows, {stats.get('total_tokens')} tokens")


@main.command()
@click.option("--model-path", required=True, help="HF checkpoint dir or id (initialize_from_hf)")
@click.option("--model-type", default="snowball", show_default=True,
              help="Levanter LmConfig key (snowball for grug_moe exports, qwen3, llama, ...)")
@click.option("--tokenizer", default=None, help="default: --model-path")
@click.option("--cache-path", required=True, help="output of `prepare`")
@click.option("--output-path", required=True, help="checkpoints and <output>/hf exports")
@click.option("--run-id", default=None, help="default: $RUN_ID, else basename(--output-path)")
@_common_data_options
@click.option("--batch-size", type=int, default=32, show_default=True, help="global batch (sequences)")
@click.option("--per-device-parallelism", type=click.IntRange(min=1), default=None,
              help="microbatch sequences per data shard; accumulate to --batch-size")
@click.option("--steps", type=int, default=None)
@click.option("--epochs", type=float, default=None)
@click.option("--lr", type=float, default=1e-5, show_default=True)
@click.option("--min-lr-ratio", type=float, default=0.1, show_default=True)
@click.option("--warmup", type=float, default=0.03, show_default=True, help="fraction of steps (<1) or steps")
@click.option("--lr-schedule", default="cosine", show_default=True)
@click.option("--weight-decay", type=float, default=0.0, show_default=True)
@click.option("--max-grad-norm", type=float, default=1.0, show_default=True)
@click.option("--beta2", type=float, default=0.95, show_default=True)
@click.option("--mesh", "mesh_json", default=DEFAULT_MESH, show_default=True, help="MeshConfig as JSON")
@click.option("--hf-save-steps", type=int, default=None, help="HF export interval; default: only at the end")
@click.option("--checkpoint-minutes", type=int, default=30, show_default=True)
@click.option("--eos-token-ids", default=LLAMA3_CHAT_EOS_TOKEN_IDS, show_default=True)
@click.option("--hf-save-dtype", default="bfloat16", show_default=True)
@click.option("--wandb-project", default="grug-sft", show_default=True)
@click.option("--model-max-seq-len", type=int, default=None,
              help="override the HF model context length (required for Prelude long-context stages)")
@click.option("--rope-theta", type=float, default=None,
              help="override the HF RoPE theta and keep it fixed across stages")
@click.option("--disable-sliding-window", is_flag=True,
              help="disable Qwen sliding-window attention for full-context training")
@click.option("--attention-backend", type=click.Choice(["auto", "fa4", "default", "nvte", "splash", "jax_flash", "vanilla"]),
              default="auto", show_default=True,
              help="attention backend; auto selects FA4 for Prelude mid-training")
@click.option("--resume-from", default=None,
              help="native Levanter checkpoint directory for full optimizer/step continuation")
@click.option("--reset-data-on-resume", is_flag=True,
              help="restart a fresh stage cache at batch zero while preserving the native train state")
@click.option("--schedule-steps", type=int, default=None,
              help="optimizer schedule horizon; defaults to --steps")
@click.option("--finite-stage", is_flag=True, help="finite Prelude cache, zero-loss final padding, no repeats")
@click.option("--stage-start-step", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--dry-run", is_flag=True, help="print the resolved TrainLmConfig and exit")
@click.option("--allow-cpu", is_flag=True, help="allow CPU-only trainer correctness tests with tiny models")
def train(model_path, model_type, tokenizer, cache_path, output_path, run_id, fmt, messages_field, chat_template,
          seq_len, pack, batch_size, steps, epochs, lr, min_lr_ratio, warmup, lr_schedule, weight_decay,
          max_grad_norm, beta2, mesh_json, hf_save_steps, checkpoint_minutes, eos_token_ids, hf_save_dtype,
          wandb_project, model_max_seq_len, rope_theta, disable_sliding_window, attention_backend, resume_from,
          reset_data_on_resume, schedule_steps, dry_run, allow_cpu, per_device_parallelism, finite_stage, stage_start_step) -> None:
    """SFT MODEL_PATH on the prepared cache (run under srun on every rank)."""
    from datetime import timedelta

    from levanter.checkpoint import CheckpointerConfig
    from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
    from levanter.models.lm_model import LmConfig
    from levanter.optim.config import AdamConfig
    from levanter.utils.mesh import MeshConfig
    from marin.training.training import apply_output_path

    from experiments.sft.launcher import DatasetSpec, HFModel, SFTSpec, _levanter_train_config

    if (steps is None) == (epochs is None):
        raise click.UsageError("pass exactly one of --steps / --epochs")
    if reset_data_on_resume and resume_from is None:
        raise click.UsageError("--reset-data-on-resume requires --resume-from")
    if resume_from is not None and not Path(resume_from).exists():
        raise click.UsageError(f"--resume-from does not exist: {resume_from}")
    if finite_stage and (fmt != "midtrain" or epochs is not None or reset_data_on_resume):
        raise click.UsageError("--finite-stage requires midtrain/--steps and excludes --reset-data-on-resume")
    if stage_start_step and not finite_stage:
        raise click.UsageError("--stage-start-step requires --finite-stage")
    tokenizer = tokenizer or model_path
    if fmt == "midtrain":
        if model_type != "qwen3":
            raise click.UsageError("Prelude mid-training requires --model-type qwen3")
        if model_max_seq_len is None or rope_theta is None:
            raise click.UsageError("Prelude mid-training requires --model-max-seq-len and --rope-theta")
        if eos_token_ids == LLAMA3_CHAT_EOS_TOKEN_IDS:
            eos_token_ids = "262146,2"
        if set(eos_token_ids.split(",")) != {"262146", "2"}:
            raise click.UsageError("Prelude mid-training requires EOS IDs 262146,2")
    stats = _cache_stats(cache_path)
    if steps is None:
        # without packing one row is one sequence; with packing the launcher's packed-sequence count
        per_epoch = stats["total_elements"] / batch_size if not pack else stats["total_tokens"] / (seq_len * batch_size)
        steps = math.ceil(epochs * per_epoch)
    if schedule_steps is not None and schedule_steps < steps:
        raise click.UsageError(f"--schedule-steps {schedule_steps} is shorter than --steps {steps}")

    template = None if fmt in ("prerendered", "midtrain") else _read_template(chat_template, tokenizer)
    fmt_obj = _format(fmt, messages_field=messages_field, chat_template=template, seq_len=seq_len, pack=pack)
    data_config = LmDataConfig(
        tokenizer=tokenizer,
        chat_template=template,
        enforce_eos=True,
        auto_build_caches=False,  # built by `prepare`
        components={"sft": DatasetComponent(source=UrlDatasetSourceConfig(train_urls=[], cache_dir=cache_path,
                                                                          format=fmt_obj),
                                            cache_dir=cache_path, format=fmt_obj, split="train")},
        train_weights={"sft": 1.0},
    )
    mesh = json.loads(mesh_json)
    selected_attention_backend = "fa4" if fmt == "midtrain" and attention_backend == "auto" else attention_backend
    model_cls = LmConfig.get_choice_class(model_type)
    use_config_overrides = resume_from is not None or any(
        value is not None for value in (model_max_seq_len, rope_theta)
    ) or disable_sliding_window
    if use_config_overrides:
        model_config = _model_config_with_overrides(
            model_cls, model_path, seq_len, model_max_seq_len, rope_theta, disable_sliding_window
        )
    else:
        model_config = model_cls()
    if selected_attention_backend != "auto":
        from levanter.layers.attention import AttentionBackend

        if selected_attention_backend == "fa4":
            model_config = dataclasses.replace(model_config, attn_backend=AttentionBackend.DEFAULT)
        elif hasattr(model_config, "attn_backend"):
            model_config = dataclasses.replace(model_config, attn_backend=AttentionBackend(selected_attention_backend))

    spec = SFTSpec(
        name=f"checkpoints/{run_id or Path(output_path).name}",
        version="slurm",
        model=HFModel(model_ref=model_path, tokenizer_path=tokenizer, model_type=model_type),
        chat_template=template or "",
        # unused by _levanter_train_config; SFTSpec requires one source for its step/epoch check
        datasets=[DatasetSpec(slug="sft", hf_dataset_id="local", revision="local", adapter_kwargs={}, weight=1.0)],
        optimizer=_scheduled_optimizer_config(
            AdamConfig,
            schedule_steps=schedule_steps,
            learning_rate=lr,
            beta1=0.9,
            beta2=beta2,
            epsilon=1e-8,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_schedule=lr_schedule,
            warmup=warmup,
            min_lr_ratio=min_lr_ratio,
        ),
        mesh=MeshConfig(
            axes=mesh["axes"],
            dcn_axes=mesh.get("dcn_axes", {"replica_dcn": -1}),
            shared_mapping=mesh.get("shared_mapping", {}),
            compute_mapping=mesh.get("compute_mapping", {}),
            param_mapping=mesh.get("param_mapping", {"embed": "data"}),
        ),
        seq_len=seq_len,
        pack=pack,
        batch_size=batch_size,
        num_train_steps=steps,
        hf_save_dtype=hf_save_dtype,
        wandb_project=wandb_project,
    )
    config = _levanter_train_config(
        spec,
        model_config=model_config,
        data_config=data_config,
        num_train_steps=steps,
        initialize_from_hf=False if resume_from is not None else model_path,
        initialize_model_from_checkpoint_path=None,
        use_hf_model_config=not use_config_overrides,
        eos_token_ids=[int(x) for x in eos_token_ids.split(",")],
        gpu_allocator=True,
    )
    trainer = dataclasses.replace(
        config.trainer,
        id=run_id or os.environ.get("RUN_ID") or Path(output_path).name,
        checkpointer=dataclasses.replace(config.trainer.checkpointer, save_interval=timedelta(minutes=checkpoint_minutes)),
        log_dir=Path(output_path) / "logs",  # Levanter's default is a relative "logs/" (read-only in the image)
        require_accelerator=False if allow_cpu else config.trainer.require_accelerator,
        initialize_from=resume_from,
    )
    if per_device_parallelism is not None:
        trainer = dataclasses.replace(trainer, per_device_parallelism=per_device_parallelism)
    config = dataclasses.replace(config, trainer=trainer, hf_save_steps=hf_save_steps or steps)
    config = apply_output_path(config, output_path)
    config = dataclasses.replace(config, hf_save_dtype=hf_save_dtype)

    if dry_run:
        click.echo(f"steps={steps} rows={stats.get('total_elements')} tokens={stats.get('total_tokens')}")
        click.echo(f"model={type(config.model).__name__} init={config.initialize_from_hf} "
                   f"use_hf_model_config={config.use_hf_model_config} max_seq_len={config.model.max_seq_len} "
                   f"rope_theta={getattr(config.model.rope, 'theta', None)} attention={selected_attention_backend}")
        click.echo(f"optimizer={config.optimizer}")
        click.echo(f"trainer id={config.trainer.id} batch={config.trainer.train_batch_size} "
                   f"steps={config.trainer.num_train_steps} mesh={config.trainer.mesh}")
        click.echo(f"checkpoints={config.trainer.checkpointer.base_path} hf_save_path={config.hf_save_path} "
                   f"every {config.hf_save_steps} steps")
        click.echo(f"format={type(fmt_obj).__name__} pack={pack} seq_len={config.train_seq_len} "
                   f"schedule_steps={schedule_steps or steps} resume_from={resume_from}")
        return

    os.environ.setdefault("WANDB_MODE", "offline" if not os.environ.get("WANDB_API_KEY") else "online")
    if fmt == "midtrain":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prelude"))
        from prepare_midtrain_tokenizer import install_checkpoint_hooks
        contract = Path(tokenizer)/"tokenizer_contract.json"
        if not contract.is_file(): contract = Path(tokenizer).parent/"tokenizer_contract.json"
        install_checkpoint_hooks(contract)
    if selected_attention_backend == "fa4":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prelude"))
        from qwen3_fa4 import install_qwen3_fa4_attention
        install_qwen3_fa4_attention()
    _default_converter_reference(type(config.model), model_path)
    from levanter.main.train_lm import main as train_lm_main

    if finite_stage:
        from finite_stage import finite_stage as finite_stage_context
        with finite_stage_context(config.data, cache_path, tokenizer, seq_len, batch_size,
                                  stage_start_step, steps) as stage_info:
            click.echo(f"finite_stage={stage_info}")
            train_lm_main(config)
        return

    reset_loader = reset_data_on_resume and resume_from is not None and not _has_own_checkpoints(output_path)
    if not reset_loader:
        train_lm_main(config)
        return

    from levanter.data.loader import DataLoader

    original_iter_from_step = DataLoader.iter_from_step

    def iter_from_fresh_stage(_self, _start_from_batch=None):
        return original_iter_from_step(_self, 0)

    DataLoader.iter_from_step = iter_from_fresh_stage
    try:
        train_lm_main(config)
    finally:
        DataLoader.iter_from_step = original_iter_from_step


if __name__ == "__main__":
    main()
