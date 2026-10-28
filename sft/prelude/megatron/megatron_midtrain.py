"""Prelude 9B mid-training on Megatron-Bridge from the exported Levanter stages (export_stage.py).

One process per GPU (srun). Reads a stage directory of packed rows, feeds them as THD packed sequences with
per-document attention, trains with a per-token loss over each global batch, and resumes from CKPT_DIR when
it holds a checkpoint. Env: MODEL OUT RUN_NAME CKPT_DIR STAGE_DIR SEQ_LEN GBS [MBS] [TP] [CP] ROPE_THETA
END_ITER [SCHEDULE_ITERS] [STAGE_START] [STAGE_START_SAMPLE] [RECOMPUTE] [GRAD_FP32] [SAVE_INTERVAL] [EXIT_MINS]
[LR] [WARMUP] [YARN_FACTOR] [YARN_ORIG] [CONCAT] [DECAY_START] [DECAY_ITERS].
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import os

import torch
import megatron.core.jit as _mcore_jit

# Megatron-Bridge 0.3.0rc0 imports disable_jit_fuser, which mcore@71e418 (in mn5_sft_cu13.sif) lacks.
# Bridge only calls it when dist.disable_jit_fuser is set (a Blackwell workaround), which stays off here.
if not hasattr(_mcore_jit, "disable_jit_fuser"):
    def _disable_jit_fuser():
        raise RuntimeError("disable_jit_fuser is not available in this megatron-core build")
    _mcore_jit.disable_jit_fuser = _disable_jit_fuser

from transformers import Qwen3Config

# transformers 5.x keeps rope_theta in rope_parameters; Megatron-Bridge 0.3's Qwen3 bridge reads
# hf_config.rope_theta. Expose it read-only from rope_parameters.
def _get_rope_theta(self):
    rope = self.__dict__.get("rope_parameters") or {}
    if "rope_theta" in rope:
        return rope["rope_theta"]
    if "_rope_theta_compat" in self.__dict__:
        return self.__dict__["_rope_theta_compat"]
    raise AttributeError("rope_theta")  # transformers probes this while building the config


def _set_rope_theta(self, value):
    self.__dict__["_rope_theta_compat"] = value


if "rope_theta" not in vars(Qwen3Config):
    Qwen3Config.rope_theta = property(_get_rope_theta, _set_rope_theta)

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.qwen.qwen3 import _qwen3_common
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.config import DatasetProvider
from megatron.bridge.training.pretrain import pretrain
import megatron.bridge.training.optim as _bridge_optim

# Bridge 0.3.0rc0 calls get_megatron_optimizer(config, model, no_weight_decay_cond, scale_lr_cond, lr_mult, ...)
# positionally; mcore@71e418 replaced those three with config_overrides. Drop them and use the core's
# default parameter groups. Weight-decay grouping does not affect this throughput test.
_core_get_megatron_optimizer = _bridge_optim.get_megatron_optimizer


def _get_megatron_optimizer_compat(config, model_chunks, no_weight_decay_cond=None, scale_lr_cond=None,
                                   lr_mult=1.0, **kwargs):
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[throughput] get_megatron_optimizer shim: dropped no_weight_decay_cond="
              f"{no_weight_decay_cond is not None} scale_lr_cond={scale_lr_cond is not None} lr_mult={lr_mult}",
              flush=True)
    return _core_get_megatron_optimizer(config, model_chunks, **kwargs)


_bridge_optim.get_megatron_optimizer = _get_megatron_optimizer_compat

# Bridge's training_log reads optimizer.decoupled_lr, which this mcore OptimizerConfig no longer has.
# None means "no decoupled LR" in Bridge, matching how it is configured here.
from megatron.core.optimizer import OptimizerConfig as _CoreOptimizerConfig

if not hasattr(_CoreOptimizerConfig, "decoupled_lr"):
    _CoreOptimizerConfig.decoupled_lr = None

# Megatron-FSDP's get_mesh_names reads torch's private _mesh_resources.child_to_root_mapping, which
# torch 2.13 removed; flattened sub-meshes now live in root_mesh._flatten_mapping.
import megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor as _fsdp_uneven
import megatron.core.distributed.fsdp.src.megatron_fsdp.utils as _fsdp_utils


def _get_mesh_names(device_mesh=None):
    if device_mesh is None:
        return []
    names = list(device_mesh.mesh_dim_names or [])
    if device_mesh._get_root_mesh() == device_mesh:
        names += [n for n in getattr(device_mesh, "_flatten_mapping", {}) if n not in names]
    return names


if not hasattr(_fsdp_utils._mesh_resources, "child_to_root_mapping"):
    _fsdp_utils.get_mesh_names = _get_mesh_names
    _fsdp_uneven.get_mesh_names = _get_mesh_names

env = os.environ.get
STAGE_DIR = Path(env("STAGE_DIR"))
SEQ_LEN = int(env("SEQ_LEN"))
GBS, MBS = int(env("GBS")), int(env("MBS", "1"))
TP, CP = int(env("TP", "4")), int(env("CP", "1"))
ROPE_THETA = float(env("ROPE_THETA"))  # required: the production theta is still being decided
END_ITER = int(env("END_ITER"))  # global step to stop at (stage end, or earlier for a smoke)
SCHEDULE_ITERS = int(env("SCHEDULE_ITERS", "4419"))
STAGE_START = int(env("STAGE_START", "0"))  # global step at which this stage's row 0 is consumed
# Samples Megatron has consumed before this stage's row 0. It counts samples, not steps, so a stage that follows one
# with a different GBS must pass it (stage 2 after 607 steps x 64: 38848).
STAGE_START_SAMPLE = int(env("STAGE_START_SAMPLE", str(STAGE_START * GBS)))
LR = float(env("LR", "1e-5"))  # peak; cosine to 1e-6
WARMUP = int(env("WARMUP", "100"))  # LR warmup steps (100 for every mid-train/SFT stage so far)
# YaRN on top of ROPE_THETA, as in HF rope_parameters {rope_type: yarn, factor, original_max_position_embeddings}
# with the default attention factor (0.1 ln factor + 1) and integer correction range.
YARN_FACTOR, YARN_ORIG = float(env("YARN_FACTOR", "0")), int(env("YARN_ORIG", "65536"))
# Smoke only: sample = CONCAT consecutive stage rows, so SEQ_LEN = CONCAT x the stage's row length.
CONCAT = int(env("CONCAT", "1"))
MAX_DOCS = 64

import megatron.bridge.training.gpt_step as _gpt_step

# Bridge 0.3 splits every batch tensor into 2*CP equal slices of the flat row, which cuts across packed documents
# and also slices cu_seqlens. For THD use TE's per-document partition (each document in 2*CP chunks, so document
# lengths must be multiples of 2*CP) and keep cu_seqlens global; padded == actual since padding is in-segment.
_core_cp_split, _core_packed_params = _gpt_step.get_batch_on_this_cp_rank, _gpt_step.get_packed_seq_params


def _thd_cp_split(batch):
    if CP == 1 or batch.get("cu_seqlens") is None:
        return _core_cp_split(batch)
    import transformer_engine_torch as tex
    from megatron.core import parallel_state
    cu = batch["cu_seqlens"].squeeze(0)[: int(batch["cu_seqlens_argmin"].item())].to(torch.int32)
    index = tex.thd_get_partitioned_indices(cu, batch["tokens"].size(1), CP, parallel_state.get_context_parallel_rank())
    for key in ("tokens", "labels", "loss_mask", "position_ids"):
        if batch.get(key) is not None:
            batch[key] = batch[key].index_select(1, index)
    return batch


def _thd_packed_params(batch):
    params = _core_packed_params(batch)
    if CP > 1:
        params.cu_seqlens_q_padded = params.cu_seqlens_kv_padded = params.cu_seqlens_q
    return params


_gpt_step.get_batch_on_this_cp_rank, _gpt_step.get_packed_seq_params = _thd_cp_split, _thd_packed_params

# GPTModel calls the YaRN embedding without packed_seq, so under CP it is sliced per rank and THD RoPE (which
# slices per document itself) gets too few positions. Every batch here is THD-packed.
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding as _Yarn

_yarn_forward = _Yarn.forward  # an lru_cache; Megatron calls forward.cache_clear()


def _yarn_forward_packed(self, max_seq_len, offset=0, packed_seq=True):
    return _yarn_forward(self, max_seq_len, offset, True)


_yarn_forward_packed.cache_clear = _yarn_forward.cache_clear
_Yarn.forward = _yarn_forward_packed
# The yarn_* settings are not provider dataclass fields, so run_config.yaml keeps only position_embedding_type: yarn.
# Class defaults let a reload of a YaRN checkpoint (export_hf.py, parity_loss.py) rebuild the same rotary embedding.
if YARN_FACTOR:
    from megatron.bridge.models.gpt_provider import GPTModelProvider as _Provider
    for _k, _v in dict(yarn_rotary_scaling_factor=YARN_FACTOR, yarn_original_max_position_embeddings=YARN_ORIG,
                       yarn_beta_fast=32.0, yarn_beta_slow=1.0, yarn_mscale=1.0, yarn_mscale_all_dim=0.0,
                       yarn_correction_range_round_to_int=True).items():
        setattr(_Provider, _k, _v)

# TE's fused cross entropy allocates an fp32 gradient the size of the local logits (16 GiB at 64K tokens x 64K
# vocab shard). Mid-run the caching allocator can hold that much free memory only in fragments; release the cache
# first when the free memory would not fit the buffer.
import megatron.core.models.common.language_module.language_module as _lm_module

_te_cross_entropy = _lm_module.te_parallel_cross_entropy


def _te_cross_entropy_defrag(logits, *args, **kwargs):
    if torch.cuda.mem_get_info()[0] < logits.numel() * 4 + (1 << 30):
        torch.cuda.empty_cache()
    return _te_cross_entropy(logits, *args, **kwargs)


if _te_cross_entropy is not None:
    _lm_module.te_parallel_cross_entropy = _te_cross_entropy_defrag

from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler as _Scheduler

# The LR schedule follows optimizer steps (warmup WARMUP, default 100, cosine over SCHEDULE_ITERS) like Levanter's. Megatron counts
# it in samples (warmup/decay x GBS, stepped by GBS), which breaks when GBS halves between stages at constant
# positions per step. Count steps instead; stage-1 checkpoints stored samples (decay 4419 x 64) and are converted.
_scheduler_init, _scheduler_step = _Scheduler.__init__, _Scheduler.step


def _scheduler_init_steps(self, *args, **kwargs):
    _scheduler_init(self, *args, **kwargs)
    self.lr_warmup_steps, self.lr_decay_steps, self.wd_incr_steps = WARMUP, SCHEDULE_ITERS, SCHEDULE_ITERS
    _scheduler_step(self, 0)


def _scheduler_step_steps(self, increment):
    _scheduler_step(self, increment // GBS)  # the train loop passes the samples of one optimizer step


def _scheduler_load_steps(self, sd):
    decay = sd.get("lr_decay_steps", sd.get("decay_steps"))
    self.num_steps = round(sd["num_steps"] * SCHEDULE_ITERS / decay)
    _scheduler_step(self, 0)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[midtrain] scheduler from checkpoint num_steps={sd['num_steps']} decay={decay} -> step "
              f"{self.num_steps} lr={self.get_lr({'max_lr': self.max_lr, 'min_lr': self.min_lr}):.4e}", flush=True)


_Scheduler.__init__, _Scheduler.step, _Scheduler.load_state_dict = (
    _scheduler_init_steps, _scheduler_step_steps, _scheduler_load_steps)

# Cooldown branch (WSD-style): from step DECAY_START the LR falls linearly from the scheduled value at DECAY_START to
# min_lr over DECAY_ITERS steps, so an intermediate checkpoint can be annealed and compared with the full run.
DECAY_START, DECAY_ITERS = int(env("DECAY_START", "0")), int(env("DECAY_ITERS", "0"))
_scheduler_get_lr = _Scheduler.get_lr


def _scheduler_get_lr_cooldown(self, param_group):
    n = self.num_steps
    if not DECAY_ITERS or n <= DECAY_START:
        return _scheduler_get_lr(self, param_group)
    self.num_steps = DECAY_START
    lr0 = _scheduler_get_lr(self, param_group)
    self.num_steps = n
    min_lr = param_group.get("min_lr", self.min_lr)
    return lr0 + (min_lr - lr0) * min(1.0, (n - DECAY_START) / DECAY_ITERS)


_Scheduler.get_lr = _scheduler_get_lr_cooldown


@dataclass
class PackedStageConfig(DatasetProvider):
    """Rows of an exported stage (export_stage.py) as Megatron packed (THD) samples."""

    stage_dir: str = ""
    seq_length: int = 0
    stage_start_sample: int = 0
    stage_end_sample: int = 0
    skip_getting_attention_mask_from_dataset: bool = True

    def build_datasets(self, context):
        length = max(context.train_samples, self.stage_end_sample)
        return PackedStage(self.stage_dir, self.seq_length, self.stage_start_sample, length), None, None


class PackedStage(torch.utils.data.Dataset):
    """Sample k of the run is stage row (k - stage_start_sample).

    Megatron's sampler numbers samples globally (consumed_samples carries across stages on resume), so the
    stage offset keeps row 0 aligned with the stage's first step. Indices past the last row return a
    zero-loss sequence, like Levanter's padded final batch.
    """

    def __init__(self, root, seq_length, stage_start_sample, num_samples):
        root = Path(root)
        self.tokens = np.load(root / "tokens.npy", mmap_mode="r")
        self.loss_mask = np.load(root / "loss_mask.npy", mmap_mode="r")
        self.cu = np.load(root / "cu_seqlens.npy", mmap_mode="r")
        if self.tokens.shape[1] * CONCAT != seq_length:
            raise ValueError(f"{root}: rows have length {self.tokens.shape[1]} x CONCAT {CONCAT}, expected {seq_length}")
        self.rows, self.offset = len(self.tokens) // CONCAT, stage_start_sample
        self.length = max(num_samples, self.offset + self.rows)

    def __len__(self):
        return self.length

    def __getitem__(self, k):
        row = k - self.offset
        if row < 0:
            raise IndexError(f"sample {k} precedes this stage (starts at sample {self.offset})")
        real = row < self.rows
        r = row if real else 0
        rows = range(r * CONCAT, (r + 1) * CONCAT)
        tokens = torch.from_numpy(np.concatenate([np.asarray(self.tokens[i], dtype=np.int64) for i in rows]))
        loss_mask = torch.from_numpy(np.concatenate([np.asarray(self.loss_mask[i], dtype=np.float32) for i in rows]))
        if not real:
            loss_mask.zero_()
        width, bounds = self.tokens.shape[1], [0]
        for j, i in enumerate(rows):
            b = [int(x) for x in self.cu[i] if x >= 0]
            if b[-1] < width:  # tail padding attends only to itself
                b.append(width)
            bounds += [j * width + x for x in b[1:]]
        if CP > 1:
            m = 2 * CP
            if CONCAT > 1:  # smoke only: move interior boundaries down to a multiple of 2*CP
                bounds = sorted({x - x % m for x in bounds[:-1]} | {bounds[-1]})
            if any(x % m for x in bounds):
                raise ValueError(f"sample {k}: segment boundaries must be multiples of 2*CP={m} for THD CP")
        lengths = np.diff(bounds)
        position_ids = torch.from_numpy(np.concatenate([np.arange(n) for n in lengths]).astype(np.int64))
        labels = torch.roll(tokens, -1)
        labels[-1] = 0  # never trained: loss_mask is 0 at every document end and on padding
        cu = torch.full((CONCAT * self.cu.shape[1] + 1,), -1, dtype=torch.int32)  # stage width + tail padding
        cu[:len(bounds)] = torch.tensor(bounds, dtype=torch.int32)
        return {
            "tokens": tokens, "labels": labels, "loss_mask": loss_mask, "position_ids": position_ids,
            "cu_seqlens": cu, "cu_seqlens_argmin": torch.tensor(len(bounds)),
            "max_seqlen": torch.tensor(int(lengths.max()), dtype=torch.int32),
            # get_batch reads this key unconditionally; with skip_getting_attention_mask_from_dataset it is
            # replaced by None (packed THD attention needs no mask). A tensor, since collate cannot batch None.
            "attention_mask": torch.zeros(1, dtype=torch.bool),
        }


def main():
    """Build the config and train. Guarded: checkpoint writers spawn processes that re-import this module."""
    cfg = _qwen3_common(
        hf_path=env("MODEL"), dir=env("OUT"), name=env("RUN_NAME"), mock=True,
        tensor_model_parallel_size=TP, context_parallel_size=CP, sequence_parallel=TP > 1,
        train_iters=END_ITER, global_batch_size=GBS, micro_batch_size=MBS, seq_length=SEQ_LEN,
        lr=LR, min_lr=1e-6, lr_warmup_iters=WARMUP, lr_decay_iters=SCHEDULE_ITERS,
        eval_interval=10**9, save_interval=int(env("SAVE_INTERVAL", str(END_ITER))),
    )
    model = AutoBridge.from_hf_pretrained(env("MODEL")).to_megatron_provider(load_weights=True)
    for attr in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "pipeline_dtype",
                 "virtual_pipeline_model_parallel_size", "context_parallel_size", "sequence_parallel",
                 "cross_entropy_fusion_impl"):
        setattr(model, attr, getattr(cfg.model, attr))
    model.bf16, model.fp16, model.params_dtype, model.pipeline_dtype = True, False, torch.bfloat16, torch.bfloat16
    model.seq_length = SEQ_LEN
    model.max_position_embeddings = max(65536, SEQ_LEN)
    model.rotary_base = ROPE_THETA
    if YARN_FACTOR:
        model.position_embedding_type = "yarn"
        model.yarn_rotary_scaling_factor, model.yarn_original_max_position_embeddings = YARN_FACTOR, YARN_ORIG
        model.yarn_beta_fast, model.yarn_beta_slow, model.yarn_mscale, model.yarn_mscale_all_dim = 32.0, 1.0, 1.0, 0.0
        model.yarn_correction_range_round_to_int = True
    model.calculate_per_token_loss = True  # mean over every trained token in the step (all microbatches, all DP ranks)
    RECOMPUTE = env("RECOMPUTE", "selective")
    if RECOMPUTE == "full":
        model.recompute_granularity, model.recompute_method, model.recompute_num_layers = "full", "uniform", 1
    elif RECOMPUTE == "selective":
        model.recompute_granularity = "selective"
    elif RECOMPUTE.startswith("block:"):  # full recompute of the first N layers only; the rest keep activations
        model.recompute_granularity, model.recompute_method = "full", "block"
        model.recompute_num_layers = int(RECOMPUTE.split(":")[1])
    cfg.model = model

    cfg.dataset = PackedStageConfig(stage_dir=str(STAGE_DIR), seq_length=SEQ_LEN,
                                    stage_start_sample=STAGE_START_SAMPLE,
                                    stage_end_sample=STAGE_START_SAMPLE + (END_ITER - STAGE_START) * GBS,
                                    dataloader_type="single",
                                    num_workers=4, data_sharding=True, pin_memory=True)

    from megatron.bridge.training.mixed_precision import bf16_mixed

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = cfg.ddp.grad_reduce_in_fp32 = env("GRAD_FP32", "1") == "1"
    cfg.ddp.average_in_collective = False  # per-token loss: sum gradients, normalize by global token count
    cfg.optimizer.lr, cfg.optimizer.min_lr = LR, 1e-6
    cfg.optimizer.weight_decay, cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2 = 0.0, 0.9, 0.95
    cfg.optimizer.adam_eps, cfg.optimizer.clip_grad = 1e-8, 1.0
    cfg.scheduler.lr_warmup_iters, cfg.scheduler.lr_decay_iters = WARMUP, SCHEDULE_ITERS
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.train.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.logger.tensorboard_dir = None
    cfg.checkpoint.save = cfg.checkpoint.load = env("CKPT_DIR")
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.rng.seed = 20261017
    if env("EXIT_MINS"):  # save and stop before the Slurm time limit instead of losing the steps since the last save
        cfg.train.exit_duration_in_mins = int(env("EXIT_MINS"))

    if int(env("RANK", "0")) == 0:
        print(f"[midtrain] stage={STAGE_DIR} seq={SEQ_LEN} gbs={GBS} mbs={MBS} tp={TP} cp={CP} theta={ROPE_THETA} "
              f"yarn={YARN_FACTOR}/{YARN_ORIG} lr={LR} concat={CONCAT} "
              f"iters {STAGE_START}->{END_ITER} of {SCHEDULE_ITERS} start_sample={STAGE_START_SAMPLE} recompute={RECOMPUTE} "
              f"grad_fp32={cfg.ddp.grad_reduce_in_fp32} ckpt={cfg.checkpoint.save}", flush=True)
    pretrain(cfg, forward_step)


if __name__ == "__main__":
    main()
