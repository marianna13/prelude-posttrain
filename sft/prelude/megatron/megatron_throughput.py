"""Megatron-Bridge throughput smoke for Prelude 9B (Qwen3ForCausalLM) on MN5.

Loads the HF checkpoint through AutoBridge, trains a few steps on Bridge's mock data at a fixed
sequence length, and logs per-iteration time/throughput. Not a training run: no checkpoints saved.
Knobs (env): MODEL, OUT, SEQ_LEN, GBS, MBS, ITERS, TP, CP, RECOMPUTE (full|selective|none),
FSDP (0|1), LOAD_WEIGHTS (0|1), GRAD_FP32 (0|1).
"""
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
MODEL = env("MODEL")
SEQ_LEN = int(env("SEQ_LEN", "16384"))
TP, CP = int(env("TP", "1")), int(env("CP", "1"))
FSDP = env("FSDP", "0") == "1"
RECOMPUTE = env("RECOMPUTE", "full")

cfg = _qwen3_common(
    hf_path=MODEL, dir=env("OUT"), name=env("RUN_NAME", "throughput"), mock=True,
    tensor_model_parallel_size=TP, context_parallel_size=CP, sequence_parallel=TP > 1,
    use_megatron_fsdp=FSDP, train_iters=int(env("ITERS", "12")),
    global_batch_size=int(env("GBS", "64")), micro_batch_size=int(env("MBS", "1")),
    seq_length=SEQ_LEN, lr=1e-5, min_lr=1e-6, lr_warmup_iters=2, eval_interval=10**6, save_interval=10**6,
)

# Same model settings as the recipe, but with the HF weights loaded into the Megatron model.
model = AutoBridge.from_hf_pretrained(MODEL).to_megatron_provider(load_weights=env("LOAD_WEIGHTS", "1") == "1")
for attr in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "pipeline_dtype",
             "virtual_pipeline_model_parallel_size", "context_parallel_size", "sequence_parallel", "seq_length",
             "cross_entropy_fusion_impl"):
    setattr(model, attr, getattr(cfg.model, attr))
if RECOMPUTE == "full":
    model.recompute_granularity, model.recompute_method, model.recompute_num_layers = "full", "uniform", 1
elif RECOMPUTE == "selective":
    model.recompute_granularity = "selective"
# Bridge reads torch_dtype, which transformers 5 writes as "dtype" (torch_dtype is null), so it falls
# back to fp32 weights and grad buffers (2 x 33.9 GiB). Train with bf16 weights; the distributed
# optimizer keeps sharded fp32 master copies.
model.bf16, model.fp16, model.params_dtype, model.pipeline_dtype = True, False, torch.bfloat16, torch.bfloat16
model.max_position_embeddings = max(getattr(model, "max_position_embeddings", 0) or 0, SEQ_LEN)
cfg.model = model

# The bf16_mixed recipe is applied at finalize and overrides ddp.grad_reduce_in_fp32 (default True: an
# unsharded fp32 grad buffer of 33.9 GiB per GPU at TP1), so set it on the precision config.
from megatron.bridge.training.mixed_precision import bf16_mixed

cfg.mixed_precision = bf16_mixed()
cfg.mixed_precision.grad_reduce_in_fp32 = env("GRAD_FP32", "0") == "1"
cfg.ddp.grad_reduce_in_fp32 = cfg.mixed_precision.grad_reduce_in_fp32
if FSDP:
    cfg.ddp.average_in_collective = False
cfg.train.eval_iters = 0
cfg.checkpoint.save = None
cfg.checkpoint.load = None
cfg.logger.log_interval = 1
cfg.logger.log_throughput = True
cfg.logger.tensorboard_dir = None

if int(env("RANK", "0")) == 0:
    print(f"[throughput] model={MODEL} seq={SEQ_LEN} gbs={cfg.train.global_batch_size} mbs={cfg.train.micro_batch_size} "
          f"tp={TP} cp={CP} fsdp={FSDP} recompute={RECOMPUTE} grad_fp32={cfg.ddp.grad_reduce_in_fp32} "
          f"attention_backend={getattr(model, 'attention_backend', None)} params_dtype={model.params_dtype}", flush=True)
if env("DRY_RUN") == "1":
    print(cfg.model)
    print(cfg.ddp)
    raise SystemExit(0)
pretrain(cfg, forward_step)
