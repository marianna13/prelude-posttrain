"""Per-sequence masked next-token loss on exported Levanter packed sequences, for parity checks.

engine=hf    : HF transformers Qwen3, each document run separately (no packing) -> ground truth
engine=mcore : Megatron-Core model built by Megatron-Bridge from the same HF checkpoint, whole packed
               sequence in THD format with per-document attention (cu_seqlens) and per-document positions
engine=mcore_ckpt : as mcore, but MODEL is a Megatron-Bridge checkpoint iteration (megatron_midtrain.py);
               the Bridge is built from $BASE_HF and the checkpoint's own config sets theta
Writes [{seq, loss_sum, count}] JSON. Single GPU, single process.
Usage: python parity_loss.py ENGINE MODEL NPZ FIRST LAST THETA OUT.json [MASTER_PORT]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

engine, model_path, npz, first, last, theta, out = sys.argv[1:8]
first, last, theta = int(first), int(last), float(theta)
data = np.load(npz)
tokens, weights, segments = data["tokens"], data["loss_weight"], data["segment_ids"]
CHUNK = 2048


def documents(seg):
    """[(start, end)] for each document; padding (-1) is only at the tail."""
    real = np.where(seg >= 0)[0]
    if not len(real):
        return []
    end = real[-1] + 1
    starts = [0] + [i for i in range(1, end) if seg[i] != seg[i - 1]]
    return list(zip(starts, starts[1:] + [end]))


def masked_ce(hidden, lm_head, ids, w):
    """Sum of CE(next token) * w over positions, chunked; hidden [T, H] for one contiguous span."""
    total = torch.zeros((), dtype=torch.float64, device="cuda")
    for c in range(0, len(ids) - 1, CHUNK):
        sl = slice(c, min(c + CHUNK, len(ids) - 1))
        logits = lm_head(hidden[sl]).float()
        ce = F.cross_entropy(logits, ids[sl.start + 1:sl.stop + 1], reduction="none")
        total += (ce * w[sl]).sum(dtype=torch.float64)
    return total


results = []
if engine == "hf":
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_path)
    cfg.rope_parameters = {**(cfg.rope_parameters or {}), "rope_theta": theta}
    cfg.max_position_embeddings = max(cfg.max_position_embeddings, tokens.shape[1])
    model = AutoModelForCausalLM.from_pretrained(model_path, config=cfg, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    print("hf rope", model.config.rope_parameters, flush=True)
    with torch.no_grad():
        for i in range(first, last + 1):
            ids_all = torch.as_tensor(tokens[i], dtype=torch.long, device="cuda")
            w_all = torch.as_tensor(weights[i], device="cuda")
            loss = torch.zeros((), dtype=torch.float64, device="cuda")
            for s, e in documents(segments[i]):
                h = model.model(input_ids=ids_all[s:e][None]).last_hidden_state[0]
                loss += masked_ce(h, model.lm_head, ids_all[s:e], w_all[s:e])
            results.append({"seq": i, "loss_sum": loss.item(), "count": float(weights[i].sum())})
            print(i, results[-1], flush=True)
else:
    os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=sys.argv[8] if len(sys.argv) > 8 else "29600")
    from transformers import Qwen3Config

    def _get_rope_theta(self):
        rope = self.__dict__.get("rope_parameters") or {}
        if "rope_theta" in rope:
            return rope["rope_theta"]
        if "_rope_theta_compat" in self.__dict__:
            return self.__dict__["_rope_theta_compat"]
        raise AttributeError("rope_theta")

    def _set_rope_theta(self, value):
        self.__dict__["_rope_theta_compat"] = value

    if "rope_theta" not in vars(Qwen3Config):
        Qwen3Config.rope_theta = property(_get_rope_theta, _set_rope_theta)

    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    if engine == "mcore_ckpt":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import megatron_midtrain  # noqa: F401  version shims only
        model = AutoBridge.from_hf_pretrained(os.environ["BASE_HF"]).load_megatron_model(
            model_path, wrap_with_ddp=False)[0].eval()
        provider = model.config
        assert provider.rotary_base == theta, (provider.rotary_base, theta)
    else:
        provider = AutoBridge.from_hf_pretrained(model_path).to_megatron_provider(load_weights=True)
        provider.bf16, provider.fp16, provider.params_dtype = True, False, torch.bfloat16
        provider.seq_length = provider.max_position_embeddings = tokens.shape[1]
        provider.rotary_base = theta
        if hasattr(provider, "finalize"):
            provider.finalize()
        model = provider.provide_distributed_model(wrap_with_ddp=False)[0].eval()
    gpt = model.module if hasattr(model, "module") else model
    print("mcore rotary_base", provider.rotary_base, "attention_backend", getattr(provider, "attention_backend", None),
          flush=True)
    with torch.no_grad():
        for i in range(first, last + 1):
            docs = documents(segments[i])
            T = docs[-1][1]
            ids = torch.as_tensor(tokens[i][:T], dtype=torch.long, device="cuda")
            pos = torch.as_tensor(np.concatenate([np.arange(e - s) for s, e in docs]), dtype=torch.long, device="cuda")
            cu = torch.as_tensor([0] + [e for _, e in docs], dtype=torch.int32, device="cuda")
            maxlen = max(e - s for s, e in docs)
            psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                                  max_seqlen_q=maxlen, max_seqlen_kv=maxlen)
            logits = model(input_ids=ids[None], position_ids=pos[None], attention_mask=None,
                           packed_seq_params=psp)
            logits = logits.reshape(-1, logits.shape[-1])[:T]
            w = torch.as_tensor(weights[i][:T], device="cuda")
            loss = torch.zeros((), dtype=torch.float64, device="cuda")
            for c in range(0, T - 1, CHUNK):
                sl = slice(c, min(c + CHUNK, T - 1))
                ce = F.cross_entropy(logits[sl].float(), ids[sl.start + 1:sl.stop + 1], reduction="none")
                loss += (ce * w[sl]).sum(dtype=torch.float64)
            results.append({"seq": i, "loss_sum": loss.item(), "count": float(weights[i].sum())})
            print(i, results[-1], flush=True)
            del logits
json.dump(results, open(out, "w"))
print("done", out)
