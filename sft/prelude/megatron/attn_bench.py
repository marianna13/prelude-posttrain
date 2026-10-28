"""Attention backend A/B on one H100: TE DotProductAttention, THD packed, causal per document, fwd+bwd.

Shapes are one TP4 shard of Prelude 9B (8 query heads, 2 KV heads, head dim 128, bf16). Document boundaries come
from real packed rows (cu_seqlens.npy of a Megatron stage). The backend is forced with NVTE_FLASH_ATTN /
NVTE_FUSED_ATTN in the environment, one backend per process. Usage: attn_bench.py LABEL STAGE_DIR [ROWS] [OUT.jsonl]
"""
import json
import os
import sys
import time

import numpy as np
import torch
from transformer_engine.pytorch import DotProductAttention

label, stage = sys.argv[1], sys.argv[2]
n_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 8
out = sys.argv[4] if len(sys.argv) > 4 else None
H, HKV, D = 8, 2, 128
cu_all = np.load(f'{stage}/cu_seqlens.npy', mmap_mode='r')
L = np.load(f'{stage}/tokens.npy', mmap_mode='r').shape[1]
attn = DotProductAttention(H, D, num_gqa_groups=HKV, qkv_format='thd', attn_mask_type='padding_causal',
                           attention_dropout=0.0).cuda()
rows = np.random.default_rng(0).choice(len(cu_all), n_rows, replace=False)
times, flops = [], 0.0
for k, r in enumerate(rows):
    c = cu_all[r]
    c = c[c >= 0]
    T = int(c[-1])
    cu = torch.tensor(c, dtype=torch.int32, device='cuda')
    lens = np.diff(c).astype(np.float64)
    q = torch.randn(T, H, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    kv = [torch.randn(T, HKV, D, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
    g = torch.randn(T, H, D, device='cuda', dtype=torch.bfloat16)
    mx = int(lens.max())

    def step():
        o = attn(q, kv[0], kv[1], cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=mx, max_seqlen_kv=mx)
        o.backward(g.reshape(o.shape))

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 5
    f = 3.5 * 4 * H * D * float((lens * lens / 2).sum())  # causal fwd 4*H*D*n^2/2, bwd ~2.5x
    times.append(dt); flops += f
    print(f'{label} row {r}: {T} tokens, {len(lens)} docs, max {mx}: {dt * 1e3:.1f} ms  {f / dt / 1e12:.0f} TFLOP/s', flush=True)
    del q, kv, g
res = dict(label=label, stage=stage, context=L, rows=len(rows), ms_per_row=1e3 * float(np.mean(times)),
           tflops=flops / sum(times) / 1e12, flash=os.environ.get('NVTE_FLASH_ATTN'), fused=os.environ.get('NVTE_FUSED_ATTN'))
print(json.dumps(res), flush=True)
if out:
    open(out, 'a').write(json.dumps(res) + '\n')
