"""Dump the first N packed sequences of a Prelude mid-training stage exactly as Levanter's loader builds them.

Uses the runtime's own stage_dataset() (ChatDataset packing + seed-20261017 bijective shuffle), so index i
here is Levanter's i-th training sequence. Writes tokens / loss_weight / segment_ids as an .npz.
Usage: python export_levanter_batch.py CACHE_DIR TOKENIZER CONTEXT N OUT.npz
"""
import sys

import numpy as np
from levanter.utils.thread_utils import blocking_wait
from prepare_midtrain_levanter import stage_dataset

cache, tokenizer, context, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ds = stage_dataset(cache, tokenizer, context)
print("packed sequences:", blocking_wait(ds.async_len()))
examples = blocking_wait(ds.get_batch(list(range(n))))
ex = examples[0]
print("example type:", type(ex).__name__, "fields:", [f for f in getattr(ex, "__dataclass_fields__", {})])
print("attn_mask:", type(ex.attn_mask).__name__, {k: type(v).__name__ for k, v in vars(ex.attn_mask).items()})


def arr(x):
    return np.asarray(x.array if hasattr(x, "array") else x)


tokens = np.stack([arr(e.tokens) for e in examples]).astype(np.int32)
loss_weight = np.stack([arr(e.loss_weight) for e in examples]).astype(np.float32)
seg = ex.attn_mask.segment_ids
seg = seg[0] if isinstance(seg, tuple) else seg
segment_ids = np.stack([arr((e.attn_mask.segment_ids[0] if isinstance(e.attn_mask.segment_ids, tuple)
                             else e.attn_mask.segment_ids)) for e in examples]).astype(np.int32)
np.savez(out, tokens=tokens, loss_weight=loss_weight, segment_ids=segment_ids)
for i in range(3):
    s = segment_ids[i]
    print(f"seq {i}: docs={len(np.unique(s[s >= 0]))} pad={int((s < 0).sum())} "
          f"loss_tokens={int(loss_weight[i].sum())} first_tokens={tokens[i][:6].tolist()} "
          f"seg_head={s[:4].tolist()} lw_head={loss_weight[i][:4].tolist()}")
print("totals: loss_tokens=", int(loss_weight.sum()), "positions=", tokens.size)
