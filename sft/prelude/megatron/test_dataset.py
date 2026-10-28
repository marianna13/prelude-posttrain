"""CPU check of PackedStage (megatron_midtrain.py) against the parity export of Levanter's first rows."""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

src = open(sys.argv[1]).read()
ns = {"np": np, "torch": torch, "Path": Path, "dataclass": dataclass, "MAX_DOCS": 64}
exec(src[src.index("class PackedStage(torch.utils.data.Dataset)"):src.index("def main():")], ns)
ds = ns["PackedStage"](sys.argv[2], 16384, 0, 607 * 64)
ref = np.load(sys.argv[3])
print("len", len(ds), "rows", ds.rows)
for k in (0, 5, 127):
    it = ds[k]
    cu = it["cu_seqlens"][: int(it["cu_seqlens_argmin"])].tolist()
    pos = it["position_ids"].numpy()
    print(f"sample {k}: tokens==ref {np.array_equal(it['tokens'].numpy(), ref['tokens'][k])} "
          f"mask==ref {np.array_equal(it['loss_mask'].numpy(), ref['loss_weight'][k])} "
          f"trained={int(it['loss_mask'].sum())} segments={len(cu) - 1} cu_head={cu[:3]} cu_tail={cu[-2:]} "
          f"positions_reset={all(pos[s] == 0 for s in cu[:-1])} "
          f"labels_shifted={np.array_equal(it['labels'].numpy()[:-1], it['tokens'].numpy()[1:])} "
          f"max_seqlen={int(it['max_seqlen'])}")
print("past last row: trained", int(ds[38791 + 3]["loss_mask"].sum()))
batch = torch.utils.data.default_collate([ds[0], ds[1]])
print({k: tuple(v.shape) for k, v in batch.items()})
