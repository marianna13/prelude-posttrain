"""CPU check of the step-based LR schedule: stage-1 values unchanged, stage-1 checkpoint converted for GBS 32."""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ckpt = sys.argv[1]
common = torch.load(os.path.join(ckpt, "common.pt"), weights_only=False)
sd = common["opt_param_scheduler"]
print("checkpoint scheduler", {k: sd[k] for k in ("num_steps", "lr_warmup_steps", "lr_decay_steps")})
import megatron_midtrain as m  # GBS from env
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

def sched(gbs):
    opt = types.SimpleNamespace(param_groups=[{}])
    return OptimizerParamScheduler(opt, init_lr=0.0, max_lr=1e-5, min_lr=1e-6, lr_warmup_steps=100 * gbs,
                                   lr_decay_steps=4419 * gbs, lr_decay_style="cosine", start_wd=0.0, end_wd=0.0,
                                   wd_incr_steps=4419 * gbs, wd_incr_style="constant")
s = sched(m.GBS)
s.load_state_dict(sd)
lrs = {}
for it in range(608, 612):
    s.step(m.GBS)
    lrs[it] = s.optimizer.param_groups[0]["lr"]
print("GBS", m.GBS, "lr after resume", {k: "%.6e" % v for k, v in lrs.items()})
print("state_dict", {k: s.state_dict()[k] for k in ("num_steps", "lr_warmup_steps", "lr_decay_steps")})
