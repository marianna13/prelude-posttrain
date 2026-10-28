#!/usr/bin/env python3
"""Drop benchmark-contaminated rows from rendered SFT parts using decontam_conversations.py flags.

usage: python apply_flags_to_rendered.py MIX_DIR   (reads MIX_DIR/{rendered,decontam}, writes MIX_DIR/rendered_clean)

Rendered rows carry id "<mix_source>-<n>" with n the 1-based row index in the normalized part of the same name
(prerender_levanter.py), and flags carry the 0-based `row`, so row = n - 1.
"""
import glob
import json
import os
import sys
from collections import Counter

import pyarrow.compute as pc
import pyarrow.parquet as pq

mix = sys.argv[1]
stats = {}
for rp in sorted(glob.glob(f"{mix}/rendered/*/part-*.parquet")):
    sub, name = rp.split("/")[-2], os.path.basename(rp)
    flags = pq.read_table(f"{mix}/decontam/{sub}/{name.replace('.parquet', '.flags.parquet')}", columns=["row", "benchmark_free"]).to_pydict()
    bad = {r + 1 for r, ok in zip(flags["row"], flags["benchmark_free"]) if not ok}
    t = pq.read_table(rp)
    n = [int(i.rsplit("-", 1)[1]) for i in t.column("id").to_pylist()]
    keep = [x not in bad for x in n]
    out = f"{mix}/rendered_clean/{sub}/{name}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kept = t.filter(keep)
    pq.write_table(kept, out, compression="zstd")
    c = stats.setdefault(sub, Counter())
    c["rows_in"] += t.num_rows
    c["rows_out"] += kept.num_rows
    c["tokens_out"] += sum(len(x) for x in kept.column("ids").to_pylist()) if kept.num_rows < 200000 else int(pc.sum(pc.list_value_length(kept.column("ids"))).as_py())
summary = {s: dict(c) for s, c in stats.items()}
json.dump(summary, open(f"{mix}/rendered_clean/summary.json", "w"), indent=1)
for s, c in sorted(summary.items()):
    print(f"{s:26s} {c['rows_in']:>9,} -> {c['rows_out']:>9,}  (-{c['rows_in'] - c['rows_out']})  tokens {c['tokens_out'] / 1e9:.2f}B")
print("total tokens", round(sum(c["tokens_out"] for c in summary.values()) / 1e9, 2), "B")
