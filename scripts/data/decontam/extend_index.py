#!/usr/bin/env python3
"""Extend the mv pipeline's benchmark n-gram index with the evals of the Prelude SFT/mid-training work.

usage:
  python extend_index.py --base /p/data1/datasets/mmlaion/mv_pipeline/decontam/index \\
      --spec benchmark_index_prelude.json --out /p/data1/mmlaion/marianna/decontam/index_prelude_v1

Output is a drop-in index for mv/orchestration/jobs/decontaminate_ray.py (ngrams.npy + manifest.json):
the base index's hashes plus those of every spec entry. Hashing is identical to
mv/orchestration/jobs/build_decontam_index.py: lowercase, whitespace-split words, md5 of each
space-joined word n-gram, first 8 bytes big-endian as uint64.

Spec entries are either HF datasets ({name, hf_dataset, config?, split?, text_fields, revision?}) or local harbor task sets
({name, local_glob, text_fields: ["file"]}: every matching file's text is one item, e.g. instruction.md).
A gated dataset that is only in a local cache needs a separate pass with that cache in the environment
(datasets reads HF_HOME at import):  HF_HOME=<cache> HF_DATASETS_CACHE=<cache dir used at download> HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python extend_index.py
--base <previous out> --spec <spec with only that entry> --out <new out>.
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np


def add_ngrams(text, n, acc):
    words = str(text).lower().split()
    for i in range(len(words) - n + 1):
        h = hashlib.md5(" ".join(words[i:i + n]).encode("utf-8", "ignore")).digest()[:8]
        acc.add(int.from_bytes(h, "big"))


def field(row, f):
    v = row.get(f, "")
    if isinstance(v, (list, tuple)):
        return " ".join(map(str, v))
    if isinstance(v, dict):
        return " ".join(map(str, v.values()))
    return v


def load_items(b):
    if "local_glob" in b:
        return [open(p, errors="ignore").read() for p in sorted(glob.glob(b["local_glob"]))]
    from datasets import load_dataset
    ds = load_dataset(b["hf_dataset"], b.get("config"), split=b.get("split", "test"), revision=b.get("revision"))
    keep = [c for c in b["text_fields"] if c in ds.column_names]
    if not keep:
        raise ValueError(f"none of {b['text_fields']} in {ds.column_names}")
    return [" ".join(str(field(r, f)) for f in keep) for r in ds.select_columns(keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base_manifest = json.load(open(os.path.join(a.base, "manifest.json")))
    n = base_manifest["n"]
    acc = set(np.load(os.path.join(a.base, "ngrams.npy")).tolist())
    per = {"_base": {"index": os.path.abspath(a.base), "ngrams": len(acc), "benchmarks": sorted(base_manifest["benchmarks"])}}
    failed = []
    for b in json.load(open(a.spec)):
        try:
            items = load_items(b)
        except Exception as e:  # noqa: BLE001  report and continue; the manifest records it
            print(f"WARN {b['name']}: {e}")
            per[b["name"]] = {"error": str(e)[:200]}
            failed.append(b["name"])
            continue
        before = len(acc)
        for t in items:
            add_ngrams(t, n, acc)
        per[b["name"]] = {"items": len(items), "new_ngrams": len(acc) - before}
        print(f"{b['name']:28s} items {len(items):6d}  +{len(acc) - before} ngrams", flush=True)
    os.makedirs(a.out, exist_ok=True)
    arr = np.fromiter(acc, dtype=np.uint64)
    np.save(os.path.join(a.out, "ngrams.npy"), arr)
    json.dump({"n": n, "total_ngrams": int(arr.size), "benchmarks": {**base_manifest["benchmarks"], **per}},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(f"total {arr.size} {n}-gram hashes -> {a.out}" + (f"  FAILED: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
