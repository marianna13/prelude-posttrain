#!/usr/bin/env python3
"""Flag benchmark-contaminated rows of conversation parquets (sft_rows / materialize_prelude_sft_v1 layout).

usage:
  python decontam_conversations.py --index /p/data1/mmlaion/marianna/decontam/index_prelude_v1 \\
      --inp "/p/.../normalized/*/part-*.parquet" --out /p/.../decontam [--workers 64] [--threshold 1]

Same hashing as mv/orchestration/jobs/decontaminate_ray.py (lowercase, whitespace words, md5[:8] of word 13-grams),
with two false-positive guards measured on SFT v1 (most raw hits were Miller-Rabin code, identity matrices, prime
lists, pytest `_ _ _` separators and traceback text): n-grams with fewer than 7 distinct words are skipped, and an
index n-gram that matches more than --max-df corpus rows is treated as boilerplate. A row is contaminated if it
still shares >= threshold distinct n-grams with the index. Checked text: user + system turns
(the prompt; also tool outputs) and, separately, assistant turns, so prompt leaks and answer leaks are
counted apart. Writes, per input part, <out>/<subset>/<part>.flags.parquet {row, uuid, prompt_hits,
assistant_hits, benchmark_free, raw_*_hits}, <out>/generic_hashes.json and <out>/summary.json (per subset counts). Rows are not dropped here: the
render step filters on benchmark_free.
"""

import argparse
import glob
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NGRAMS, N, T = None, 13, 1


def _init(index, threshold):
    global NGRAMS, N, T
    NGRAMS = set(np.load(os.path.join(index, "ngrams.npy")).tolist())
    N = json.load(open(os.path.join(index, "manifest.json")))["n"]
    T = threshold


MIN_DISTINCT = 7   # n-grams with <= 6 distinct words (`_ _ _ ...`, `0 & 1 & 0 \\ ...`) are not evidence


def hits(text):
    """Matched index hashes (distinct), skipping low-diversity n-grams; capped at 50."""
    words = text.lower().split()
    found = set()
    for i in range(len(words) - N + 1):
        gram = words[i:i + N]
        if len(set(gram)) < MIN_DISTINCT:
            continue
        h = int.from_bytes(hashlib.md5(" ".join(gram).encode("utf-8", "ignore")).digest()[:8], "big")
        if h in NGRAMS:
            found.add(h)
            if len(found) >= 50:   # enough evidence; saves time on fully copied benchmark items
                break
    return sorted(found)


def process(job):
    inp, out = job
    rows = []
    pf = pq.ParquetFile(inp)
    cols = [c for c in ("conversations", "uuid") if c in pf.schema_arrow.names]
    k = 0
    for b in pf.iter_batches(batch_size=500, columns=cols):
        for r in b.to_pylist():
            conv = r["conversations"]
            prompt = "\n".join(m["content"] or "" for m in conv if m["role"] != "assistant")
            answer = "\n".join((m.get("reasoning_content") or "") + "\n" + (m["content"] or "") for m in conv if m["role"] == "assistant")
            ph, ah = hits(prompt), hits(answer)
            rows.append({"row": k, "uuid": r.get("uuid"), "prompt_hashes": ph, "assistant_hashes": ah})
            k += 1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    schema = pa.schema([("row", pa.int64()), ("uuid", pa.string()), ("prompt_hashes", pa.list_(pa.uint64())),
                        ("assistant_hashes", pa.list_(pa.uint64()))])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out)
    df = Counter()   # per-hash number of rows it matched (for the generic-n-gram filter)
    for r in rows:
        for h in set(r["prompt_hashes"]) | set(r["assistant_hashes"]):
            df[h] += 1
    return out, dict(df)


def finalize(flag_files, generic, threshold):
    """Rewrite each flags file with prompt_hits / assistant_hits / benchmark_free after dropping generic hashes."""
    tot = {}
    for f in flag_files:
        t = pq.read_table(f).to_pylist()
        sub = os.path.basename(os.path.dirname(f))
        c = tot.setdefault(sub, Counter())
        out = []
        for r in t:
            ph = [h for h in r["prompt_hashes"] if h not in generic]
            ah = [h for h in r["assistant_hashes"] if h not in generic]
            free = len(ph) < threshold and len(ah) < threshold
            out.append({"row": r["row"], "uuid": r["uuid"], "prompt_hits": len(ph), "assistant_hits": len(ah),
                        "benchmark_free": free, "raw_prompt_hits": len(r["prompt_hashes"]),
                        "raw_assistant_hits": len(r["assistant_hashes"])})
            c["rows"] += 1
            c["prompt_flagged"] += len(ph) >= threshold
            c["assistant_only_flagged"] += len(ph) < threshold <= len(ah)
            c["dropped"] += not free
            c["dropped_before_generic_filter"] += max(len(r["prompt_hashes"]), len(r["assistant_hashes"])) >= threshold
        pq.write_table(pa.Table.from_pylist(out), f)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--inp", required=True, help="glob of conversation parquets; parent dir name = subset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--threshold", type=int, default=1)
    ap.add_argument("--max-df", type=int, default=20,
                    help="an index n-gram matching more than this many corpus rows is boilerplate, not a leak")
    a = ap.parse_args()
    files = sorted(glob.glob(a.inp))
    jobs = [(f, os.path.join(a.out, os.path.basename(os.path.dirname(f)), os.path.basename(f).replace(".parquet", ".flags.parquet"))) for f in files]
    df, outs = Counter(), []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.index, a.threshold)) as pool:
        for out, d in pool.map(process, jobs):
            outs.append(out)
            df.update(d)
    generic = {h for h, n in df.items() if n > a.max_df}
    tot = finalize(outs, generic, a.threshold)
    summary = {s: dict(c) | {"dropped_pct": round(100 * c["dropped"] / max(c["rows"], 1), 3)} for s, c in sorted(tot.items())}
    summary["_config"] = {"index": os.path.abspath(a.index), "threshold": a.threshold, "inputs": len(files),
                          "min_distinct_words": MIN_DISTINCT, "max_df": a.max_df, "generic_hashes": len(generic),
                          "matched_hashes": len(df)}
    json.dump(sorted(int(h) for h in generic), open(os.path.join(a.out, "generic_hashes.json"), "w"))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for s, c in summary.items():
        if not s.startswith("_"):
            print(f"{s:26s} rows {c['rows']:>9,}  prompt-flagged {c['prompt_flagged']:>7,}  answer-only {c['assistant_only_flagged']:>6,}  drop {c['dropped_pct']:.2f}%")


if __name__ == "__main__":
    main()
