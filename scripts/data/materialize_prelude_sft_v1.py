#!/usr/bin/env python3
"""Normalize the Prelude SFT v1 sources (math / science / code / IF-chat / SWE) into one conversation schema.

usage:
  python3 materialize_prelude_sft_v1.py --src /p/data1/mmlaion/marianna/sft_sources/nvidia \\
      --out /p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v1/normalized [--subsets math,ocr,...] \\
      [--limit N] [--workers 32]

Output: ``<subset>/part-<file>-<chunk>.parquet`` (<= 50k rows each), the sft_rows.py layout plus tool fields:
  conversations  list<{role, content, reasoning_content, tool_calls, train_loss}>
                 tool_calls = JSON list of {"name", "arguments": object} (null if none)
                 train_loss 1 on trained assistant turns, 0 on everything else
  tools_json     JSON list of OpenAI-style {"type": "function", "function": {...}} (null if none)
  mix_source     the subset name; plus uuid, license, hf_repo, hf_revision, file
and ``<subset>.stats.json`` (kept / rejected by reason). Rendering, the length cap and tokenization happen
later (prerender with rl/sft/prelude's chat tokenizer), so this step is tokenizer-independent.

Normalization:
  - reasoning_content kept on assistant turns after the last user message; dropped on earlier (context-only)
    turns, as Qwen3-style serving never shows past reasoning.
  - train_loss: IF-Chat uses its per-message ``metadata.train_turns`` (chat split: last turn only); every
    other source trains all assistant turns.
  - Rows with withheld (null) prompts are rejected (IF chat lmsys/WildChat seeds: run the dataset's
    prepare_chat_prompts.py first to restore them).
  - role "developer" is treated as "system".
  - tool_calls arrive as Python-repr strings; arguments must parse as JSON objects or the row is rejected.
  - OpenCodeReasoning (split_0): user = input, assistant = R1 output split at </think>.
"""

from __future__ import annotations

import argparse
import ast
import glob
import gzip
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

MESSAGE = pa.struct([("role", pa.string()), ("content", pa.string()), ("reasoning_content", pa.string()),
                     ("tool_calls", pa.string()), ("train_loss", pa.int8())])
SCHEMA = pa.schema([("conversations", pa.list_(MESSAGE)), ("tools_json", pa.string()), ("mix_source", pa.string()),
                    ("uuid", pa.string()), ("license", pa.string()), ("hf_repo", pa.string()),
                    ("hf_revision", pa.string()), ("file", pa.string())])

REVISIONS = {
    "Nemotron-SFT-Math-v4": "84d42ad0cb960f07f951b9baa9ed2b46a5a18c66",
    "Nemotron-SFT-Science-v2": "6536a5021222a94126968e8c92f29ee47fc8a7df",
    "Nemotron-SFT-Instruction-Following-Chat-v3": "be3b3e04ef605ac9d3f8f35b9d5a632f4a3a3402",
    "Nemotron-SFT-SWE-v3.5": "ad64129216649f86179a759a6df65d0975c5df07",
    "OpenCodeReasoning": "20a1ca19c0d050fe9057fc08339d6b370ec1c67a",
}
# subset -> (repo dir, file glob relative to it, format)
SUBSETS = {
    "math": ("Nemotron-SFT-Math-v4", "data/*.parquet", "parquet"),
    "science_syn_mcq": ("Nemotron-SFT-Science-v2", "syn_mcq.jsonl", "jsonl"),
    "science_rqa": ("Nemotron-SFT-Science-v2", "rqa.jsonl", "jsonl"),
    "science_so": ("Nemotron-SFT-Science-v2", "so.jsonl", "jsonl"),
    "science_vendor": ("Nemotron-SFT-Science-v2", "vendor.jsonl", "jsonl"),
    "if_instruction_following": ("Nemotron-SFT-Instruction-Following-Chat-v3", "data/instruction_following.jsonl", "jsonl"),
    "if_chat": ("Nemotron-SFT-Instruction-Following-Chat-v3", "data/chat.jsonl", "jsonl"),
    "swe": ("Nemotron-SFT-SWE-v3.5", "data/*.jsonl.gz", "jsonl"),
    "ocr": ("OpenCodeReasoning", "split_0/*.parquet", "parquet"),
}


class Reject(Exception):
    pass


def _loads(v, what):
    if v is None or isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        if not v.strip():
            return None
        for parse in (json.loads, ast.literal_eval):
            try:
                return parse(v)
            except (ValueError, SyntaxError):
                pass
    raise Reject(f"unparsable_{what}")


def norm_tool_calls(raw):
    calls = _loads(raw, "tool_calls")
    if not calls:
        return None
    out = []
    for c in calls:
        f = c.get("function") or c
        args = f.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                raise Reject("tool_args_not_json")
        if not isinstance(args, dict) or not f.get("name"):
            raise Reject("tool_call_malformed")
        out.append({"name": f["name"], "arguments": args})
    return json.dumps(out, ensure_ascii=False)


def norm_tools(raw):
    tools = _loads(raw, "tools")
    if not tools:
        return None
    out = []
    for t in tools:
        f = dict(t.get("function") or t)
        if isinstance(f.get("parameters"), str):
            f["parameters"] = _loads(f["parameters"], "tool_parameters")
        if not f.get("name"):
            raise Reject("tool_def_malformed")
        out.append({"type": "function", "function": {k: f[k] for k in ("name", "description", "parameters") if k in f}})
    return json.dumps(out, ensure_ascii=False)


def norm_messages(messages, train_turns=None):
    msgs = [_loads(m, "message") for m in messages]
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
    if last_user < 0:
        raise Reject("no_user")
    out = []
    for i, m in enumerate(msgs):
        role, content = m.get("role"), m.get("content")
        if role == "developer":   # OpenAI-style system prompt (Science rqa: 26% of rows)
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            raise Reject(f"role_{role}")
        if role in ("system", "user") and content is None:
            raise Reject("withheld_prompt")
        if role == "system" and not content.strip():
            continue
        if role == "user" and not content.strip():
            raise Reject("empty_user")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        tool_calls = norm_tool_calls(m.get("tool_calls")) if role == "assistant" else None
        reasoning = (m.get("reasoning_content") or "").strip() if role == "assistant" and i > last_user else ""
        trained = role == "assistant" and (train_turns[i] if train_turns is not None else True)
        if role == "assistant" and not ((content or "").strip() or tool_calls or reasoning):
            raise Reject("empty_assistant")
        out.append({"role": role, "content": content or "", "reasoning_content": reasoning or None,
                    "tool_calls": tool_calls, "train_loss": int(bool(trained))})
    if out[-1]["role"] != "assistant" or not out[-1]["train_loss"]:
        raise Reject("last_not_trained_assistant")
    return out


def convert(subset, rec):
    if subset == "ocr":
        out = rec["output"] or ""
        if "</think>" not in out or not rec.get("input") or rec["input"] == "-":
            raise Reject("ocr_no_think_or_input")
        think, answer = out.split("</think>", 1)
        conv = [{"role": "user", "content": rec["input"], "reasoning_content": None, "tool_calls": None, "train_loss": 0},
                {"role": "assistant", "content": answer.strip(), "reasoning_content": think.replace("<think>", "").strip() or None,
                 "tool_calls": None, "train_loss": 1}]
        return conv, None, rec.get("id"), rec.get("license")
    meta = rec.get("metadata")
    meta = _loads(meta, "metadata") if isinstance(meta, str) else meta
    train_turns = meta.get("train_turns") if subset.startswith("if_") and isinstance(meta, dict) else None
    if subset.startswith("if_") and (train_turns is None or len(train_turns) != len(rec["messages"])):
        raise Reject("if_train_turns_mismatch")
    conv = norm_messages(rec["messages"], train_turns)
    return conv, norm_tools(rec.get("tools")), rec.get("uuid"), rec.get("license")


def iter_records(path, fmt, limit):
    n = 0
    if fmt == "parquet":
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            for rec in pf.read_row_group(g).to_pylist():
                yield rec
                n += 1
                if limit and n >= limit:
                    return
    else:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            for line in f:
                yield json.loads(line)
                n += 1
                if limit and n >= limit:
                    return


ROWS_PER_PART = 50_000   # big single-file sources (Science so: 2.1M rows) become many parts for parallel rendering


def process_file(job):
    subset, path, fmt, out_prefix, repo, limit = job
    stats = Counter()
    part = 0
    writer = pq.ParquetWriter(f"{out_prefix}-{part:03d}.parquet", SCHEMA, compression="zstd")
    in_part = 0
    buf = []
    for rec in iter_records(path, fmt, limit):
        stats["seen"] += 1
        try:
            conv, tools, uuid, lic = convert(subset, rec)
        except Reject as e:
            stats[f"reject_{e}"] += 1
            continue
        stats["kept"] += 1
        stats["kept_tool_rows"] += bool(tools)
        buf.append({"conversations": conv, "tools_json": tools, "mix_source": subset, "uuid": uuid, "license": lic,
                    "hf_repo": f"nvidia/{repo}", "hf_revision": REVISIONS[repo], "file": os.path.basename(path)})
        in_part += 1
        if len(buf) >= 2000:
            writer.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
            buf = []
        if in_part >= ROWS_PER_PART:
            if buf:
                writer.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
                buf = []
            writer.close()
            part += 1
            in_part = 0
            writer = pq.ParquetWriter(f"{out_prefix}-{part:03d}.parquet", SCHEMA, compression="zstd")
    if buf:
        writer.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
    writer.close()
    return subset, dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="records per input file (smoke test)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for subset in a.subsets.split(","):
        repo, pattern, fmt = SUBSETS[subset]
        files = sorted(glob.glob(os.path.join(a.src, repo, pattern)))
        if not files:
            raise SystemExit(f"{subset}: no files for {repo}/{pattern}")
        os.makedirs(os.path.join(a.out, subset), exist_ok=True)
        for k, p in enumerate(files):
            jobs.append((subset, p, fmt, os.path.join(a.out, subset, f"part-{k:03d}"), repo, a.limit))
    totals = {}
    with ProcessPoolExecutor(a.workers) as pool:
        for subset, stats in pool.map(process_file, jobs):
            totals.setdefault(subset, Counter()).update(stats)
            print(subset, stats, flush=True)
    for subset, stats in totals.items():
        json.dump(dict(stats), open(os.path.join(a.out, f"{subset}.stats.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps({k: dict(v) for k, v in totals.items()}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
