#!/usr/bin/env python3
"""Resumable source adapters for the frozen Step 1 inventory.

prepare freezes inputs/code; run --task ID handles one core work unit or replay
file; pilot checks every included source/subset/language plus selected repositories.
Normalized output retains overlength documents. Rendered output defers them with
an explicit counter for the later chunking/curriculum stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prelude_midtrain import CASCADE, STACK, SPECIALIZED, Reject, adapt, dump, render

ROOT = Path(__file__).resolve().parents[2]
CODE = ["scripts/data/prelude_midtrain.py", "scripts/data/materialize_prelude_midtrain.py",
        "scripts/data/prelude_lf_counting.py", "scripts/sft_gen/prerender_levanter.py",
        "sft/prelude/prepare_midtrain_tokenizer.py", "sft/prelude/midtrain_format.py",
        "sft/levanter/scripts_grug/grug_datakit_chat.py", "sft/levanter/scripts_grug/sft_main.py"]
NORMALIZED = pa.schema([(k, pa.string()) for k in
    ("id", "kind", "text", "messages_json", "tools_json", "provenance_json", "policy", "mix_source")])
RENDERED = pa.schema([("id", pa.string()), ("ids", pa.list_(pa.int32())), ("loss", pa.list_(pa.int8())), ("mix_source", pa.string())])


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(16 * 1024**2): h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(step2):
    out = Path(step2)
    contract = out / "tokenizer_contract.json"
    c = json.loads(contract.read_text())
    step1 = Path(c["step1"])
    if sha(step1 / "source_selection.json") != c["step1_sha256"]:
        raise ValueError("Step 1 selection changed")
    selection = json.loads((step1 / "source_selection.json").read_text())
    core_path, replay_path = step1 / "census_v2/verified_input_manifest.json", step1 / "replay_v1/verified_input_manifest.json"
    core, replay = json.loads(core_path.read_text()), json.loads(replay_path.read_text())
    files = [dict(e, namespace="core") for e in core["files"] if e["source"] != SPECIALIZED or e["subset"] in selection["sources"]["specialized"]]
    files += [dict(e, namespace="replay", subset=e["family"]) for e in replay["entries"]]
    file_ids = {e["file_id"] for e in files if e["namespace"] == "core"}
    tasks = [{**t, "task_key": "core-" + str(t["task_id"]), "namespace": "core"}
             for t in core["tasks"] if t["kind"] != "checksum" and t["file_id"] in file_ids]
    tasks += [{"task_key": "replay-" + str(e["file_id"]), "namespace": "replay", "file_id": e["file_id"], "kind": "replay"}
              for e in replay["entries"]]
    m = {"schema_version": 1, "contract_sha256": sha(contract), "step1_sha256": c["step1_sha256"],
         "core_manifest_sha256": sha(core_path), "replay_manifest_sha256": sha(replay_path),
         "repo_selection_path": selection["sources"]["stack"]["selection_path"],
         "repo_selection_sha256": selection["sources"]["stack"]["selection_sha256"],
         "code_sha256": {name: sha(ROOT / name) for name in CODE}, "files": files, "tasks": tasks,
         "scope": "adapters and verification; full run creates pre-dedup normalized rows; no final mixture or training",
         "max_render_tokens": 65536, "license_policy": ["permissive", "no_license"],
         "overlength": "keep normalized row; defer rendering without truncation"}
    m["build_id"] = hashlib.sha256(dump(m).encode()).hexdigest()
    path = out / "adapter_manifest.json"
    if path.exists() and json.loads(path.read_text()) != m:
        raise ValueError("different adapter build; use a new output directory")
    write(path, m)
    print(json.dumps({"files": len(files), "tasks": len(tasks), "build_id": m["build_id"]}), flush=True)


def validate(step2):
    step2 = Path(step2)
    m = json.loads((step2 / "adapter_manifest.json").read_text())
    if sha(step2 / "tokenizer_contract.json") != m["contract_sha256"]:
        raise ValueError("tokenizer contract changed")
    c = json.loads((step2 / "tokenizer_contract.json").read_text())
    for name, expected in c["tokenizer_files_sha256"].items():
        if sha(step2 / "tokenizer" / name) != expected:
            raise ValueError("tokenizer changed")
    for name, expected in m["code_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"adapter implementation changed: {name}")
    if sha(m["repo_selection_path"]) != m["repo_selection_sha256"]:
        raise ValueError("repo selection changed")
    return m


def parquet_rows(entry, task, wanted=None):
    pf = pq.ParquetFile(entry["path"])
    groups = [task["row_group"]] if "row_group" in task else range(pf.num_row_groups)
    start = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if g in groups and (wanted is None or any(start <= x < start+n for x in wanted)):
            cursor = start
            for batch in pf.iter_batches(batch_size=16, row_groups=[g], use_threads=False):
                for row in batch.to_pylist():
                    if wanted is None or cursor in wanted:
                        yield cursor, row, None
                    cursor += 1
                    if wanted is not None and cursor > max(wanted): return
        start += n


def records(entry, task, wanted=None):
    if entry["path"].endswith(".parquet"):
        yield from parquet_rows(entry, task, wanted)
    elif entry["path"].endswith(".jsonl"):
        with open(entry["path"], "rb") as f:
            start = task.get("start", 0)
            if start:
                f.seek(start-1)
                if f.read(1) != b"\n": f.readline()
            end = task.get("end", entry["state"]["bytes"])
            while (offset := f.tell()) < end:
                line = f.readline()
                if not line: break
                try: row = json.loads(line)
                except (ValueError, UnicodeError):
                    yield offset, None, "invalid_json"
                else: yield offset, row, None
    else:
        if entry["path"].endswith((".zst", ".zstd")):
            import zstandard
            stream = open(entry["path"], "rb")
            reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(stream))
        elif entry["path"].endswith(".gz"):
            reader = gzip.open(entry["path"], "rt")
            stream = None
        else:
            reader, stream = open(entry["path"]), None
        try:
            for index, line in enumerate(reader):
                if entry.get("row_limit") and index >= entry["row_limit"]: break
                try: row = json.loads(line)
                except ValueError: yield index, None, "invalid_json"
                else: yield index, row, None
        finally:
            reader.close()
            if stream is not None: stream.close()


def pack_normalized(row):
    return {k: row[k] for k in ("id", "kind", "text", "policy", "mix_source")} | {
        "messages_json": dump(row["messages"]) if row["messages"] is not None else None,
        "tools_json": dump(row["tools"]) if row["tools"] else None, "provenance_json": dump(row["provenance"])}


def process(step2, m, entry, task, tok, selection, *, limit=0, wanted=None, pilot=False):
    state = os.stat(entry["path"])
    if {"bytes": state.st_size, "mtime_ns": state.st_mtime_ns} != entry["state"]:
        raise ValueError("frozen source changed")
    root = Path(step2) / ("pilot" if pilot else "normalized_build") / task["task_key"]
    done = root / "stats.json"
    if done.exists():
        old = json.loads(done.read_text())
        if old["build_id"] != m["build_id"] or old["limit"] != limit or old["wanted"] != (sorted(wanted) if wanted is not None else None):
            raise ValueError("resume configuration differs")
        for file, expected in old["artifact_sha256"].items():
            if sha(root / file) != expected: raise ValueError("completed output changed")
        return old
    root.mkdir(parents=True, exist_ok=True)
    stats, normalized, rendered = Counter(), [], []
    n = 0
    with pq.ParquetWriter(root / "normalized.parquet.tmp", NORMALIZED, compression="zstd") as nw, \
         pq.ParquetWriter(root / "rendered.parquet.tmp", RENDERED, compression="zstd") as rw, \
         open(root / "rejections.jsonl.tmp", "w") as reject:
        def flush():
            if normalized:
                nw.write_table(pa.Table.from_pylist(normalized, schema=NORMALIZED)); normalized.clear()
            if rendered:
                rw.write_table(pa.Table.from_pylist(rendered, schema=RENDERED)); rendered.clear()
        for locator, record, error in records(entry, task, wanted):
            n += 1
            if limit and n > limit: break
            if entry["namespace"] == "replay" and entry.get("row_limit") and locator >= entry["row_limit"]: break
            stats["scanned_rows"] += 1
            try:
                if error: raise Reject(error)
                selected = selection.get(f"{entry['file_id']}:{locator}") if entry["source"] == STACK else None
                row = adapt(record, entry, locator, stats, selected_repo=selected)
                normalized.append(pack_normalized(row))
                try:
                    r = render(row, tok, m["max_render_tokens"])
                except Reject as e:
                    stats["render_deferred_" + str(e)] += 1
                    reject.write(dump({"locator": locator, "stage": "render", "reason": str(e), "id": row["id"]}) + "\n")
                else:
                    rendered.append(r)
                    stats["rendered_rows"] += 1
                    stats["tokens"] += len(r["ids"])
                    stats["target_tokens"] += sum(r["loss"])
            except Reject as e:
                stats["rejected_" + str(e)] += 1
                reject.write(dump({"locator": locator, "stage": "normalize", "reason": str(e)}) + "\n")
            if n % 64 == 0 or (normalized and len(normalized[-1].get("text") or "") > 8*1024**2): flush()
        flush()
    artifacts = {}
    for name in ("normalized.parquet", "rendered.parquet", "rejections.jsonl"):
        (root / (name + ".tmp")).replace(root / name)
        artifacts[name] = sha(root / name)
    report = {"build_id": m["build_id"], "task_key": task["task_key"], "source": entry["source"],
              "subset": entry["subset"], "file_id": entry["file_id"], "namespace": entry["namespace"],
              "limit": limit, "wanted": sorted(wanted) if wanted is not None else None,
              "scope": "bounded source verification" if pilot else "full work unit",
              "counts": dict(stats), "artifact_sha256": artifacts}
    write(done, report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare", "run", "pilot"))
    ap.add_argument("--step2", required=True)
    ap.add_argument("--task")
    ap.add_argument("--limit", type=int, default=3, help="pilot records per group; run always processes the whole task")
    ap.add_argument("--deps", default="/p/project1/laionize/marianna/dc_agent/.prelude_census_deps")
    a = ap.parse_args()
    if a.command == "prepare":
        prepare(a.step2); return
    sys.path.insert(0, a.deps)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Path(a.step2) / "tokenizer", local_files_only=True)
    m = validate(a.step2)
    selected = {r["id"]: r for r in pq.read_table(m["repo_selection_path"]).to_pylist()}
    entries = {(e["namespace"], e["file_id"]): e for e in m["files"]}
    if a.command == "run":
        task = next(t for t in m["tasks"] if t["task_key"] == a.task)
        process(a.step2, m, entries[(task["namespace"], task["file_id"])], task, tok, selected)
        return
    # One file for every core source/subset and every replay file (all 23 EU
    # languages and both disjoint English PDF files). Stack uses selected rows.
    groups, reports = {}, []
    for e in m["files"]:
        key = (e["source"], e["relative_path"] if e["source"] == CASCADE else e["subset"]) if e["namespace"] == "core" else ("replay", e["file_id"])
        groups.setdefault(key, e)
    stack_entry = next(e for e in m["files"] if e["source"] == STACK and any(s.startswith(str(e["file_id"])+":") for s in selected))
    groups[(STACK, "data")] = stack_entry
    for e in groups.values():
        wanted = None
        if e["source"] == STACK:
            candidates = [r for k,r in selected.items() if k.startswith(str(e["file_id"])+":")]
            preferred = [min(r["row"] for r in candidates if r["mix_category"] == category)
                         for category in ("code", "long_documents") if any(r["mix_category"] == category for r in candidates)]
            wanted = set((preferred + sorted(r["row"] for r in candidates if r["row"] not in preferred))[:a.limit])
        if e.get("family") == "long_documents":
            step1 = Path(json.loads((Path(a.step2)/"tokenizer_contract.json").read_text())["step1"])
            metrics = pq.read_table(step1 / "replay_v1" / f"{e['file_id']:04d}.parquet").to_pylist()
            eligible = [r for r in metrics if r["eligible"]]
            preferred = [min(r["row"] for r in eligible if lo <= r["tokens"] < hi)
                         for lo,hi in ((32768,65537),(65537,float("inf"))) if any(lo <= r["tokens"] < hi for r in eligible)]
            wanted = set((preferred + sorted(r["row"] for r in eligible if r["row"] not in preferred))[:a.limit])
        task = {"task_key": f"{e['namespace']}-{e['file_id']}", "kind": "pilot"}
        report = process(a.step2, m, e, task, tok, selected, limit=a.limit, wanted=wanted, pilot=True)
        reports.append(report)
        print(e["source"], e["subset"], report["counts"], flush=True)
    total = Counter()
    for r in reports: total.update(r["counts"])
    write(Path(a.step2)/"pilot_summary.json", {"build_id": m["build_id"], "scope": "bounded adapter verification; not a full source materialization",
          "groups": reports, "counts": dict(total), "deduplication_completed": False,
          "benchmark_decontamination_completed": False, "training_has_run": False})


if __name__ == "__main__": main()
