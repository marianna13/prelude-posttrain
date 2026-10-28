#!/usr/bin/env python3
"""Freeze Prelude source files and run a resumable, exact token census on CPU nodes.

Commands: prepare, run, summarize. Source payloads are never rewritten. Each work
unit produces small row-metadata Parquet and a completion record; resume checks
the frozen inventory, tokenizer, source-file metadata, and implementation hash.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time

DEFAULT_SOURCES = "/p/data1/mmlaion/marianna/midtrain_sources"
DEFAULT_TOKENIZER = "/p/data1/mmlaion/marianna/sft_sources/prelude_chat_tokenizer"
DEFAULT_OUTPUT = "/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step1"
DEFAULT_DEPS = "/p/project1/laionize/marianna/dc_agent/.prelude_census_deps"
LID_MODEL = "/p/data1/datasets/mmlaion/mv_pipeline/models/lid.176.ftz"
SOURCE_REPOS = {
    "Nemotron-Cascade-2-SFT-Data": "nvidia/Nemotron-Cascade-2-SFT-Data",
    "Nemotron-Pretraining-Specialized-v1": "nvidia/Nemotron-Pretraining-Specialized-v1",
    "Nemotron-Terminal-Corpus": "nvidia/Nemotron-Terminal-Corpus",
    "UltraData-Stack-v3-L2": "openbmb/UltraData-Stack-v3-L2",
    "stack-v3-train": "HuggingFaceCode/stack-v3-train",
}
CASCADE_FILES = {
    "math/math_notool.jsonl", "math/math_tool.jsonl", "math/math_proof.jsonl",
    "science/science.jsonl", "swe/swe_agentic.jsonl", "swe/swe_agentless.jsonl",
    "conversational_agent/conversational_agent.jsonl", "chat/chat_part_1.jsonl",
    "instruction_following/instruction_following.jsonl",
}
BLOCK = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n", re.S)
RESPONSE = re.compile(r"<tool_response>.*?</tool_response>", re.S)
RESPONSE_ONLY = re.compile(r"\s*(?:<tool_response>.*?</tool_response>\s*)+", re.S)
TOKEN_BINS = [(0, 4096, "<4K"), (4096, 8192, "4-8K"),
              (8192, 16384, "8-16K"), (16384, 32768, "16-32K"),
              (32768, 65537, "32-64K"), (65537, None, ">64K")]
_TOK = _HF = _LID = _MANIFEST = _OUTPUT = _MAX_ROWS = None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(16 * 1024**2):
            h.update(b)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)


def file_state(path):
    s = os.stat(path)
    return {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def check_file(entry):
    if file_state(entry["path"]) != entry["state"]:
        raise RuntimeError(f"Frozen input changed: {entry['path']}")


def prepare(args):
    import pyarrow.parquet as pq

    sys.path.insert(0, args.deps)
    root, out = Path(args.sources).resolve(), Path(args.out).resolve()
    rev_path = root / "midtrain_revs.json"
    revisions = json.loads(rev_path.read_text())
    tokenizer_files = {p.name: sha256(p) for p in sorted(Path(args.tokenizer).iterdir()) if p.is_file()}
    files, tasks = [], []
    selection = set((root / "stack_v3_pick.txt").read_text().splitlines())
    for source, hf_repo in SOURCE_REPOS.items():
        base = root / source
        paths = sorted(p for p in base.rglob("*") if p.is_file()
                       and ".cache" not in p.parts and p.suffix in (".parquet", ".jsonl"))
        relative = {p.relative_to(base).as_posix() for p in paths}
        if source == "stack-v3-train" and relative != selection:
            raise ValueError("Local Stack files do not exactly match stack_v3_pick.txt")
        if source == "Nemotron-Cascade-2-SFT-Data" and relative != CASCADE_FILES:
            raise ValueError(f"Unexpected Cascade selection: {relative ^ CASCADE_FILES}")
        for path in paths:
            rel = path.relative_to(base).as_posix()
            entry = {"file_id": len(files), "source": source, "hf_repo": hf_repo,
                     "revision": revisions[hf_repo], "relative_path": rel,
                     "subset": str(Path(rel).parent), "path": str(path), "state": file_state(path)}
            md = base / ".cache/huggingface/download" / (rel + ".metadata")
            if md.exists():
                lines = md.read_text().splitlines()
                if lines[0] != entry["revision"]:
                    raise ValueError(f"Download revision mismatch: {path}")
                entry["upstream_etag"] = lines[1]
                if re.fullmatch(r"[a-f0-9]{64}", lines[1]):
                    entry["expected_sha256"] = lines[1]
            if path.suffix == ".parquet":
                pf = pq.ParquetFile(path)
                entry.update(rows=pf.metadata.num_rows, row_groups=pf.num_row_groups,
                             schema=str(pf.schema_arrow))
                start = 0
                for rg in range(pf.num_row_groups):
                    n = pf.metadata.row_group(rg).num_rows
                    tasks.append({"kind": "parquet", "file_id": entry["file_id"],
                                  "row_group": rg, "row_start": start, "rows": n,
                                  "weight": pf.metadata.row_group(rg).total_byte_size})
                    start += n
            else:
                step = args.jsonl_chunk_mib * 1024**2
                for start in range(0, entry["state"]["bytes"], step):
                    tasks.append({"kind": "jsonl", "file_id": entry["file_id"], "start": start,
                                  "end": min(start + step, entry["state"]["bytes"]), "weight": step})
            tasks.append({"kind": "checksum", "file_id": entry["file_id"],
                          "weight": max(1, entry["state"]["bytes"] // 10)})
            files.append(entry)
    policy = {
        "documents": {"bos": 1, "eos": 2, "preserve_whitespace": True},
        "conversations": {"template": "frozen Prelude ChatML", "all_turns": True,
                          "mask": "complete tool-response wrappers and payloads in user/tool turns",
                          "terminal_corpus": "first user is the task; later terminus-2 user turns are environment feedback",
                          "train_headers": True, "bos_has_no_target": True},
        "stack": {"primary_license": args.license_policy, "compare_license_groups": True,
                  "exclude_vendor": True, "order": "file_path, content_id",
                  "format": "repo_path + newline + join(File: path + newline + content)",
                  "min_tokens": 8192, "min_l2_file_density": 0.2, "min_l2_matches": 1},
        "language_id": {"model": LID_MODEL, "sha256": sha256(LID_MODEL), "max_chars": 3000},
        "stage": "raw inventory and license/vendor selection; dedup/decontamination are later stages",
        "percentile_boundaries": TOKEN_BINS,
    }
    manifest = {"schema_version": 1, "sources_root": str(root), "tokenizer": str(Path(args.tokenizer).resolve()),
                "tokenizer_files_sha256": tokenizer_files, "source_revisions": revisions,
                "selection_sha256": sha256(root / "stack_v3_pick.txt"),
                "implementation_sha256": sha256(__file__), "dependencies": args.deps,
                "runtime_versions": {name: importlib.metadata.version(name) for name in
                                     ("tokenizers", "transformers", "pyarrow", "numpy", "duckdb", "fasttext-wheel")},
                "source_revision_file_sha256": sha256(rev_path), "policy": policy,
                "files": files, "tasks": tasks, "status": "frozen-selection; byte verification pending"}
    manifest["build_id"] = json_hash(manifest)
    for i, task in enumerate(tasks):
        task["task_id"] = i
    existing = out / "input_manifest.json"
    if existing.exists() and json.loads(existing.read_text())["build_id"] != manifest["build_id"]:
        raise ValueError("Output already contains a different inventory; choose a new output directory")
    write_json(existing, manifest)
    print(json.dumps({"manifest": str(existing), "files": len(files), "tasks": len(tasks),
                      "bytes": sum(f["state"]["bytes"] for f in files), "build_id": manifest["build_id"]}), flush=True)


def init_worker(manifest, output, max_rows):
    global _TOK, _HF, _LID, _MANIFEST, _OUTPUT, _MAX_ROWS
    _MANIFEST, _OUTPUT, _MAX_ROWS = manifest, output, max_rows
    tokenizer = manifest["tokenizer"]
    sys.path.insert(0, manifest["dependencies"])
    import pyarrow as pa
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    import fasttext
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    _TOK = Tokenizer.from_file(str(Path(tokenizer) / "tokenizer.json"))
    _HF = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    _LID = fasttext.load_model(LID_MODEL)


def language(text):
    # Use the native API so fasttext 0.9.2 does not hit its NumPy 2 copy=False incompatibility.
    sample = " ".join(text[:6000].split())[:3000]
    if not sample:
        return "unknown"
    pred = _LID.f.predict(sample + "\n", 1, 0.0, "strict")
    return pred[0][1].replace("__label__", "") if pred else "unknown"


def conversation(record, source=""):
    messages = record.get("messages", record.get("conversations"))
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing_messages")
    normalized, seen_user = [], False
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("message_not_object")
        m = dict(m)
        role = m.get("role")
        if role == "developer":
            m["role"] = role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("unknown_role")
        if source == "Nemotron-Terminal-Corpus" and role == "user":
            if record.get("agent") != "terminus-2":
                raise ValueError("unexpected_terminal_agent")
            if seen_user:
                m["role"] = "tool"
            seen_user = True
        content = m.get("content")
        if content is None:
            if role in ("system", "user"):
                raise ValueError("withheld_prompt")
            m["content"] = ""
        elif not isinstance(content, str):
            raise ValueError("nontext_content")
        if "tool_calls" in m and isinstance(m["tool_calls"], str):
            m["tool_calls"] = json.loads(m["tool_calls"])
        normalized.append(m)
    text = _HF.apply_chat_template(normalized, tokenize=False, add_generation_prompt=False)
    spans = []
    for block in BLOCK.finditer(text):
        body = block.group(2)
        if block.group(1) == "user" and RESPONSE_ONLY.fullmatch(body):
            spans.append((block.start(2), block.end(2)))
    lang_text = "\n".join(str(m.get("content") or "") for m in normalized
                          if m["role"] == "assistant")
    return text, spans, language(lang_text)


def teacher_name(record):
    meta = record.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            meta = {}
    value = record.get("generator") or record.get("model") or meta.get("models_used") or "unknown"
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)


def repo_key(name):
    return str(name or "").removeprefix("https://github.com/").strip("/").lower()


def repo_text(name, files):
    return name + "\n\n" + "\n\n".join("File: " + f["file_path"] + "\n" + f["content"] for f in files)


def record_view(record, entry, row_id):
    source = entry["source"]
    base = {"id": f"{entry['file_id']}:{row_id}", "file_id": entry["file_id"], "row": row_id,
            "source": source, "subset": entry["subset"], "teacher": teacher_name(record),
            "license": str(record.get("license") or "unspecified"), "repo_key": "",
            "repo_path": "", "commit_id": "", "stars": 0, "num_files": 0,
            "permissive_files": 0, "eligible_files": 0, "language": "unknown",
            "quality_score": 0.0, "algo_rel_score": 0.0, "is_vendor": False}
    if source in ("Nemotron-Cascade-2-SFT-Data", "Nemotron-Terminal-Corpus"):
        text, spans, lang = conversation(record, source)
        base.update(kind="conversation", language=lang)
        return base, text, None, spans, []
    if source == "stack-v3-train":
        fs = record.get("files")
        if not isinstance(fs, list) or not fs:
            raise ValueError("missing_repo_files")
        if any(not isinstance(f.get("content"), str) or not isinstance(f.get("file_path"), str) for f in fs):
            raise ValueError("nontext_repo_file")
        fs = sorted(fs, key=lambda f: (f["file_path"], f.get("content_id") or ""))
        eligible = [f for f in fs if f.get("license_type") == "permissive" and not f.get("is_vendor")]
        nonvendor = [f for f in fs if not f.get("is_vendor")]
        path = record["repo_path"]
        base.update(kind="document", repo_key=repo_key(path), repo_path=path,
                    commit_id=str(record.get("commit_id") or ""),
                    stars=int((record.get("github_metadata") or {}).get("stars") or 0),
                    num_files=len(fs), permissive_files=sum(f.get("license_type") == "permissive" for f in fs),
                    eligible_files=len(eligible), license="mixed", language="code:repository")
        text = repo_text(path, fs)
        base["_nonvendor_text"] = text if len(nonvendor) == len(fs) else repo_text(path, nonvendor) if nonvendor else ""
        side = [{"id": base["id"], "repo_key": base["repo_key"], "content_id": str(f.get("content_id") or ""),
                 "file_path": f["file_path"], "license": str(f.get("license_type") or "unknown"),
                 "language": str(f.get("language") or "unknown"), "is_vendor": bool(f.get("is_vendor"))}
                for f in fs]
        return base, text, text if len(eligible) == len(fs) else repo_text(path, eligible) if eligible else "", [], side
    if source == "UltraData-Stack-v3-L2":
        text = record.get("content")
        meta = record.get("source_meta") or {}
        fm = meta.get("file_meta") or {}
        base.update(kind="document", repo_key=repo_key(record.get("repo_name")),
                    repo_path=str(record.get("repo_name") or ""), commit_id=str(meta.get("commit_id") or ""),
                    license=str(fm.get("license_type") or "unknown"),
                    language="code:" + str(fm.get("language") or entry["subset"].split("/")[-1]),
                    quality_score=float(record.get("quality_score") or 0),
                    algo_rel_score=float(record.get("algo_rel_score") or 0), is_vendor=bool(fm.get("is_vendor")))
        base["_nonvendor_text"] = "" if base["is_vendor"] else text
        side = [{"id": base["id"], "repo_key": base["repo_key"], "content_id": str(fm.get("content_id") or ""),
                 "file_path": str(record.get("relative_path") or ""), "license": base["license"],
                 "language": base["language"], "is_vendor": base["is_vendor"],
                 "quality_score": base["quality_score"], "algo_rel_score": base["algo_rel_score"]}]
        return base, text, text if base["license"] == "permissive" and not base["is_vendor"] else "", [], side
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty_or_nontext_document")
    base.update(kind="document", language=language(text))
    return base, text, None, [], []


def iter_records(entry, task):
    if task["kind"] == "parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(entry["path"])
        n = task["row_start"]
        for batch in pf.iter_batches(batch_size=64, row_groups=[task["row_group"]], use_threads=False):
            for row in batch.to_pylist():
                yield n, row, None
                n += 1
    else:
        with open(entry["path"], "rb") as f:
            if task["start"]:
                f.seek(task["start"] - 1)
                if f.read(1) != b"\n":
                    f.readline()
            while f.tell() < task["end"]:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("record_not_object")
                    yield offset, row, None
                except (ValueError, UnicodeError):
                    yield offset, None, "invalid_json"


def task_paths(out, task):
    key = f"{task['task_id']:06d}"
    return Path(out) / "tasks" / (key + ".json"), Path(out) / "metrics" / (key + ".parquet"), key


def process_task(task):
    manifest, output, max_rows = _MANIFEST, _OUTPUT, _MAX_ROWS
    import pyarrow as pa
    import pyarrow.parquet as pq
    entry = manifest["files"][task["file_id"]]
    done, metric, key = task_paths(output, task)
    if done.exists():
        previous = json.loads(done.read_text())
        if previous.get("build_id") == manifest["build_id"] and not previous.get("limited_to_rows"):
            return {"task_id": task["task_id"], "resumed": True}
    check_file(entry)
    start = time.monotonic()
    if task["kind"] == "checksum":
        digest = sha256(entry["path"])
        expected = entry.get("expected_sha256")
        if expected and expected != digest:
            raise ValueError(f"Upstream checksum mismatch: {entry['path']}")
        check_file(entry)
        result = {"task_id": task["task_id"], "kind": "checksum", "file_id": entry["file_id"],
                  "build_id": manifest["build_id"], "sha256": digest, "upstream_match": bool(expected),
                  "seconds": time.monotonic() - start}
        write_json(done, result)
        return result
    metric.parent.mkdir(parents=True, exist_ok=True)
    side_path = Path(output) / ("stack_files" if entry["source"] == "stack-v3-train" else "ultra_files") / (key + ".parquet")
    writer = side_writer = None
    counts, errors = Counter(), []
    buffer, side_buffer, views, texts = [], [], [], []
    pending_chars = 0

    def flush_rows():
        nonlocal writer, side_writer, buffer, side_buffer
        if buffer:
            table = pa.Table.from_pylist(buffer)
            if writer is None:
                writer = pq.ParquetWriter(str(metric) + ".tmp", table.schema, compression="zstd")
            writer.write_table(table)
            buffer = []
        if side_buffer:
            side_path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist(side_buffer)
            if side_writer is None:
                side_writer = pq.ParquetWriter(str(side_path) + ".tmp", table.schema, compression="zstd")
            side_writer.write_table(table)
            side_buffer = []

    def flush_tokens():
        nonlocal views, texts, pending_chars
        if not texts:
            return
        encoded = _TOK.encode_batch(texts, add_special_tokens=False)
        for view, encoding in zip(views, encoded):
            base, text, permissive, spans, side = view
            ids = encoding.ids
            total = len(ids) + (2 if base["kind"] == "document" else 0)
            trainable = total - 1
            if spans:
                offsets = encoding.offsets
                # Sorted, disjoint semantic response spans; scan tokens once.
                si = 0
                for ti, (a, b) in enumerate(offsets):
                    while si < len(spans) and spans[si][1] <= a:
                        si += 1
                    if ti and si < len(spans) and b > a and a < spans[si][1] and b > spans[si][0]:
                        trainable -= 1
            if permissive is None:
                eligible_tokens = total
            elif not permissive:
                eligible_tokens = 0
            elif permissive == text:
                eligible_tokens = total
            else:
                eligible_tokens = len(_TOK.encode(permissive, add_special_tokens=False).ids) + 2
            nonvendor = base.pop("_nonvendor_text", None)
            if nonvendor is None or nonvendor == text:
                nonvendor_tokens = total
            elif not nonvendor:
                nonvendor_tokens = 0
            elif nonvendor == permissive:
                nonvendor_tokens = eligible_tokens
            else:
                nonvendor_tokens = len(_TOK.encode(nonvendor, add_special_tokens=False).ids) + 2
            base.update(tokens=total, trainable_tokens=trainable, permissive_tokens=eligible_tokens,
                        all_nonvendor_tokens=nonvendor_tokens,
                        permissive_trainable_tokens=(trainable if eligible_tokens == total else max(0, eligible_tokens - 1)),
                        tool_response_spans=len(spans), chars=len(text))
            buffer.append(base)
            side_buffer.extend(side)
            counts["valid_rows"] += 1
            counts["tokens"] += total
            counts["trainable_tokens"] += trainable
            counts["permissive_tokens"] += eligible_tokens
        views, texts, pending_chars = [], [], 0
        if len(buffer) >= 4096 or len(side_buffer) >= 8192:
            flush_rows()

    try:
        for row_id, record, error in iter_records(entry, task):
            if max_rows and counts["rows_in"] >= max_rows:
                break
            counts["rows_in"] += 1
            try:
                if error:
                    raise ValueError(error)
                view = record_view(record, entry, row_id)
                text = view[1]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("empty_or_nontext_document")
                views.append(view)
                texts.append(text)
                pending_chars += len(text)
            except (ValueError, TypeError, KeyError) as exc:
                why = str(exc)[:160]
                counts["rejected_rows"] += 1
                counts["rejected:" + why] += 1
                if len(errors) < 20:
                    errors.append({"row": row_id, "reason": why})
            if len(texts) >= 128 or pending_chars >= 2 * 1024**2:
                flush_tokens()
        flush_tokens()
        flush_rows()
    finally:
        if writer is not None:
            writer.close()
        if side_writer is not None:
            side_writer.close()
    if writer is not None:
        os.replace(str(metric) + ".tmp", metric)
    if side_writer is not None:
        os.replace(str(side_path) + ".tmp", side_path)
    check_file(entry)
    result = {"task_id": task["task_id"], "kind": task["kind"], "file_id": entry["file_id"],
              "build_id": manifest["build_id"], "counts": dict(counts), "errors_sample": errors,
              "seconds": time.monotonic() - start, "limited_to_rows": max_rows}
    write_json(done, result)
    return result


def run(args):
    out = Path(args.out)
    manifest = json.loads((out / "input_manifest.json").read_text())
    if sha256(__file__) != manifest["implementation_sha256"]:
        raise ValueError("Implementation changed since prepare; use a frozen script copy or a new inventory")
    tasks = manifest["tasks"]
    if args.smoke:
        selected = {}
        for t in tasks:
            entry = manifest["files"][t["file_id"]]
            if t["kind"] == "checksum":
                continue
            label = entry["source"]
            if entry["relative_path"] == "math/math_tool.jsonl":
                label += ":tool"
            selected.setdefault(label, t)
        tasks = list(selected.values())
        args.max_rows = 8
    if args.only:
        tasks = [t for t in tasks if t["kind"] == args.only]
    if args.source:
        tasks = [t for t in tasks if manifest["files"][t["file_id"]]["source"] == args.source]
    if args.limit_tasks:
        tasks = tasks[:args.limit_tasks]
    # Greedy scheduling follows MV's independent CPU-shard pattern, without a Ray head for tokenization.
    loads, groups = [0] * args.world, [[] for _ in range(args.world)]
    for task in sorted(tasks, key=lambda t: (-t["weight"], t["task_id"])):
        i = min(range(args.world), key=lambda j: loads[j])
        groups[i].append(task)
        loads[i] += task["weight"]
    mine = groups[args.rank]
    print(json.dumps({"rank": args.rank, "world": args.world, "tasks": len(mine), "workers": args.workers}), flush=True)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=init_worker,
                             initargs=(manifest, str(out), args.max_rows)) as pool:
        futures = {pool.submit(process_task, task): task for task in mine}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            print(json.dumps({"rank": args.rank, "task_id": task["task_id"], "kind": task["kind"],
                              "source": manifest["files"][task["file_id"]]["source"],
                              "seconds": round(result.get("seconds", 0), 2),
                              "counts": result.get("counts", {}), "resumed": result.get("resumed", False)}), flush=True)


def summarize(args):
    sys.path.insert(0, args.deps)
    import duckdb
    out = Path(args.out)
    manifest = json.loads((out / "input_manifest.json").read_text())
    results = [json.loads(p.read_text()) for p in sorted((out / "tasks").glob("*.json"))]
    valid = {r["task_id"]: r for r in results if r["build_id"] == manifest["build_id"] and not r.get("limited_to_rows")}
    missing = [t["task_id"] for t in manifest["tasks"] if t["task_id"] not in valid]
    if missing and not args.allow_partial:
        raise ValueError(f"Incomplete census: {len(missing)} of {len(manifest['tasks'])} work units missing")
    connection = duckdb.connect()
    connection.execute("SET threads=16")
    connection.execute("SET memory_limit='64GB'")
    metric_files = [str(task_paths(out, t)[1]) for t in manifest["tasks"]
                    if t["kind"] != "checksum" and t["task_id"] in valid and task_paths(out, t)[1].exists()]
    if metric_files:
        connection.read_parquet(metric_files).create_view("metrics")
    else:
        connection.execute('''CREATE TABLE metrics(source VARCHAR, subset VARCHAR, kind VARCHAR,
            tokens BIGINT, trainable_tokens BIGINT, permissive_tokens BIGINT, all_nonvendor_tokens BIGINT,
            permissive_trainable_tokens BIGINT, tool_response_spans BIGINT,
            language VARCHAR, teacher VARCHAR, license VARCHAR)''')
    columns = "source,subset,kind"
    summary = connection.execute(f"""SELECT {columns}, count(*) AS "rows", sum(tokens) tokens,
        sum(trainable_tokens) trainable_tokens, sum(permissive_tokens) permissive_tokens,
        sum(all_nonvendor_tokens) all_nonvendor_tokens,
        sum(permissive_trainable_tokens) permissive_trainable_tokens,
        quantile_cont(tokens, [0.5,0.9,0.95,0.99]) length_percentiles,
        quantile_cont(permissive_tokens,[0.5,0.9,0.95,0.99]) FILTER(WHERE permissive_tokens>0) permissive_length_percentiles,
        max(tokens) max_tokens, sum(tool_response_spans) tool_response_spans
        FROM metrics GROUP BY {columns} ORDER BY {columns}""").fetchdf().to_dict("records")
    for r in summary:
        for k, v in r.items():
            if hasattr(v, "tolist"):
                r[k] = v.tolist()
    distributions = {}
    for col in ("language", "teacher", "license"):
        distributions[col] = connection.execute(f"""SELECT source,{col},count(*) AS "rows",sum(tokens) tokens,
            sum(permissive_tokens) permissive_tokens FROM metrics GROUP BY source,{col}
            ORDER BY source,tokens DESC""").fetchdf().to_dict("records")
    bins = []
    for low, high, label in TOKEN_BINS:
        for field in ("tokens", "permissive_tokens"):
            condition = f"{field}>={low}" + (f" AND {field}<{high}" if high else "")
            if field == "permissive_tokens":
                condition += f" AND {field}>0"
            rows = connection.execute(f"""SELECT source,subset,count(*) AS "rows",sum({field}) tokens
                FROM metrics WHERE {condition} GROUP BY source,subset ORDER BY source,subset""").fetchdf().to_dict("records")
            bins.extend(dict(r, length_bin=label, policy=field) for r in rows)
    counts = Counter()
    for result in valid.values():
        counts.update(result.get("counts", {}))
    hashes = {r["file_id"]: r["sha256"] for r in valid.values() if r["kind"] == "checksum"}
    frozen = dict(manifest)
    frozen["files"] = [dict(f, sha256=hashes.get(f["file_id"])) for f in manifest["files"]]
    frozen["status"] = "complete" if not missing else "partial"
    write_json(out / "verified_input_manifest.json", frozen)
    report = {"status": frozen["status"], "build_id": manifest["build_id"], "counts": dict(counts),
              "completed_tasks": len(valid), "expected_tasks": len(manifest["tasks"]), "missing_tasks": missing,
              "verified_files": len(hashes), "summary": summary, "distributions": distributions,
              "length_bins": bins, "policy": manifest["policy"]}
    write_json(out / "census_summary.json", report)
    print(json.dumps({"status": report["status"], "counts": dict(counts), "verified_files": len(hashes),
                      "completed_tasks": len(valid), "expected_tasks": len(manifest["tasks"])}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "run", "summarize"])
    ap.add_argument("--sources", default=DEFAULT_SOURCES)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out", default=DEFAULT_OUTPUT)
    ap.add_argument("--deps", default=DEFAULT_DEPS)
    ap.add_argument("--license-policy", choices=["permissive", "all"], default="permissive")
    ap.add_argument("--jsonl-chunk-mib", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", "0")))
    ap.add_argument("--world", type=int, default=int(os.environ.get("SLURM_NTASKS", "1")))
    ap.add_argument("--only", choices=["checksum", "parquet", "jsonl"])
    ap.add_argument("--source")
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="8 rows from each source and a Cascade tool trace")
    args = ap.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
