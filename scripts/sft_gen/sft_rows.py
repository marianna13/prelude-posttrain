"""Row writer shared by export_sft.py and agenttrove_convert.py.

Parquet (default) is the SkyRL SFT format used by the Grug mix (rl/scripts/data/build_sft_mix_5k.py, issue 0048):
  conversations  list<struct<role: string, content: string, reasoning_content: string, train_loss: int8>>
                 (messages_key=conversations; reasoning_content null unless --think-to-reasoning)
                 train_loss 1 = train on this assistant turn; 0 = context only (all user turns, rejected
                 or injected malformed assistant turns)
  kind           sample kind (main, post_handoff, summary, questions, answers, agenttrove, agenttrove_handoff)
  original_source, original_teacher, trial_name   provenance (strings, may be null)
  meta_json      every metadata field as JSON
A path ending in .jsonl writes the older JSONL format instead ({"kind", "messages": [{role, content,
weight}], "meta"}).
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

MESSAGE = pa.struct([("role", pa.string()), ("content", pa.string()), ("reasoning_content", pa.string()),
                     ("train_loss", pa.int8())])
SCHEMA = pa.schema([
    ("conversations", pa.list_(MESSAGE)),
    ("kind", pa.string()),
    ("original_source", pa.string()),
    ("original_teacher", pa.string()),
    ("trial_name", pa.string()),
    ("meta_json", pa.string()),
])


class SftWriter:
    def __init__(self, path: str, batch_rows: int = 2000):
        self.path, self.batch_rows, self.rows, self.n = path, batch_rows, [], 0
        self.jsonl = path.endswith(".jsonl")
        self._fh = open(path, "w") if self.jsonl else None
        self._pq = None if self.jsonl else pq.ParquetWriter(path, SCHEMA, compression="zstd")

    def write(self, kind: str, messages: list[dict], meta: dict) -> None:
        """messages: [{role, content, weight}] (weight 0/1 on assistant turns)."""
        self.n += 1
        if self.jsonl:
            self._fh.write(json.dumps({"kind": kind, "messages": messages, "meta": meta}) + "\n")
            return
        conv = [{"role": m["role"], "content": m["content"], "reasoning_content": m.get("reasoning_content"),
                 "train_loss": int(m.get("weight", 0)) if m["role"] == "assistant" else 0} for m in messages]

        def s(*keys):
            for k in keys:
                if meta.get(k) is not None:
                    return str(meta[k])
            return None

        self.rows.append({
            "conversations": conv, "kind": kind,
            "original_source": s("original_source", "source"),
            "original_teacher": s("original_teacher", "teacher"),
            "trial_name": s("trial_name", "trial"),
            "meta_json": json.dumps(meta, default=str),
        })
        if len(self.rows) >= self.batch_rows:
            self._flush()

    def _flush(self) -> None:
        if self.rows:
            self._pq.write_table(pa.Table.from_pylist(self.rows, schema=SCHEMA))
            self.rows = []

    def __enter__(self) -> "SftWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self.jsonl:
            self._fh.close()
        else:
            self._flush()
            self._pq.close()
