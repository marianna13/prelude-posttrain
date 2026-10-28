"""Boundary and semantic regression checks for the bulk census, using tiny inputs."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("census", Path(__file__).with_name("prelude_midtrain_census.py"))
census = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(census)


class CensusTests(unittest.TestCase):
    def test_jsonl_chunks_visit_each_record_once(self):
        rows = [{"text": "α" * i} for i in range(1, 25)]
        raw = b"\n".join(json.dumps(r, ensure_ascii=False).encode() for r in rows)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_bytes(raw)  # deliberately no trailing newline
            visited = []
            for start in range(0, len(raw), 37):
                visited.extend(census.iter_records({"path": str(p)},
                               {"kind": "jsonl", "start": start, "end": min(start + 37, len(raw))}))
            self.assertEqual([r for _, r, _ in visited], rows)
            self.assertEqual(len({offset for offset, _, _ in visited}), len(rows))

    def test_malformed_json_is_countable_and_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_bytes(b'{"text":"ok"}\nINVALID\n{"text":"after"}\n')
            rows = list(census.iter_records({"path": str(p)}, {"kind": "jsonl", "start": 0, "end": p.stat().st_size}))
            self.assertEqual(rows[1][2], "invalid_json")
            self.assertEqual(rows[2][1]["text"], "after")

    def test_repo_selection_keeps_exact_code_and_path_order(self):
        files = [{"file_path": "b.py", "content": "x = 'A'\n\treturn x\n", "content_id": "b",
                  "license_type": "permissive", "is_vendor": False},
                 {"file_path": "a.py", "content": "private = 1\n", "content_id": "a",
                  "license_type": "no_license", "is_vendor": False},
                 {"file_path": "vendor.py", "content": "vendor = 1\n", "content_id": "c",
                  "license_type": "permissive", "is_vendor": True}]
        record = {"repo_path": "Owner/Repo", "files": files, "commit_id": "c"}
        base, raw, eligible, _, side = census.record_view(record, {"source": "stack-v3-train", "file_id": 3, "subset": "data"}, 2)
        self.assertLess(raw.index("File: a.py"), raw.index("File: b.py"))
        self.assertIn("x = 'A'\n\treturn x\n", eligible)
        self.assertNotIn("private =", eligible)
        self.assertNotIn("vendor =", eligible)
        self.assertEqual(base["eligible_files"], 1)
        self.assertEqual(len(side), 3)

    def test_only_complete_response_messages_are_masked(self):
        # System instructions may describe tags; a real user may ask about tags.
        rendered = ("<bos><|im_start|>system\nExample: <tool_response>x</tool_response><|im_end|>\n"
                    "<|im_start|>user\nExplain <tool_response>x</tool_response> syntax<|im_end|>\n"
                    "<|im_start|>user\n<tool_response>actual output</tool_response>\n<|im_end|>\n"
                    "<|im_start|>assistant\n<think>keep this</think>answer<|im_end|>\n")
        spans = []
        for block in census.BLOCK.finditer(rendered):
            if block.group(1) == "user" and census.RESPONSE_ONLY.fullmatch(block.group(2)):
                spans.extend(rendered[block.start(2) + r.start():block.start(2) + r.end()]
                             for r in census.RESPONSE.finditer(block.group(2)))
        self.assertEqual(spans, ["<tool_response>actual output</tool_response>"])

    def test_input_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data"
            p.write_text("first")
            frozen = {"path": str(p), "state": census.file_state(p)}
            p.write_text("changed data")
            with self.assertRaisesRegex(RuntimeError, "Frozen input changed"):
                census.check_file(frozen)

    def test_terminal_feedback_is_an_environment_response(self):
        captured = []
        class Template:
            def apply_chat_template(self, messages, **kwargs):
                captured.extend(messages)
                return "<bos>"
        record = {"agent": "terminus-2", "conversations": [
            {"role": "user", "content": "Task: solve this"},
            {"role": "assistant", "content": "<think>reasoning</think>command"},
            {"role": "user", "content": "New Terminal Output:\nresult"},
            {"role": "assistant", "content": "corrected command"},
            {"role": "user", "content": "Previous response had parsing errors:\nfix"}]}
        with patch.object(census, "_HF", Template()), patch.object(census, "language", return_value="en"):
            census.conversation(record, "Nemotron-Terminal-Corpus")
        self.assertEqual([m["role"] for m in captured], ["user", "assistant", "tool", "assistant", "tool"])
        self.assertEqual(record["conversations"][2]["role"], "user")

    def test_summary_sql_and_license_filtered_percentiles(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "metrics").mkdir()
            (out / "tasks").mkdir()
            rows = [dict(source="fixture", subset="train", kind="document", tokens=n,
                         trainable_tokens=n-1, permissive_tokens=p, all_nonvendor_tokens=n,
                         permissive_trainable_tokens=max(0,p-1), tool_response_spans=0,
                         language="en", teacher="unknown", license="mixed")
                    for n,p in [(65536,0),(65537,4)]]
            pq.write_table(pa.Table.from_pylist(rows), out / "metrics/000000.parquet")
            census.write_json(out / "input_manifest.json", dict(build_id="fixture", policy={}, files=[],
                               tasks=[dict(task_id=0, kind="parquet")]))
            census.write_json(out / "tasks/000000.json", dict(build_id="fixture", task_id=0, kind="parquet",
                               counts={"rows_in":2,"valid_rows":2}, limited_to_rows=0))
            census.summarize(SimpleNamespace(out=str(out), deps=census.DEFAULT_DEPS, allow_partial=False))
            report = json.loads((out / "census_summary.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["summary"][0]["rows"], 2)
            self.assertEqual(report["summary"][0]["permissive_length_percentiles"], [4,4,4,4])
            bins = {(b["length_bin"],b["policy"]):b["rows"] for b in report["length_bins"]}
            self.assertEqual(bins[("32-64K","tokens")],1)
            self.assertEqual(bins[(">64K","tokens")],1)


if __name__ == "__main__":
    unittest.main()
