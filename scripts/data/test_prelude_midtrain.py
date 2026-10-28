"""Normalization and exact native-token fixtures; no model weights required."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from prelude_midtrain import (CASCADE, TERMINAL, STACK, ULTRA, SPECIALIZED, Reject,
                             TOKEN_IDS, adapt, normalize_messages, render, render_conversation)

TOKENIZER = "/p/data1/mmlaion/marianna/sft_sources/prelude_chat_tokenizer"


def fixture():
    return {"messages": [
        {"role": "system", "content": "Rules."},
        {"role": "user", "content": "Compute."},
        {"role": "assistant", "content": "", "reasoning_content": "First reasoning.",
         "tool_calls": [{"id": "c1", "function": {"name": "calc", "arguments": {"x": 1}}}]},
        {"role": "user", "content": "<tool_response>1</tool_response>\n<tool_response>2</tool_response>"},
        {"role": "assistant", "content": "<think>Second reasoning.</think>Answer."},
        {"role": "user", "content": "Quote: <tool_response>literal</tool_response> in a question."},
        {"role": "assistant", "content": "Done.", "reasoning_content": "Earlier turns stay."},
    ]}


class NormalizationTest(unittest.TestCase):
    def test_jsonl_byte_partitions_and_invalid_rows(self):
        from materialize_prelude_midtrain import records
        payload = '{"text":"é"}\nmalformed\n{"text":"last"}'.encode()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.jsonl";p.write_bytes(payload)
            e={"path":str(p),"state":{"bytes":len(payload)}}
            whole=list(records(e,{}))
            for split in range(1,len(payload)):
                partitioned=list(records(e,{"start":0,"end":split}))+list(records(e,{"start":split,"end":len(payload)}))
                self.assertEqual(whole,partitioned)
            self.assertEqual([r[2] for r in whole],[None,"invalid_json",None])

    def test_zstd_replay_formats_and_row_cut(self):
        sys.path.insert(0,"/p/project1/laionize/marianna/dc_agent/.prelude_census_deps")
        import zstandard
        from materialize_prelude_midtrain import records
        with tempfile.TemporaryDirectory() as d:
            for suffix in ("zst","zstd"):
                p=Path(d)/("x.jsonl."+suffix)
                p.write_bytes(zstandard.ZstdCompressor().compress(b'{"text":"a"}\n{"text":"b"}\n'))
                rows=list(records({"path":str(p),"row_limit":1},{}))
                self.assertEqual(rows,[(0,{"text":"a"},None)])

    def test_response_semantics_and_earlier_reasoning(self):
        c = Counter(); original = fixture(); saved = copy.deepcopy(original)
        m, _ = normalize_messages(original, CASCADE, c)
        self.assertEqual([x["role"] for x in m], ["system", "user", "assistant", "tool", "tool", "assistant", "user", "assistant"])
        self.assertEqual(m[2]["reasoning_content"], "First reasoning.")
        self.assertEqual(m[3]["content"], "1")
        self.assertEqual(c["quoted_response_in_user"], 1)
        self.assertEqual(c["response_wrappers_unwrapped"], 2)
        self.assertEqual(original, saved)
        original["messages"][1]["content"] = "Explain <|im_start|> literally."
        m,_ = normalize_messages(original,CASCADE,c)
        self.assertEqual(m[1]["content"],original["messages"][1]["content"])

    def test_malformed_ambiguous_and_conflicting(self):
        bad = ["<tool_response>unterminated", "<tool_response>x</tool_response> extra",
               "prefix <tool_response>x</tool_response>", "<tool_response a='b'>x</tool_response>",
               "<tool_response><tool_response>x</tool_response></tool_response>"]
        for content in bad:
            with self.subTest(content=content), self.assertRaises(Reject):
                normalize_messages({"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": "ok"}]}, CASCADE, Counter())
        with self.assertRaisesRegex(Reject, "conflicting_reasoning"):
            normalize_messages({"messages": [{"role": "assistant", "content": "<think>a</think>b", "reasoning_content": "c"}]}, CASCADE, Counter())

    def test_empty_and_tools_and_duplicate_wrappers(self):
        r = {"messages": [{"role": "developer", "content": ""}, {"role": "user", "content": ""},
             {"role": "assistant", "content": "<think>a</think>b", "reasoning_content": "<think>a</think>"},
             {"role": "tool", "content": "<tool_response></tool_response>"}, {"role": "assistant", "content": None}],
             "tools": [{"type": "function", "function": {"name": "x", "parameters": {"type": "object"}}}]}
        c = Counter(); m, tools = normalize_messages(r, CASCADE, c)
        self.assertEqual(m[2]["content"], "b")
        self.assertEqual(m[2]["reasoning_content"], "a")
        self.assertEqual(m[3]["content"], "")
        self.assertEqual(c["duplicate_reasoning_removed"], 1)
        self.assertEqual(c["tool_wrapper_removed"], 1)
        self.assertEqual(tools, r["tools"])
        r["messages"][1]["content"] = None
        with self.assertRaisesRegex(Reject, "nontext_content"):
            normalize_messages(r, CASCADE, Counter())

    def test_terminal_feedback(self):
        r = {"agent": "terminus-2", "conversations": [{"role": "user", "content": "Task"},
             {"role": "assistant", "content": "command"}, {"role": "user", "content": "Parser failed"},
             {"role": "assistant", "content": "retry"}]}
        m, _ = normalize_messages(r, TERMINAL, Counter())
        self.assertEqual([x["role"] for x in m], ["user", "assistant", "tool", "assistant"])
        r["agent"] = "unknown"
        with self.assertRaisesRegex(Reject, "unexpected_terminal_agent"):
            normalize_messages(r, TERMINAL, Counter())

    def test_source_provenance_and_code_policy(self):
        e = {"source": STACK, "path": "x.parquet", "file_id": 42, "hf_repo": "org/repo", "revision": "abc", "sha256": "123", "relative_path": "x.parquet"}
        files = [{"content_id": "2", "file_path": "b.py", "content": "  B\n", "is_vendor": False, "license_type": "no_license"},
                 {"content_id": "1", "file_path": "a.py", "content": "\tA\n", "is_vendor": False, "license_type": "permissive"},
                 {"content_id": "3", "file_path": "v.py", "content": "vendor", "is_vendor": True, "license_type": "permissive"}]
        r = {"repo_path": "Owner/Repo", "commit_id": "xyz", "files": files}
        c = Counter(); row = adapt(r, e, 3, c, selected_repo={"mix_category": "code"})
        self.assertEqual(row["text"], "Owner/Repo\n\nFile: a.py\n\tA\n\n\nFile: b.py\n  B\n")
        self.assertEqual(len(row["provenance"]["files"]), 2)
        self.assertEqual(row["provenance"]["metadata"]["commit_id"], "xyz")
        self.assertEqual(c["eligible_license_no_license"], 1)
        self.assertEqual(c["excluded_vendor_file"], 1)
        self.assertEqual(row["id"], adapt(r, {**e, "file_id": 999}, 3, selected_repo={"mix_category": "code"})["id"])
        self.assertNotEqual(row["id"], adapt(r, e, 4, selected_repo={"mix_category": "code"})["id"])
        with self.assertRaisesRegex(Reject, "repo_not_selected"): adapt(r, e, 3)
        doc = adapt({"text": " \ttext\n  raw", "metadata": {"models_used": "teacher"}}, {**e, "source": SPECIALIZED}, 2)
        self.assertEqual(doc["text"], " \ttext\n  raw")
        ultra = adapt({"content": "  x\n", "relative_path": "x", "source_meta": {"file_meta": files[0]}}, {**e, "source": ULTRA}, 2)
        self.assertEqual(ultra["text"], "  x\n")


class TokenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)

    def test_exact_ids_and_masks(self):
        m, tools = normalize_messages(fixture(), CASCADE, Counter())
        ids, loss = render_conversation(self.tok, m, tools)
        # Independently written expected template output and semantic tool block.
        block = '<|im_start|>user\n<tool_response>\n1\n</tool_response>\n<tool_response>\n2\n</tool_response><|im_end|>\n'
        prefix = ('<bos><|im_start|>system\nRules.<|im_end|>\n<|im_start|>user\nCompute.<|im_end|>\n'
                  '<|im_start|>assistant\n<think>\nFirst reasoning.\n</think>\n\n'
                  '<tool_call>\n{"name": "calc", "arguments": {"x": 1}}\n</tool_call><|im_end|>\n')
        suffix = ('<|im_start|>assistant\n<think>Second reasoning.</think>Answer.<|im_end|>\n'
                  '<|im_start|>user\nQuote: <tool_response>literal</tool_response> in a question.<|im_end|>\n'
                  '<|im_start|>assistant\n<think>\nEarlier turns stay.\n</think>\n\nDone.<|im_end|>\n')
        expected = self.tok(prefix+block+suffix, add_special_tokens=False)["input_ids"]
        self.assertEqual(ids, expected)
        n = len(self.tok(prefix, add_special_tokens=False)["input_ids"])
        b = len(self.tok(block, add_special_tokens=False)["input_ids"])
        self.assertEqual(loss, [0]+[1]*(n-1)+[0]*b+[1]*(len(ids)-n-b))
        self.assertEqual([ids[i] for i in range(len(ids)) if loss[i] and ids[i] == 262149], [262149])

    def test_document_and_length_boundary(self):
        row = {"id": "doc", "kind": "document", "text": "\tdef f():\n    return 1\n", "mix_source": "code"}
        r = render(row, self.tok)
        expected = [1]+self.tok(row["text"], add_special_tokens=False)["input_ids"]+[2]
        self.assertEqual(r["ids"], expected)
        self.assertEqual(r["loss"], [0]+[1]*(len(expected)-1))
        self.assertEqual(render(row, self.tok, len(expected)), r)
        with self.assertRaisesRegex(Reject, "overlength"): render(row, self.tok, len(expected)-1)
        # LF is a native, unmergeable token. Exact real 64K boundary.
        row["text"] = "\n" * (65536-2)
        self.assertEqual(len(render(row, self.tok)["ids"]), 65536)
        row["text"] += "\n"
        with self.assertRaisesRegex(Reject, "overlength"): render(row, self.tok)

    def test_resume_checks_output_and_configuration(self):
        from materialize_prelude_midtrain import process
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"source.jsonl";p.write_text('{"text":"example"}\n')
            s=p.stat()
            e={"source":SPECIALIZED,"namespace":"core","file_id":1,"subset":"fixture",
               "path":str(p),"sha256":"fixture","state":{"bytes":s.st_size,"mtime_ns":s.st_mtime_ns}}
            m={"build_id":"fixture","max_render_tokens":65536}
            task={"task_key":"fixture"}
            first=process(d,m,e,task,self.tok,{})
            self.assertEqual(first,process(d,m,e,task,self.tok,{}))
            with self.assertRaisesRegex(ValueError,"resume configuration"):
                process(d,m,e,task,self.tok,{},wanted={0})
            (Path(d)/"normalized_build/fixture/rendered.parquet").write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,"completed output changed"):
                process(d,m,e,task,self.tok,{})

    def test_default_sft_unchanged(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"sft_gen"))
        from prerender_levanter import render_row
        from materialize_prelude_sft_v1 import norm_messages
        raw = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "first", "reasoning_content": "old"},
               {"role": "user", "content": "two"}, {"role": "assistant", "content": "last", "reasoning_content": "new"}]
        sft = norm_messages(raw)
        self.assertIsNone(sft[1]["reasoning_content"])
        ids, default = render_row(self.tok, sft, {})
        self.assertEqual((ids, default), render_row(self.tok, sft, {}, mask_policy="assistant-only"))
        full = self.tok.decode(ids)
        expected = [0]*len(ids)
        for i in (1,3):
            pre = self.tok.apply_chat_template([{k:v for k,v in m.items() if k in ("role","content","reasoning_content") and v is not None} for m in sft[:i]], tokenize=False, add_generation_prompt=True)
            end = self.tok.apply_chat_template([{k:v for k,v in m.items() if k in ("role","content","reasoning_content") and v is not None} for m in sft[:i+1]], tokenize=False)
            a = len(self.tok(pre, add_special_tokens=False)["input_ids"])
            b = len(self.tok(end, add_special_tokens=False)["input_ids"])
            expected[a:b] = [1]*(b-a)
        self.assertEqual(default, expected)
        normalized, _ = normalize_messages({"messages": raw}, CASCADE, Counter())
        mid_ids, mid_loss = render_row(self.tok, normalized, {}, mask_policy="midtrain")
        self.assertIn("old", self.tok.decode(mid_ids))
        self.assertGreater(sum(mid_loss), sum(default))


if __name__ == "__main__": unittest.main()
