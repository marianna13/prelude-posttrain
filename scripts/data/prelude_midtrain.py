"""Prelude mid-training v1 adapters and target-token masks; SFT normalization is separate.

The canonical row contains raw document text OR normalized messages, plus immutable
source provenance. No deduplication, decontamination, truncation or sampling occurs here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

CASCADE = "Nemotron-Cascade-2-SFT-Data"
TERMINAL = "Nemotron-Terminal-Corpus"
SPECIALIZED = "Nemotron-Pretraining-Specialized-v1"
ULTRA = "UltraData-Stack-v3-L2"
STACK = "stack-v3-train"
POLICY = "prelude_midtrain_v1"
TOKEN_IDS = {"<bos>": 1, "<eos>": 2, "<pad>": 262144,
             "<|im_start|>": 262145, "<|im_end|>": 262146,
             "<think>": 262147, "</think>": 262148,
             "<tool_response>": 262149, "</tool_response>": 262150,
             "<tool_call>": 11, "</tool_call>": 12}
BOUNDARY = re.compile(r"<\|im_start\|>|<\|im_end\|>|<bos>|<eos>|<pad>")
RESPONSE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
THINK = re.compile(r"\A\s*<think>(.*?)</think>(.*)\Z", re.S)


class Reject(ValueError):
    """A stable, counted reason; never include source payloads in error messages."""


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_json(value, kind):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            raise Reject("invalid_json_" + kind) from None
    return value


def text(value, kind, *, allow_none=False):
    if value is None and allow_none:
        return ""
    if not isinstance(value, str):
        raise Reject("nontext_" + kind)
    return value


def response_parts(content):
    """Only whole boundary-wrapped responses are semantic tool output.

    Embedded quotations stay user text. A wrapper touching either message edge
    without a complete wrapper-only parse is ambiguous and is rejected.
    """
    s = content.strip()
    if not (s.startswith("<tool_response") or s.endswith("</tool_response>")):
        return None
    parts, end = [], 0
    for m in RESPONSE.finditer(s):
        if s[end:m.start()].strip() or "<tool_response>" in m[1] or "</tool_response>" in m[1]:
            raise Reject("ambiguous_tool_response")
        parts.append(m[1])
        end = m.end()
    if not parts or s[end:].strip():
        raise Reject("malformed_tool_response")
    return parts


def normalize_calls(raw, stats):
    calls = parse_json(raw, "tool_calls")
    if calls is None or calls == []:
        return None
    if not isinstance(calls, list):
        raise Reject("tool_calls_not_list")
    out = []
    for call in calls:
        if not isinstance(call, dict):
            raise Reject("malformed_tool_call")
        f = call.get("function", call)
        if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"]:
            raise Reject("malformed_tool_call")
        args = parse_json(f.get("arguments"), "tool_arguments")
        if not isinstance(args, dict):
            raise Reject("tool_arguments_not_object")
        # Retain call IDs and other source fields for provenance; the frozen
        # template renders name/arguments as it does for existing SFT.
        out.append({**call, "function": {**f, "arguments": args}})
        stats["structured_tool_calls"] += 1
    return out


def normalize_messages(record, source, stats):
    raw = record.get("messages", record.get("conversations"))
    if not isinstance(raw, list) or not raw:
        raise Reject("missing_messages")
    if source == TERMINAL and record.get("agent") != "terminus-2":
        raise Reject("unexpected_terminal_agent")
    out, seen_task = [], False
    for message in raw:
        if not isinstance(message, dict):
            raise Reject("message_not_object")
        m = {k: v for k, v in message.items() if v is not None}
        if "train_loss" in m:
            stats["source_loss_metadata_ignored"] += 1
        role = m.get("role")
        if role == "developer":
            role = "system"
            stats["developer_to_system"] += 1
        if role not in ("system", "user", "assistant", "tool"):
            raise Reject("unknown_role")
        content = text(message.get("content"), "content", allow_none=role in ("assistant", "tool"))
        if message.get("content") is None:
            stats["null_content_to_empty"] += 1
        if not content:
            stats["empty_" + role] += 1
        if BOUNDARY.search(content):
            stats["literal_boundary_in_content_preserved"] += 1
        if role != "assistant" and (message.get("reasoning_content") or message.get("tool_calls")):
            raise Reject("assistant_fields_on_nonassistant")
        if source == TERMINAL and role == "user":
            if seen_task:
                role = "tool"
                stats["terminal_feedback_to_tool"] += 1
            else:
                seen_task = True
                stats["terminal_initial_task"] += 1
        if role in ("user", "tool"):
            parts = response_parts(content)
            if parts is not None:
                stats["user_wrapped_response_to_tool" if role == "user" else "tool_wrapper_removed"] += 1
                for part in parts:
                    out.append({**m, "role": "tool", "content": part})
                stats["response_wrappers_unwrapped"] += len(parts)
                continue
            if role == "tool":
                # A malformed embedded wrapper in a tool payload would be nested
                # by the template. Reject rather than invent a repair.
                if "<tool_response>" in content or "</tool_response>" in content:
                    raise Reject("ambiguous_tool_payload")
                stats["tool_role_response"] += 1
            else:
                stats["genuine_user_turn"] += 1
                if "<tool_response>" in content or "</tool_response>" in content:
                    stats["quoted_response_in_user"] += 1
        if role == "assistant":
            reasoning = text(message.get("reasoning_content"), "reasoning", allow_none=True)
            if BOUNDARY.search(reasoning):
                stats["literal_boundary_in_reasoning_preserved"] += 1
            if reasoning:
                wrapped = THINK.fullmatch(reasoning)
                if wrapped:
                    if wrapped[2].strip() or "<think>" in wrapped[1] or "</think>" in wrapped[1]:
                        raise Reject("malformed_reasoning_wrapper")
                    reasoning = wrapped[1]
                    stats["reasoning_wrapper_removed"] += 1
                elif "<think>" in reasoning or "</think>" in reasoning:
                    raise Reject("malformed_reasoning_wrapper")
                inline = THINK.fullmatch(content)
                if inline:
                    if inline[1].strip() != reasoning.strip():
                        raise Reject("conflicting_reasoning_fields")
                    content = inline[2]
                    stats["duplicate_reasoning_removed"] += 1
                elif content.lstrip().startswith("<think>"):
                    raise Reject("malformed_inline_reasoning")
                m["reasoning_content"] = reasoning
                stats["separate_reasoning_preserved"] += 1
                if reasoning != reasoning.strip():
                    stats["reasoning_template_edge_trim"] += 1
            elif "reasoning_content" in m:
                del m["reasoning_content"]
            if "<tool_call>" in content:
                stats["inline_tool_calls_preserved"] += 1
            if content.lstrip().startswith("<think>"):
                inline = THINK.fullmatch(content)
                if inline is None or "<think>" in inline[1] or "</think>" in inline[1]:
                    raise Reject("malformed_inline_reasoning")
                stats["inline_reasoning_preserved"] += 1
            calls = normalize_calls(message.get("tool_calls"), stats)
            if calls:
                if "<tool_call>" in content or "</tool_call>" in content:
                    raise Reject("duplicate_tool_call_representations")
                m["tool_calls"] = calls
            else:
                m.pop("tool_calls", None)
        # Input SFT train_turns/train_loss never restrict mid-training turns.
        m.pop("train_loss", None)
        out.append({**m, "role": role, "content": content})
    if not any(m["role"] == "assistant" for m in out):
        raise Reject("no_assistant")
    seen_assistant = False
    for m in out:
        if m["role"] == "assistant": seen_assistant = True
        if m["role"] == "tool" and not seen_assistant:
            raise Reject("orphan_tool_response")
    tools = parse_json(record.get("tools", record.get("tools_json")), "tools")
    if tools is not None:
        if not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools):
            raise Reject("malformed_tools")
        stats["tool_definitions_preserved"] += len(tools)
    return out, tools


def eligible_file(file, stats):
    if file.get("is_vendor") is True:
        stats["excluded_vendor_file"] += 1
        return False
    if not isinstance(file.get("is_vendor"), bool):
        raise Reject("missing_vendor_flag")
    license_type = file.get("license_type")
    if license_type not in ("permissive", "no_license"):
        stats["excluded_license_file"] += 1
        return False
    stats["eligible_license_" + license_type] += 1
    return True


def adapt(record, entry, locator, stats=None, *, selected_repo=None):
    """locator is a Parquet/global decoded row index or JSONL byte offset.

    ID depends on source bytes and location, never on filtering, task partitioning
    or normalization version. Retain raw-record hash and all non-payload metadata.
    """
    stats = stats if stats is not None else Counter()
    stats["input_rows"] += 1
    if not isinstance(record, dict):
        raise Reject("record_not_object")
    source = entry["source"]
    provenance = {k: entry.get(k) for k in ("source", "hf_repo", "revision", "relative_path", "sha256", "family", "language")}
    provenance.update(locator=locator, locator_kind="byte_offset" if entry["path"].endswith(".jsonl") else "row_index")
    stable = hashlib.sha256(dump(provenance).encode()).hexdigest()
    metadata = {k: v for k, v in record.items() if k not in ("text", "content", "messages", "conversations", "files")}
    provenance.update(record_sha256=hashlib.sha256(dump(record).encode()).hexdigest(), metadata=metadata)
    row = {"id": "prelude:" + stable, "policy": POLICY, "mix_source": source + "/" + entry.get("subset", entry.get("family", "")),
           "provenance": provenance, "kind": "document", "text": None, "messages": None, "tools": None}
    if source in (CASCADE, TERMINAL):
        row["messages"], row["tools"] = normalize_messages(record, source, stats)
        row["kind"] = "conversation"
    elif source == STACK:
        if selected_repo is None:
            raise Reject("repo_not_selected")
        files = record.get("files")
        if not isinstance(files, list) or not files:
            raise Reject("missing_repo_files")
        keep = [f for f in files if eligible_file(f, stats)]
        if not keep:
            raise Reject("no_eligible_repo_files")
        for f in keep:
            text(f.get("content"), "repo_content")
            text(f.get("file_path"), "repo_path")
            if not f.get("content_id"):
                raise Reject("missing_content_id")
        keep.sort(key=lambda f: (f["file_path"], f["content_id"]))
        name = text(record.get("repo_path"), "repo_name")
        row["text"] = name + "\n\n" + "\n\n".join("File: " + f["file_path"] + "\n" + f["content"] for f in keep)
        provenance["files"] = [{k: v for k, v in f.items() if k != "content"} for f in keep]
        provenance["repo_selection"] = selected_repo
        row["mix_source"] += "/" + selected_repo["mix_category"]
    elif source == ULTRA:
        fm = (record.get("source_meta") or {}).get("file_meta") or {}
        if not eligible_file(fm, stats):
            raise Reject("ineligible_code_file")
        if not fm.get("content_id") or not record.get("relative_path"):
            raise Reject("missing_file_provenance")
        row["text"] = text(record.get("content"), "code")
    else:
        if source != SPECIALIZED and not entry.get("family"):
            raise Reject("unknown_document_source")
        row["text"] = text(record.get(entry.get("text_field", "text")), "document")
    if row["kind"] == "document":
        if not row["text"].strip():
            raise Reject("empty_document")
        stats["raw_document_preserved"] += 1
    stats["normalized_rows"] += 1
    return row


def render_conversation(tok, messages, tools=None):
    """Loss 1 on every ordinary target, 0 on BOS and entire tool message groups.

    Tool groups include ChatML header, response delimiters, payload, end delimiter
    and final LF. Prefix rendering is used only at complete tool-group boundaries;
    consecutive tools share a user header in the frozen Prelude template.
    """
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    if tools:
        kwargs["tools"] = tools
    full = tok.apply_chat_template(messages, **kwargs)
    spans, i = [], 0
    while i < len(messages):
        if messages[i]["role"] != "tool":
            i += 1
            continue
        j = i + 1
        while j < len(messages) and messages[j]["role"] == "tool":
            j += 1
        before = tok.apply_chat_template(messages[:i], **kwargs)
        upto = tok.apply_chat_template(messages[:j], **kwargs)
        if not full.startswith(before) or not full.startswith(upto):
            raise Reject("nonprefix_tool_group")
        spans.append((len(before), len(upto)))
        i = j
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    loss = [1] * len(ids)
    for a, b in spans:
        if a not in {s for s, e in offsets if e > s} or b not in {e for s, e in offsets if e > s}:
            raise Reject("unaligned_tool_group")
        for k, (s, e) in enumerate(offsets):
            if s < b and e > a:
                if s < a or e > b:
                    raise Reject("token_crosses_mask_boundary")
                loss[k] = 0
    if not ids or ids[0] != tok.bos_token_id:
        raise Reject("missing_bos")
    loss[0] = 0
    return ids, loss


def render(row, tok, max_tokens=65536):
    if row["kind"] == "conversation":
        ids, loss = render_conversation(tok, row["messages"], row.get("tools"))
    else:
        # Deliberately use native document EOS 2, not configured ChatML EOS.
        # Use actual native IDs at proven LF barriers, not the census length-only
        # proxy. Stop allocating tokens once overlength is certain.
        from pathlib import Path
        from prelude_lf_counting import chunks, verify_barrier
        if not getattr(tok, "_prelude_lf_verified", False):
            verify_barrier(tok.backend_tokenizer, Path(tok.name_or_path)/"tokenizer.json")
            tok._prelude_lf_verified = True
        ids = [1]
        for piece in chunks(row["text"]):
            if len(piece) > 8 * 1024**2:
                raise Reject("unbounded_lf_free_span")
            ids.extend(tok.backend_tokenizer.encode(piece, add_special_tokens=False).ids)
            if len(ids) + 1 > max_tokens:
                raise Reject("overlength")
        ids.append(2)
        loss = [0] + [1] * (len(ids) - 1)
    if len(ids) > max_tokens:
        # A later length curriculum/chunking stage must make an explicit decision.
        # Never silently slice here (ChatDataset's default left slice would).
        raise Reject("overlength")
    if not any(loss):
        raise Reject("no_target_tokens")
    return {"id": row["id"], "ids": ids, "loss": loss, "mix_source": row["mix_source"]}
