"""Pre-render an SFT parquet (conversations with per-message train_loss) into Levanter's ids/loss rows.

usage:
  python3 prerender_levanter.py IN.parquet OUT.parquet --tokenizer GRUG_SNAPSHOT [--enable-thinking false]
      [--max-tokens 32768]

Input: the mix format of build_sft_mix_5k.py / sft_rows.py: ``conversations`` = list<{role, content, train_loss}>,
optionally ``reasoning_content`` per message (thinkify_mix.py; rendered inside the think tokens and trained with the turn),
``tool_calls`` per message (JSON list of {name, arguments}) and a row-level ``tools_json`` (OpenAI tool list), both
passed to the chat template (materialize_prelude_sft_v1.py).
Output: one row per conversation, ``id`` (string), ``ids`` (list<int32>), ``loss`` (list<int8>), ``mix_source``
-- what ``SnowballPrerenderedChatFormat`` (rl/sft/levanter/scripts_grug/grug_datakit_chat.py) copies into the cache
unchanged.

Each row is rendered once with the model's own chat template (``tokenizer.apply_chat_template``; for Grug 09.21
``enable_thinking=false`` puts the ``Reasoning: /nothink`` header in the system turn). An assistant message with
train_loss 1 trains the characters between the generation prompt rendered before it and the end of its own
rendering: its content and the closing ``<|eot_id|>``, the span vLLM samples at serving. Headers, user / tool
turns and train_loss-0 assistant turns stay in context with loss 0. The template must render prefix-consistently
(every prefix of a conversation renders to a prefix of the whole); a row that does not, or whose span boundaries
fall inside a token, is refused rather than masked approximately.
"""

import argparse
import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SCHEMA = pa.schema([("id", pa.string()), ("ids", pa.list_(pa.int32())), ("loss", pa.list_(pa.int8())),
                    ("mix_source", pa.string())])


def render_row(tok, messages: list[dict], template_kwargs: dict, tools: list | None = None,
               *, mask_policy: str = "assistant-only") -> tuple[list[int], list[int]]:
    if mask_policy == "midtrain":
        # Separate normalization is mandatory: semantic user-role tool responses
        # must already have become tool messages, and earlier reasoning retained.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
        from prelude_midtrain import render_conversation
        return render_conversation(tok, messages, tools)
    if mask_policy != "assistant-only":
        raise ValueError(f"unknown mask policy: {mask_policy}")
    plain = [
        {"role": m["role"], "content": m.get("content") or ""}
        | ({"reasoning_content": m["reasoning_content"]} if m.get("reasoning_content") else {})
        | ({"tool_calls": json.loads(m["tool_calls"])} if m.get("tool_calls") else {})
        for m in messages
    ]
    if tools:
        template_kwargs = dict(template_kwargs, tools=tools)
    full = tok.apply_chat_template(plain, tokenize=False, **template_kwargs)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant" or m.get("train_loss") == 0:
            continue
        pre = tok.apply_chat_template(plain[:i], tokenize=False, add_generation_prompt=True, **template_kwargs)
        upto = tok.apply_chat_template(plain[: i + 1], tokenize=False, **template_kwargs)
        if not (upto.startswith(pre) and full.startswith(upto)):
            raise ValueError(f"template is not prefix-consistent at message {i}")
        spans.append((len(pre), len(upto)))
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    starts = {s for s, _ in offsets}
    ends = {e for _, e in offsets}
    loss = [0] * len(ids)
    for a, b in spans:
        if a not in starts or b not in ends:
            raise ValueError(f"trained span [{a}, {b}) does not fall on token boundaries")
        for t, (s, e) in enumerate(offsets):
            if s >= a and e <= b and e > s:
                loss[t] = 1
    return ids, loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--enable-thinking", choices=("true", "false", "unset"), default="false")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--mask-policy", choices=("assistant-only", "midtrain"), default="assistant-only",
                    help="midtrain reads canonical rows from materialize_prelude_midtrain.py")
    ap.add_argument("--skip-unrenderable", action="store_true",
                    help="count and skip rows that are not prefix-consistent / token-aligned instead of failing")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    kwargs = {} if a.enable_thinking == "unset" else {"enable_thinking": a.enable_thinking == "true"}
    stats, trained_by = Counter(), Counter()
    writer = pq.ParquetWriter(a.out, SCHEMA, compression="zstd")
    f = pq.ParquetFile(a.inp)
    fields = ("id", "kind", "text", "messages_json", "tools_json", "policy", "mix_source") if a.mask_policy == "midtrain" else (
        "conversations", "mix_source", "trial_name", "tools_json")
    cols = [c for c in fields if c in f.schema_arrow.names]
    n = 0
    for g in range(f.num_row_groups):
        out = []
        for r in f.read_row_group(g, columns=cols).to_pylist():
            n += 1
            try:
                if a.mask_policy == "midtrain":
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
                    from prelude_midtrain import POLICY, render
                    if r.get("policy") != POLICY:
                        raise ValueError("midtrain requires a canonical mid-training row")
                    rendered = render({**r, "messages": json.loads(r["messages_json"]) if r.get("messages_json") else None,
                                       "tools": json.loads(r["tools_json"]) if r.get("tools_json") else None}, tok, a.max_tokens)
                    ids, loss = rendered["ids"], rendered["loss"]
                else:
                    ids, loss = render_row(tok, r["conversations"], kwargs,
                                           json.loads(r["tools_json"]) if r.get("tools_json") else None)
            except ValueError as e:
                if not a.skip_unrenderable:
                    raise
                stats[f"skipped_unrenderable: {str(e)[:60]}"] += 1
                continue
            if ids[0] != tok.bos_token_id:
                raise ValueError(f"row {n}: rendering does not start with BOS")
            if len(ids) > a.max_tokens:
                stats["skipped_too_long"] += 1
                continue
            if not any(loss):
                stats["skipped_no_trained_token"] += 1
                continue
            src = r.get("mix_source") or ""
            stats["rows"] += 1
            stats["tokens"] += len(ids)
            trained_by[src] += sum(loss)
            out.append({"id": r["id"] if a.mask_policy == "midtrain" else f"{src}-{n}",
                        "ids": ids, "loss": loss, "mix_source": src})
        if out:
            writer.write_table(pa.Table.from_pylist(out, schema=SCHEMA))
    writer.close()
    report = dict(stats, trained_tokens=sum(trained_by.values()), trained_tokens_by_source=dict(trained_by),
                  enable_thinking=a.enable_thinking, tokenizer=a.tokenizer, input=a.inp, mask_policy=a.mask_policy)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
