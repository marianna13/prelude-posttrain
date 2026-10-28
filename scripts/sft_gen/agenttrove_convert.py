"""Convert AgentTrove rows to the weighted sample format of export_sft.py (inputs for rl/scripts/data/synthesize_structural_recovery.py).

usage:
  python3 agenttrove_convert.py "SHARD_GLOB" OUT.parquet --subset clean|handoff [--cap-per-source 20000]
      [--teachers T1,T2|all (clean default: strong teachers; handoff default: GLM-4.7)] [--no-strip-think | --think-to-reasoning] [--max-shards N] [--seed 0] [--harbor-src PATH]

Source: open-thoughts/AgentTrove @ b395a430 (rl/inventory/datasets/agenttrove-sft-traces-v1.yaml); JURECA copy
in /p/scratch/synthlaion/dc-agent-shared/agenttrove/data. Rows are terminus-2 conversations
(`conversations` column). AgentTrove has one row PER EPISODE of a trial (each a prefix of the same
trajectory), so we keep only the longest episode per (trial_name, run_id): keeping every episode, as the
Jupiter "first row per (trial, run, episode)" cut does, trains early turns many times over.

subset clean    the inventory's clean_completion policy: drop exp_* and unknown sources; cap per source;
                last turn is a non-empty assistant turn whose JSON is brace-balanced and that declares
                task_complete (or result/judgment is a success); result is not an error or timeout.
subset handoff  rows of a context-summarization handoff (any source, exp_tas included; teachers default
                GLM-4.7). AgentTrove stores one handoff as separate rows, classified by their user turns:
                  summary       old agent's context -> summary request -> prose SUMMARY
                  answers       same context + the new agent's questions -> ANSWERS
                  continuation  new agent: task + summary -> QUESTIONS -> answers -> JSON turns after the handoff
                  (questions    2-message row contained in continuation: skipped)
                One row per (trial, kind), longest episode; at most --trials-per-task trials per task.
                In summary/answers rows only the subagent reply is trained (their earlier agent turns repeat
                the trajectory); continuation rows train the questions and every valid turn after them.

Weights (same rules as export_sft.py): an assistant turn answering a handoff prompt (summary request, the
new agent's questions prompt, the answers request) is a subagent reply: prose by design, trained. Any
other assistant turn is trained only if the ORIGINAL text parses as terminus-2 JSON (what the harness
saw); rejected turns keep their text with weight 0. <think>...</think> reasoning is stripped from
accepted turns (nothink student), together with the now-stale "extra text before JSON" warning.
Output: {"kind": "agenttrove" | "agenttrove_handoff", "messages": [{role, content, weight}], "meta": {...}}.
"""

import argparse
import collections
import glob
import json
import random
import re
import os
import sys

import pyarrow.parquet as pq

THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)
REQ = "hand off your work to another AI agent"
CONT = "Here are the answers the other agent provided"
QPROMPT = "picking up work from a previous AI agent"
ANSWERS = "The next agent has a few questions for you"
WARN_BLOCK_RE = re.compile(r"^Previous response had warnings:\nWARNINGS: (.*?)\n\n", re.S)
PRE_JSON = "- Extra text detected before JSON object"
ERROR_RESULTS = re.compile(r"(?i)error|timeout|exception|cancel")
# default for --subset clean: drop the weak/small teachers (GPT 5.1 Nano, GPT-5-nano, GPT-5-mini, Qwen3, GPT-4o,
# Gemini-2.5-Flash); pass --teachers all to keep everything
STRONG_TEACHERS = "GLM-4.7,GLM-4.6,GLM-5.0,Kimi K2.0 Thinking,Kimi-2.5,MiniMax M2.0,GPT-5,GPT-OSS-120B"
META = ["original_source", "original_teacher", "trial_name", "run_id", "episode", "result", "judgment", "task"]


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_rows import SftWriter  # noqa: E402


def read_shard(shard: str):
    """(meta rows, conversations) of a shard, row group by row group: large shards overflow pyarrow's
    nested-column conversion when read in one piece."""
    f = pq.ParquetFile(shard)
    cols = [c for c in META if c in f.schema_arrow.names]
    meta, convs = [], []
    for g in range(f.num_row_groups):
        t = f.read_row_group(g, columns=cols + ["conversations"])
        meta += t.select(cols).to_pylist()
        convs += t.column("conversations").to_pylist()
    return meta, convs


def episode_num(e) -> int:
    m = re.search(r"(\d+)$", str(e or ""))
    return int(m.group(1)) if m else -1


def drop_pre_json_warning(text: str) -> str:
    m = WARN_BLOCK_RE.match(text)
    if not m:
        return text
    items = [w for w in m.group(1).split("\n") if w.strip() and w.strip() != PRE_JSON]
    head = f"Previous response had warnings:\nWARNINGS: {chr(10).join(items)}\n\n" if items else ""
    return head + text[m.end():]


THINK_BLOCK_RE = re.compile(r"^\s*<think>(.*?)</think>", re.S)


def split_think(text: str) -> tuple[str | None, str]:
    """(reasoning, rest) for a turn that opens with a closed think block; (None, text) otherwise.

    Two shapes: ``<think>...</think>rest`` and ``...</think>rest`` -- GLM-4.7's template puts the opening tag in the
    generation prompt, so its turns carry only the closing tag (exactly one, nothing tag-like before it).
    """
    m = THINK_BLOCK_RE.match(text)
    if m:
        reasoning = m.group(1).strip()
        return (reasoning or None), text[m.end():].strip()
    if text.count("</think>") == 1 and "<think>" not in text:
        reasoning, rest = text.split("</think>", 1)
        return (reasoning.strip() or None), rest.strip()
    return None, text


def handoff_kind(conv) -> str | None:
    users = [m.get("content") or "" for m in conv if m.get("role") == "user"]
    for kind, marker in (("continuation", CONT), ("answers", ANSWERS), ("questions", QPROMPT), ("summary", REQ)):
        if any(marker in u for u in users):
            return kind
    return None


def weight_messages(conv, parser, strip_think: bool, only_subagent_reply: bool = False,
                    think_to_reasoning: bool = False):
    out, prev_user, stale, n_rej = [], "", False, 0
    for m in conv:
        role, content = m.get("role"), m.get("content") or ""
        weight = 0
        reasoning = None
        if role == "assistant" and think_to_reasoning:
            # Keep the teacher's reasoning as reasoning_content (rendered in the student's think channel). Validity
            # is still judged on the original text, and the next user turn is left as the harness wrote it.
            reasoning, rest = split_think(content)
            if subagent := any(p in prev_user for p in (REQ, QPROMPT, ANSWERS)):
                weight = 1
            elif only_subagent_reply:
                weight = 0
            else:
                weight = 0 if parser.parse_response(content).error else 1
                n_rej += weight == 0
            content = rest
        elif role == "assistant":
            subagent = any(p in prev_user for p in (REQ, QPROMPT, ANSWERS))
            stripped = THINK_RE.sub("", content).strip() if strip_think else content
            if subagent:
                weight, content, stale = 1, stripped, False
            elif only_subagent_reply:
                weight, stale = 0, False
            elif parser.parse_response(content).error:
                weight, stale, n_rej = 0, False, n_rej + 1
            else:
                weight = 1
                stale = stripped != content and "before JSON" not in (parser.parse_response(stripped).warning or "")
                content = stripped
        elif role == "user":
            if stale:
                content = drop_pre_json_warning(content)
            prev_user = content
        msg = {"role": role, "content": content, "weight": weight}
        if reasoning:
            msg["reasoning_content"] = reasoning
        out.append(msg)
    return out, n_rej


def clean_completion(conv, parser, result, judgment) -> bool:
    if not conv or conv[-1].get("role") != "assistant":
        return False
    last = (conv[-1].get("content") or "").strip()
    if not last or (result and ERROR_RESULTS.search(str(result))):
        return False
    body = THINK_RE.sub("", last).strip()
    if body.count("{") != body.count("}"):
        return False
    success = str(result).lower() in ("1", "1.0", "success") or str(judgment).lower() == "yes"
    return success or bool(re.search(r'"task_complete"\s*:\s*true', body)) or parser.parse_response(body).is_task_complete


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards")
    ap.add_argument("out")
    ap.add_argument("--subset", choices=("clean", "handoff"), required=True)
    ap.add_argument("--cap-per-source", type=int, default=20000)
    ap.add_argument("--trials-per-task", type=int, default=3, help="handoff subset: trials kept per task")
    ap.add_argument("--teachers", default="")
    ap.add_argument("--no-strip-think", action="store_true")
    ap.add_argument("--think-to-reasoning", action="store_true",
                    help="move a leading <think>...</think> into reasoning_content instead of stripping it")
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--harbor-src", default="")
    a = ap.parse_args()
    if a.harbor_src:
        sys.path.insert(0, a.harbor_src)
    from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser

    parser = TerminusJSONPlainParser()
    teachers_arg = a.teachers or ("GLM-4.7" if a.subset == "handoff" else STRONG_TEACHERS)
    teachers = None if teachers_arg == "all" else {t.strip() for t in teachers_arg.split(",")}
    shards = sorted(glob.glob(a.shards))[: a.max_shards or None]
    stats = collections.Counter()

    # pass 1: candidate rows; keep the longest episode per (trial, run) or, for handoff, per task
    best: dict[tuple, tuple] = {}
    for si, shard in enumerate(shards):
        meta, convs = read_shard(shard)
        for ri, (r, conv) in enumerate(zip(meta, convs)):
            if not conv:
                stats["drop_null_conversation"] += 1
                continue
            src, teacher = r.get("original_source") or "unknown", r.get("original_teacher") or "?"
            if teachers is not None and teacher not in teachers:
                continue
            if a.subset == "clean":
                if src == "unknown" or src.startswith("exp_"):
                    continue
                if not clean_completion(conv, parser, r.get("result"), r.get("judgment")):
                    stats["drop_not_clean"] += 1
                    continue
                key = ("clean", src, r.get("trial_name"), r.get("run_id"))
            else:
                kind = handoff_kind(conv)
                if kind is None or kind == "questions":
                    continue
                key = ("handoff", src, r.get("trial_name"), r.get("run_id"), kind)
            rank = (episode_num(r.get("episode")), len(conv))
            if key not in best or rank > best[key][0]:
                best[key] = (rank, si, ri, src, teacher)
        stats["shards_scanned"] += 1

    rng = random.Random(a.seed)
    if a.subset == "handoff":
        # at most --trials-per-task trials per task (exp_tas runs the same tasks many times)
        task_of, trials_of_task = {}, collections.defaultdict(set)
        for si, shard in enumerate(shards):
            if any(v[1] == si for v in best.values()):
                meta, _ = read_shard(shard)
                for ri, r in enumerate(meta):
                    task_of[(si, ri)] = r.get("task") or r.get("trial_name")
        for key, v in best.items():
            trials_of_task[task_of.get((v[1], v[2]))].add(key[2:4])
        allowed = set()
        for task, trials in trials_of_task.items():
            trials = sorted(trials, key=str)
            rng.shuffle(trials)
            allowed.update((task, t) for t in trials[: a.trials_per_task])
        best = {k: v for k, v in best.items() if (task_of.get((v[1], v[2])), k[2:4]) in allowed}
        stats["handoff_rows_after_task_cap"] = len(best)
    # per-source cap (random, reproducible)
    by_src = collections.defaultdict(list)
    for key, v in best.items():
        by_src[v[3]].append(v)
    chosen = collections.defaultdict(set)
    for src, vs in by_src.items():
        rng.shuffle(vs)
        for v in vs[: a.cap_per_source]:
            chosen[v[1]].add(v[2])
        stats[f"src:{src}"] = min(len(vs), a.cap_per_source)

    # pass 2: convert
    kind = "agenttrove" if a.subset == "clean" else "agenttrove_handoff"
    with SftWriter(a.out) as out:
        for si, shard in enumerate(shards):
            if not chosen[si]:
                continue
            meta, convs = read_shard(shard)
            for ri in sorted(chosen[si]):
                r, conv = meta[ri], convs[ri]
                hk = handoff_kind(conv) if a.subset == "handoff" else None
                messages, n_rej = weight_messages(conv, parser, not a.no_strip_think,
                                                  only_subagent_reply=hk in ("summary", "answers"),
                                                  think_to_reasoning=a.think_to_reasoning)
                if not any(m["weight"] for m in messages):
                    stats["drop_no_trained_turn"] += 1
                    continue
                m = {k: r.get(k) for k in ("original_source", "original_teacher", "trial_name", "run_id", "episode", "result")}
                m.update(n_reasoning_turns=sum(bool(x.get("reasoning_content")) for x in messages))
                m.update(handoff_kind=hk, n_turns=sum(x["role"] == "assistant" for x in messages), n_rejected_turns=n_rej,
                         has_handoff=any(x["role"] == "user" and REQ in x["content"] for x in messages))
                out.write(kind, messages, m)
                stats["written"] += 1
                if hk:
                    stats[f"written_{hk}"] += 1
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
