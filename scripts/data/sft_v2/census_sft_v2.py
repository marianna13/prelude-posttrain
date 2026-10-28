"""Token/shape census of the Prelude SFT v2 candidate sources (issue 0058), over every row.

Per row: rendered-token estimate, assistant (trained) tokens, message count, thinking, tool calls, and for agent
sources the behavioural keep flag. Tokens are counted on message text with the Prelude chat tokenizer plus a fixed
per-message template overhead; the real render (step 6) replaces these numbers.

Writes <out>/<source>.parquet (one row per input row) and <out>/<source>.summary.json.
Usage: census_sft_v2.py --source i3:openreasoning_math --root SFT_SOURCES --out OUT [--procs N]
Sources: i3:<split> (INTELLECT-3-SFT), agenttrove, nebius.
"""
import argparse
import glob
import hashlib
import json
import os
import re
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MSG_OVERHEAD = 5  # <|im_start|>role\n ... <|im_end|>\n
TOK = None
KIND = None


def init(tokenizer_dir, kind):
    global TOK, KIND
    from tokenizers import Tokenizer
    TOK = Tokenizer.from_file(os.path.join(tokenizer_dir, 'tokenizer.json'))
    KIND = kind


def ntok(texts):
    texts = [t for t in texts if t]
    return sum(len(e.ids) for e in TOK.encode_batch(texts, add_special_tokens=False)) if texts else 0


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little', signed=True)


def msg_texts(m):
    out = [m.get('content') or '', m.get('reasoning_content') or '']
    for c in m.get('tool_calls') or []:
        f = c.get('function') or {}
        out.append((f.get('name') or '') + ' ' + (f.get('arguments') if isinstance(f.get('arguments'), str)
                                                   else json.dumps(f.get('arguments'))))
    return out


def has_think(m):
    return bool(m.get('reasoning_content')) or '<think>' in (m.get('content') or '')


def brace_balanced(s):
    s = s.strip()
    i = s.rfind('{')
    if i < 0:
        return False
    depth, instr, esc = 0, False, False
    for ch in s[s.find('{'):]:
        if instr:
            esc = (ch == '\\') and not esc
            if ch == '"' and not esc:
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
    return depth == 0


def row_stats(msgs, tools_json, extra):
    total = assistant = 0
    think = tool = False
    n_asst = 0
    for m in msgs:
        t = ntok(msg_texts(m)) + MSG_OVERHEAD
        total += t
        if m.get('role') == 'assistant':
            assistant += t
            n_asst += 1
            think |= has_think(m)
            tool |= bool(m.get('tool_calls'))
    if tools_json:
        total += ntok([tools_json])
    r = dict(tokens=total, assistant_tokens=assistant, messages=len(msgs), assistant_turns=n_asst,
             thinking=think, tool_calls=tool)
    r.update(extra)
    return r


def work(unit):
    path, rg = unit
    t = pq.ParquetFile(path).read_row_group(rg)
    rows = []
    for r in t.to_pylist():
        if KIND == 'i3':
            msgs = list(r.get('prompt') or []) + list(r.get('completion') or [])
            tools = r.get('tools')
            tools_json = tools if isinstance(tools, str) else (json.dumps(tools) if tools else '')
            first_user = next((m.get('content') or '' for m in msgs if m.get('role') == 'user'), '')
            src = r.get('source') or ''
            try:
                src = json.loads(src).get('dataset', src)
            except (ValueError, AttributeError):
                pass
            rows.append(row_stats(msgs, tools_json, dict(source=src, prompt_hash=h64(first_user))))
        elif KIND == 'agenttrove':
            msgs = r.get('conversations') or []
            key = '%s|%s|%s' % (r.get('trial_name'), r.get('run_id'), r.get('episode'))
            if not msgs:
                rows.append(dict(tokens=0, assistant_tokens=0, messages=0, assistant_turns=0, thinking=False,
                                 tool_calls=False, original_source=r.get('original_source') or '', null=True,
                                 key_hash=h64(key), content_hash=0, keep=False, teacher=r.get('model') or ''))
                continue
            last = msgs[-1]
            la = (last.get('content') or '') if last.get('role') == 'assistant' else ''
            ok = (last.get('role') == 'assistant' and bool(la.strip()) and brace_balanced(la.split('</think>')[-1])
                  and ('task_complete' in la or str(r.get('result')) in ('success', '1.0', '1')
                       or 'success' in str(r.get('judgment') or '').lower()))
            content = json.dumps(msgs, sort_keys=True)
            rows.append(row_stats(msgs, '', dict(original_source=r.get('original_source') or '', null=False,
                                                 key_hash=h64(key), content_hash=h64(content), keep=ok,
                                                 teacher=r.get('model') or '')))
        else:  # nebius
            msgs = r.get('trajectory') or []
            tools = r.get('tools')
            tools_json = tools if isinstance(tools, str) else (json.dumps(tools) if tools else '')
            rows.append(row_stats(msgs, tools_json, dict(instance_id=r.get('instance_id') or '',
                                                         resolved=int(r.get('resolved') or 0),
                                                         exit_status=r.get('exit_status') or '',
                                                         keep=bool(r.get('resolved')) and r.get('exit_status') == 'submit')))
    return rows


def pct(a, q):
    return int(np.percentile(a, q)) if len(a) else 0


def summarize(name, tbl):
    tok = np.asarray(tbl['tokens']); at = np.asarray(tbl['assistant_tokens'])
    s = dict(source=name, rows=len(tok), tokens=int(tok.sum()), assistant_tokens=int(at.sum()),
             tokens_p50=pct(tok, 50), tokens_p90=pct(tok, 90), tokens_p99=pct(tok, 99), tokens_max=int(tok.max()),
             over_64k=int((tok > 65536).sum()), messages_p50=pct(tbl['messages'], 50),
             thinking_rows=int(np.sum(tbl['thinking'])), tool_call_rows=int(np.sum(tbl['tool_calls'])))
    if 'keep' in tbl:
        keep = np.asarray(tbl['keep']) & (tok <= 65536)
        if 'key_hash' in tbl:  # agenttrove: first row per key, exact-duplicate content removed
            seen, first = set(), np.zeros(len(tok), bool)
            for i, (k, c) in enumerate(zip(tbl['key_hash'], tbl['content_hash'])):
                if not tbl['null'][i] and k not in seen:
                    seen.add(k); first[i] = True
            keep &= first
            s['null_rows'] = int(np.sum(tbl['null']))
            s['unique_keys'] = int(first.sum())
            by = {}
            for src, k, t_ in zip(tbl['original_source'], keep, tok):
                d = by.setdefault(src, [0, 0, 0]); d[0] += 1; d[1] += int(k); d[2] += int(t_) if k else 0
            s['by_original_source'] = {k: dict(rows=v[0], kept=v[1], kept_tokens=v[2])
                                       for k, v in sorted(by.items(), key=lambda kv: -kv[1][2])}
        s['kept_rows'] = int(keep.sum()); s['kept_tokens'] = int(tok[keep].sum())
        s['kept_assistant_tokens'] = int(at[keep].sum())
    if 'source' in tbl:
        by = {}
        for src, t_ in zip(tbl['source'], tok):
            d = by.setdefault(src, [0, 0]); d[0] += 1; d[1] += int(t_)
        s['by_source'] = {k: dict(rows=v[0], tokens=v[1]) for k, v in sorted(by.items(), key=lambda kv: -kv[1][1])}
        s['distinct_prompt_hashes'] = len(set(tbl['prompt_hash']))
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tokenizer', default='/p/data1/mmlaion/marianna/sft_sources/prelude_chat_tokenizer')
    p.add_argument('--procs', type=int, default=os.cpu_count())
    a = p.parse_args()
    if a.source.startswith('i3:'):
        kind, files = 'i3', sorted(glob.glob(f'{a.root}/PrimeIntellect/INTELLECT-3-SFT/{a.source[3:]}/*.parquet'))
    elif a.source == 'agenttrove':
        kind, files = 'agenttrove', sorted(glob.glob(f'{a.root}/open-thoughts/AgentTrove/**/*.parquet', recursive=True))
    elif a.source == 'nebius':
        kind, files = 'nebius', sorted(glob.glob(f'{a.root}/nebius/SWE-rebench-openhands-trajectories/*.parquet'))
    else:
        raise SystemExit('unknown source ' + a.source)
    if not files:
        raise SystemExit('no parquet files for ' + a.source)
    units = [(f, i) for f in files for i in range(pq.ParquetFile(f).num_row_groups)]
    with Pool(a.procs, initializer=init, initargs=(a.tokenizer, kind)) as pool:
        rows = [r for chunk in pool.imap(work, units, chunksize=1) for r in chunk]
    tbl = {k: [r[k] for r in rows] for k in rows[0]}
    name = re.sub(r'[^a-z0-9_]+', '_', a.source.lower())
    os.makedirs(a.out, exist_ok=True)
    pq.write_table(pa.table(tbl), f'{a.out}/{name}.parquet')
    s = summarize(a.source, tbl)
    s['files'] = len(files)
    with open(f'{a.out}/{name}.summary.json', 'w') as f:
        json.dump(s, f, indent=1)
    print(json.dumps({k: v for k, v in s.items() if not k.startswith('by_')}))


if __name__ == '__main__':
    main()
