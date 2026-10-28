"""Select and normalize the Prelude SFT v2 mix (issue 0058) into the SFT v1 conversation schema.

  select     per source, from census/<source>.parquet and decontam/flags/<source>.parquet (both row-aligned with the
             source files): decontam_drop removed, census length <= 64K, then
               i3:openreasoning_math   <= 4 responses per prompt        i3:openreasoning_code  <= 12 per prompt
               i3:openreasoning_science  random SCIENCE_FRACTION         other i3 splits        all
               agenttrove  the longest episode per (trial_name, run_id) (episodes are prefixes of one trajectory),
                           behavioural keep (census), then a per-(source, teacher) token cap to AGENT_TOKENS
               nebius      resolved + submit (census keep)
             and the thinking-off IF rows: SFT v1 if_chat / if_instruction_following (benchmark_free), sampled at
             NOTHINK_FRACTION, reasoning removed.  Writes selection/<name>.npy (selected rows) + selection.json.
  normalize  selected rows -> normalized/<mix_source>/part-<file>-<chunk>.parquet in materialize_prelude_sft_v1.py's schema
             (its norm_messages / norm_tools rules: reasoning kept only after the last user message, developer ->
             system, tool-call arguments parsed). Inline <think> is moved to reasoning_content. A row with no
             reasoning after its last user message is a thinking-off row: its assistant turns there start with
             the empty think block the template prefills for enable_thinking=false (Qwen3 hybrid convention).
             AgentTrove turns are trained only when they parse as Terminus-2 JSON or answer a handoff prompt.
uuid = "<source>:<census row>" links every normalized row back to census and flags.
Usage: materialize_prelude_sft_v2.py select|normalize --out MIX [--procs N]
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'sft_gen')]
from materialize_prelude_sft_v1 import SCHEMA, Reject, norm_messages, norm_tools  # noqa: E402
from agenttrove_convert import ANSWERS, QPROMPT, REQ, split_think  # noqa: E402

ROOT = '/p/data1/mmlaion/marianna/sft_sources'
V2 = '/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v2'
V1 = '/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v1'
SEED = 'prelude-sft-v2-20261023'
MAX_TOKENS = 65536
PER_PROMPT = {'i3:openreasoning_math': 4, 'i3:openreasoning_code': 12}
SCIENCE_FRACTION = 0.53  # 5.5B of 10.3B
AGENT_TOKENS = 8_000_000_000
NOTHINK_FRACTION = 0.75  # about 1.5B of the 2.0B rendered SFT v1 IF tokens
EMPTY_THINK = '<think>\n\n</think>\n\n'
MIX = {'i3:openreasoning_math': 'i3_math', 'i3:openreasoning_code': 'i3_code', 'i3:openreasoning_science': 'i3_science',
       'i3:am_chat': 'i3_am_chat', 'i3:am_if': 'i3_am_if', 'i3:openreasoning_tool': 'i3_tool',
       'i3:toucan_tool': 'i3_toucan', 'i3:swe_swiss': 'i3_swe_swiss', 'agenttrove': 'agenttrove', 'nebius': 'nebius'}
NOTHINK = ['if_chat', 'if_instruction_following']
PART_ROWS = {'nebius': 500, 'agenttrove': 5000}  # rows per normalized part (rendering cost grows with turns); else 20000


def name_of(source):
    return re.sub(r'[^a-z0-9_]+', '_', source.lower())


def rank(*parts):  # deterministic pseudo-random order
    return int.from_bytes(hashlib.blake2b('|'.join(map(str, (SEED,) + parts)).encode(), digest_size=8).digest(), 'big')


def source_files(source):  # same order as census_sft_v2.py
    if source.startswith('i3:'):
        return sorted(glob.glob(f'{ROOT}/PrimeIntellect/INTELLECT-3-SFT/{source[3:]}/*.parquet'))
    if source == 'agenttrove':
        return sorted(glob.glob(f'{ROOT}/open-thoughts/AgentTrove/**/*.parquet', recursive=True))
    return sorted(glob.glob(f'{ROOT}/nebius/SWE-rebench-openhands-trajectories/*.parquet'))


# ---------------------------------------------------------------- select

def water_fill(group_tokens, budget):
    """Cap c such that sum(min(t, c)) over groups = budget (all of it if the groups hold less)."""
    t = np.sort(np.asarray(list(group_tokens.values()), float))
    if t.sum() <= budget:
        return float('inf')
    lo, hi = 0.0, t[-1]
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.minimum(t, mid).sum() < budget else (lo, mid)
    return hi


def select_source(source, a):
    c = pq.read_table(f'{V2}/census/{name_of(source)}.parquet').to_pydict()
    f = pq.read_table(f'{V2}/decontam/flags/{name_of(source)}.parquet', columns=['prompt_key', 'decontam_drop']).to_pydict()
    tok = np.asarray(c['tokens'])
    ok = ~np.asarray(f['decontam_drop']) & (tok <= MAX_TOKENS) & (tok > 0)
    stats = dict(rows=len(tok), eligible=int(ok.sum()))
    if source in PER_PROMPT:
        by = collections.defaultdict(list)
        for i in np.flatnonzero(ok):
            by[f['prompt_key'][i]].append(i)
        keep = [i for p, rows in by.items() for i in sorted(rows, key=lambda r: rank(source, r))[:PER_PROMPT[source]]]
        sel = np.zeros(len(tok), bool); sel[keep] = True
    elif source == 'i3:openreasoning_science':
        sel = ok & np.asarray([rank(source, i) < SCIENCE_FRACTION * 2**64 for i in range(len(tok))])
    elif source == 'nebius':
        sel = ok & np.asarray(c['keep'])
    elif source == 'agenttrove':
        meta = []
        for p in source_files(source):
            meta += pq.read_table(p, columns=['trial_name', 'run_id', 'episode']).to_pylist()
        best = {}
        for i, m in enumerate(meta):
            if c['null'][i]:
                continue
            k = (m['trial_name'], m['run_id'])
            e = re.search(r'(\d+)$', str(m['episode'] or ''))
            e = int(e.group(1)) if e else -1
            if k not in best or (e, tok[i]) > best[k][0]:
                best[k] = ((e, tok[i]), i)
        longest = np.zeros(len(tok), bool); longest[[i for _, i in best.values()]] = True
        cand = ok & longest & np.asarray(c['keep'])
        stats.update(trials=len(best), longest_kept=int(cand.sum()), longest_kept_tokens=int(tok[cand].sum()))
        group = [f"{o}|{t}" for o, t in zip(c['original_source'], c['teacher'])]
        gt = collections.Counter()
        for i in np.flatnonzero(cand):
            gt[group[i]] += int(tok[i])
        cap = water_fill(gt, AGENT_TOKENS)
        used, sel = collections.Counter(), np.zeros(len(tok), bool)
        for i in sorted(np.flatnonzero(cand), key=lambda r: rank(source, r)):
            if used[group[i]] + tok[i] <= cap:
                used[group[i]] += int(tok[i]); sel[i] = True
        stats['group_cap_tokens'] = None if cap == float('inf') else int(cap)
        stats['by_group'] = {g: dict(candidate=gt[g], selected=used[g]) for g in sorted(gt, key=lambda g: -gt[g])}
    else:
        sel = ok
    stats.update(selected=int(sel.sum()), selected_tokens=int(tok[sel].sum()),
                 selected_assistant_tokens=int(np.asarray(c['assistant_tokens'])[sel].sum()))
    return np.flatnonzero(sel).astype(np.int64), stats


def select(a):
    os.makedirs(f'{a.out}/selection', exist_ok=True)
    report = dict(seed=SEED, max_tokens=MAX_TOKENS, per_prompt=PER_PROMPT, science_fraction=SCIENCE_FRACTION,
                  agent_tokens=AGENT_TOKENS, nothink_fraction=NOTHINK_FRACTION)
    for s in MIX:
        rows, st = select_source(s, a)
        np.save(f'{a.out}/selection/{name_of(s)}.npy', rows)
        report[s] = st
        print(s, json.dumps({k: v for k, v in st.items() if k != 'by_group'}), flush=True)
    for sub in NOTHINK:
        n = kept = 0
        for p in sorted(glob.glob(f'{V1}/normalized/{sub}/part-*.parquet')):
            fl = pq.read_table(f'{V1}/decontam/{sub}/{os.path.basename(p).replace(".parquet", ".flags.parquet")}',
                               columns=['row', 'benchmark_free']).to_pydict()
            n += len(fl['row'])
            kept += sum(free and rank(sub, os.path.basename(p), r) < NOTHINK_FRACTION * 2**64
                        for r, free in zip(fl['row'], fl['benchmark_free']))
        report['v1:' + sub] = dict(rows=n, selected=kept)
        print('v1:' + sub, report['v1:' + sub], flush=True)
    json.dump(report, open(f'{a.out}/selection/selection.json', 'w'), indent=1)


# ---------------------------------------------------------------- normalize

def terminus_ok(text):
    s = split_think(text)[1]
    i, j = s.find('{'), s.rfind('}')
    if i < 0 or j < i:
        return False
    try:
        d = json.loads(s[i:j + 1])
    except ValueError:
        return False
    return isinstance(d, dict) and bool({'commands', 'task_complete', 'analysis', 'plan'} & set(d))


def inline_think(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get('role') == 'assistant' and not m.get('reasoning_content') and isinstance(m.get('content'), str):
            r, rest = split_think(m['content'])
            if r is not None or '</think>' in m['content']:
                m['reasoning_content'], m['content'] = r, rest
        out.append(m)
    return out


def thinking_off(conv):
    """Empty think block on the assistant turns after the last user message when none of them has reasoning."""
    last_user = max(i for i, m in enumerate(conv) if m['role'] == 'user')
    tail = [m for m in conv[last_user + 1:] if m['role'] == 'assistant']
    if any(m['reasoning_content'] for m in tail):
        return conv, False
    for m in tail:
        m['content'] = EMPTY_THINK + (m['content'] or '')
    return conv, True


def convert(source, r):
    tools = None
    if source.startswith('i3:'):
        msgs = inline_think(list(r.get('prompt') or []) + list(r.get('completion') or []))
        conv, tools = norm_messages(msgs), norm_tools(r.get('tools'))
    elif source == 'nebius':
        conv, tools = norm_messages(inline_think(r.get('trajectory') or [])), norm_tools(r.get('tools'))
    elif source == 'agenttrove':
        msgs, prev_user, weights = inline_think(r.get('conversations') or []), '', []
        for m in msgs:
            if m.get('role') == 'user':
                prev_user = m.get('content') or ''
            elif m.get('role') == 'assistant':
                raw = (('<think>' + m['reasoning_content'] + '</think>') if m.get('reasoning_content') else '') + (m.get('content') or '')
                weights.append(any(p in prev_user for p in (REQ, QPROMPT, ANSWERS)) or terminus_ok(raw))
                continue
            weights.append(False)
        conv = norm_messages(msgs, weights)
    else:  # SFT v1 IF row, already normalized: thinking-off copy
        conv = [dict(m, reasoning_content=None) for m in r['conversations']]
        tools = r.get('tools_json')
    return thinking_off(conv) + (tools,)


class PartWriter:
    """<prefix>-<chunk>.parquet parts of at most `rows` rows; <prefix>.done marks the input as finished."""

    def __init__(self, prefix, rows):
        self.prefix, self.rows, self.chunk, self.buf, self.n, self.w = prefix, rows, 0, [], 0, None

    def add(self, row):
        self.buf.append(row)
        if len(self.buf) >= 2000 or self.n + len(self.buf) >= self.rows:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        if self.w is None:
            self.w = pq.ParquetWriter(f'{self.prefix}-{self.chunk:03d}.parquet', SCHEMA, compression='zstd')
        self.w.write_table(pa.Table.from_pylist(self.buf, schema=SCHEMA))
        self.n += len(self.buf); self.buf = []
        if self.n >= self.rows:
            self.w.close(); self.w, self.n, self.chunk = None, 0, self.chunk + 1

    def close(self):
        self.flush()
        if self.w is not None:
            self.w.close()
        open(self.prefix + '.done', 'w').close()


def normalize_file(job):
    source, path, rows, out, offset = job
    want = set(rows.tolist())
    stats = collections.Counter()
    writer = PartWriter(out, PART_ROWS.get(source, 20000))
    pf = pq.ParquetFile(path)
    k = offset
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if np.searchsorted(rows, k + n) == np.searchsorted(rows, k):  # nothing selected in this row group
            k += n
            continue
        for r in pf.read_row_group(g).to_pylist():
            if k in want:
                try:
                    conv, off, tools = convert(source, r)
                    stats['kept'] += 1; stats['thinking_off'] += off
                    writer.add(dict(conversations=conv, tools_json=tools, mix_source=MIX[source], uuid=f'{source}:{k}',
                                    license=None, hf_repo=None, hf_revision=None, file=os.path.basename(path)))
                except Reject as e:
                    stats[f'reject_{e}'] += 1
            k += 1
    writer.close()
    return MIX[source], dict(stats)


def normalize_v1(job):
    sub, path, out = job
    fl = pq.read_table(f'{V1}/decontam/{sub}/{os.path.basename(path).replace(".parquet", ".flags.parquet")}',
                       columns=['row', 'benchmark_free']).to_pydict()
    keep = {r for r, free in zip(fl['row'], fl['benchmark_free'])
            if free and rank(sub, os.path.basename(path), r) < NOTHINK_FRACTION * 2**64}
    stats, k = collections.Counter(), 0
    writer = PartWriter(out, 20000)
    for b in pq.ParquetFile(path).iter_batches(batch_size=1000):
        for r in b.to_pylist():
            if k in keep:
                conv, off, tools = convert('v1', r)
                stats['kept'] += 1; stats['thinking_off'] += off
                writer.add(dict(r, conversations=conv, tools_json=tools, mix_source=f'v1_{sub}_nothink',
                                uuid=f'v1:{sub}:{os.path.basename(path)}:{k}'))
            k += 1
    writer.close()
    return f'v1_{sub}_nothink', dict(stats)


def normalize(a):
    from multiprocessing import Pool
    jobs, v1_jobs = [], []
    for s, mix in MIX.items():
        rows = np.load(f'{a.out}/selection/{name_of(s)}.npy')
        os.makedirs(f'{a.out}/normalized/{mix}', exist_ok=True)
        off = 0
        for k, p in enumerate(source_files(s)):
            n = pq.ParquetFile(p).metadata.num_rows
            part = rows[(rows >= off) & (rows < off + n)]
            out = f'{a.out}/normalized/{mix}/part-{k:03d}'
            if len(part) and not os.path.exists(out + '.done'):
                jobs.append((s, p, part, out, off))
            off += n
    for sub in NOTHINK:
        os.makedirs(f'{a.out}/normalized/v1_{sub}_nothink', exist_ok=True)
        for p in sorted(glob.glob(f'{V1}/normalized/{sub}/part-*.parquet')):
            out = f'{a.out}/normalized/v1_{sub}_nothink/{os.path.basename(p)[:-len(".parquet")]}'
            if not os.path.exists(out + '.done'):
                v1_jobs.append((sub, p, out))
    totals = collections.defaultdict(collections.Counter)
    with Pool(a.procs) as pool:
        for mix, st in pool.imap_unordered(normalize_file, sorted(jobs, key=lambda j: -len(j[2]))):
            totals[mix].update(st)
        for mix, st in pool.imap_unordered(normalize_v1, v1_jobs):
            totals[mix].update(st)
    path = f'{a.out}/normalized/stats.json'  # a resumed run adds to the earlier counts
    allst = json.load(open(path)) if os.path.exists(path) else {}
    for mix, st in totals.items():
        allst[mix] = dict(collections.Counter(allst.get(mix, {})) + st)
    json.dump(allst, open(path, 'w'), indent=1, sort_keys=True)
    print(json.dumps(allst, indent=1, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['select', 'normalize'])
    p.add_argument('--out', default=V2)
    p.add_argument('--procs', type=int, default=os.cpu_count())
    a = p.parse_args()
    dict(select=select, normalize=normalize)[a.step](a)


if __name__ == '__main__':
    main()
