"""Benchmark decontamination, RL-pool / agentic-eval exclusion and prompt dedup flags for Prelude SFT v2 (issue 0058).

Three steps; rows are never dropped here (the mix step filters on the flags):

  refs      prompt keys of the reference corpora: SFT v1 (normalized), Nemotron-Cascade-2 (whole pool, a superset of
            what mid-training saw) and Nemotron-Terminal-Corpus.
  flags     first the instruction texts of the RL task pools (TMax-1k, RTS-1k, CalibForge) and the agentic evals
            (TB2.1, TBLite); then per source and row group, in census_sft_v2.py row order: benchmark 13-gram hits against index_prelude_v1
            (prompt side and assistant side apart), the prompt key, and for agent sources the best-matching RL-pool /
            eval task by instruction 13-grams.
  finalize  drops boilerplate benchmark n-grams, joins the reference keys and swe_overlap.py matches and writes
            <out>/flags/<source>.parquet (row-aligned with census/<source>.parquet) plus <out>/summary.json.

Benchmark hashing is decontam_conversations.py's (lowercase, whitespace words, md5[:8] of word 13-grams, n-grams with
fewer than 7 distinct words skipped). The boilerplate cut counts distinct prompts, not rows: INTELLECT-3 repeats each
math prompt about 15 times and each code prompt about 60, so a per-row count would discard real leaks.
Prompt key: blake2b-64 of the first user message, lowercased with whitespace collapsed.
Usage: decontam_sft_v2.py refs|flags|finalize --root SFT_SOURCES --out OUT [--source S] [--procs N]
"""
import argparse
import collections
import glob
import hashlib
import io
import json
import os
import re
import sys
import tarfile
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

INDEX = '/p/data1/mmlaion/marianna/decontam/index_prelude_v1'
DEPS = '/p/project1/laionize/marianna/dc_agent/.prelude_census_deps'  # zstandard (no zstd binary on compute nodes)
SFT_V1 = '/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v1/normalized'
MIDTRAIN = '/p/data1/mmlaion/marianna/midtrain_sources'
TEACHER = '/p/scratch/laionize/marianna/teacher_traces_rts_tmax/hf'
CALIBFORGE = '/p/scratch/laionize/marianna/calibforge_env_reliable_v1/tasks.parquet'
EVAL_TASKS = {
    'terminal_bench_2_1': '/p/scratch/synthlaion/dc-agent-shared/hf_hub/terminal-bench-2-1/tasks/*/instruction.md',
    'openthoughts_tblite': '/p/scratch/synthlaion/dc-agent-shared/hf_hub/datasets--open-thoughts--OpenThoughts-TBLite/'
                           'snapshots/7b70111339b4af23cece95d63aeec1c705790868/*/instruction.md'}
SOURCES = ['nebius', 'agenttrove', 'i3:am_chat', 'i3:am_if', 'i3:openreasoning_tool', 'i3:toucan_tool',
           'i3:swe_swiss', 'i3:openreasoning_science', 'i3:openreasoning_code', 'i3:openreasoning_math']
N, MIN_DISTINCT, CAP = 13, 7, 50
TASK_COMMON = 5  # an instruction 13-gram in more than this many target tasks is template text
MAX_DF = 20  # a benchmark n-gram matched by more than this many distinct v2 prompts is boilerplate
TASK_MIN_SHARED = 20  # task match: >= 20 shared 13-grams, or >= half of a short target's 13-grams
BENCH = TASKS = None


def name_of(source):
    return re.sub(r'[^a-z0-9_]+', '_', source.lower())


def prompt_key(text):
    s = ' '.join((text or '').lower().split())
    return int.from_bytes(hashlib.blake2b(s.encode('utf-8', 'ignore'), digest_size=8).digest(), 'big')


def grams(text, min_distinct=MIN_DISTINCT):
    w = (text or '').lower().split()
    for i in range(len(w) - N + 1):
        g = w[i:i + N]
        if len(set(g)) >= min_distinct:
            yield int.from_bytes(hashlib.md5(' '.join(g).encode('utf-8', 'ignore')).digest()[:8], 'big')


def bench_hits(text):
    found = set()
    for h in grams(text):
        if h in BENCH:
            found.add(h)
            if len(found) >= CAP:
                break
    return sorted(found)


def first_user(msgs):
    c = next((m.get('content') or '' for m in msgs if m.get('role') == 'user'), '')
    return c if isinstance(c, str) else json.dumps(c)


def msg_text(m):
    c = m.get('content') or ''
    out = [m.get('reasoning_content') or '', c if isinstance(c, str) else json.dumps(c)]
    tc = m.get('tool_calls') or []
    if isinstance(tc, str):
        out.append(tc)
    else:
        for c in tc:
            f = c.get('function') or {}
            a = f.get('arguments')
            out.append((f.get('name') or '') + ' ' + (a if isinstance(a, str) else json.dumps(a)))
    return '\n'.join(x for x in out if x)


# ---------------------------------------------------------------- refs

def keys_parquet(path):
    out = []
    for b in pq.ParquetFile(path).iter_batches(batch_size=1000, columns=['conversations']):
        out += [prompt_key(first_user(c or [])) for c in b.column('conversations').to_pylist()]
    return path, out


def keys_jsonl(unit):
    path, start, end = unit
    out = []
    with open(path, 'rb') as f:
        if start:  # skip the line containing byte start-1: it belongs to the previous unit
            f.seek(start - 1)
            f.readline()
        while f.tell() <= end:
            line = f.readline()
            if not line:
                break
            try:
                out.append(prompt_key(first_user(json.loads(line).get('messages') or [])))
            except ValueError:
                pass
    return path, out


def task_texts():
    """(pool, task_id, instruction) for the RL task pools and the agentic evals."""
    items = []
    for pool, tar in (('tmax_1k', f'{TEACHER}/tmax-legacy-1k-rootless-apptainer/data/tasks.tar.zst'),
                      ('rts_1k', f'{TEACHER}/rts-1k-rootless-apptainer/tasks.tar.zst')):
        sys.path.append(DEPS)
        import zstandard
        with open(tar, 'rb') as f, zstandard.ZstdDecompressor().stream_reader(f) as z, \
                tarfile.open(fileobj=z, mode='r|') as tf:
            for m in tf:
                if m.isfile() and m.name.endswith('/instruction.md'):
                    items.append((pool, m.name.split('/')[-2], tf.extractfile(m).read().decode('utf-8', 'ignore')))
    pf = pq.ParquetFile(CALIBFORGE)
    for i in range(pf.num_row_groups):
        for r in pf.read_row_group(i, columns=['path', 'task_binary']).to_pylist():
            with tarfile.open(fileobj=io.BytesIO(r['task_binary'])) as tf:
                items.append(('calibforge', r['path'], tf.extractfile('instruction.md').read().decode('utf-8', 'ignore')))
    for pool, g in EVAL_TASKS.items():
        for p in sorted(glob.glob(g)):
            items.append((pool, p.split('/')[-2], open(p, errors='ignore').read()))
    return items


def write_tasks(out):
    """<out>/tasks/tasks.jsonl, read by the flags workers of the agent sources."""
    path = f'{out}/tasks/tasks.jsonl'
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tasks = task_texts()
    with open(path + '.tmp', 'w') as f:
        for pool, tid, text in tasks:
            f.write(json.dumps(dict(pool=pool, task=tid, text=text)) + '\n')
    os.replace(path + '.tmp', path)
    counts = collections.Counter(p for p, _, _ in tasks)
    json.dump(counts, open(f'{out}/tasks/tasks.json', 'w'), indent=1)
    print('tasks', dict(counts), flush=True)


def refs(a):
    os.makedirs(f'{a.out}/refs', exist_ok=True)
    report = {}
    corpora = {}  # name -> list of jobs
    for f in sorted(glob.glob(f'{SFT_V1}/*/part-*.parquet')):
        corpora.setdefault('sft_v1/' + f.split('/')[-2], []).append(('pq', f))
    for f in sorted(glob.glob(f'{MIDTRAIN}/Nemotron-Terminal-Corpus/**/*.parquet', recursive=True)):
        if '/.cache/' not in f:
            corpora.setdefault('terminal_corpus', []).append(('pq', f))
    step = 1 << 30
    for f in sorted(glob.glob(f'{MIDTRAIN}/Nemotron-Cascade-2-SFT-Data/*/*.jsonl')):
        size = os.path.getsize(f)
        for s in range(0, size, step):
            corpora.setdefault('cascade2/' + f.split('/')[-2], []).append(('jsonl', (f, s, min(s + step, size) - 1)))
    jobs = [(name, kind, x) for name, js in corpora.items() for kind, x in js]
    pq_jobs = [x for _, k, x in jobs if k == 'pq']
    js_jobs = [x for _, k, x in jobs if k == 'jsonl']
    owner = {}
    for name, kind, x in jobs:
        owner[x if kind == 'pq' else x[0]] = name
    acc = collections.defaultdict(set)
    rows = collections.Counter()
    with Pool(a.procs) as pool:
        for path, ks in pool.imap_unordered(keys_parquet, pq_jobs):
            acc[owner[path]].update(ks); rows[owner[path]] += len(ks)
        for path, ks in pool.imap_unordered(keys_jsonl, js_jobs):
            acc[owner[path]].update(ks); rows[owner[path]] += len(ks)
    for name, ks in acc.items():
        np.save(f'{a.out}/refs/{name.replace("/", "__")}.npy', np.fromiter(sorted(ks), np.uint64, len(ks)))
    report['corpora'] = {n: dict(rows=rows[n], distinct_prompts=len(acc[n])) for n in sorted(acc)}
    json.dump(report, open(f'{a.out}/refs/refs.json', 'w'), indent=1)
    print(json.dumps(report['corpora'], indent=1))


# ---------------------------------------------------------------- flags

class TaskIndex:
    def __init__(self, items):
        self.ids = [f'{p}:{t}' for p, t, _ in items]
        post = collections.defaultdict(set)
        self.size = []
        for n, (_, _, text) in enumerate(items):
            g = set(grams(text, 1))
            self.size.append(len(g))
            for h in g:
                post[h].add(n)
        self.post = {h: v for h, v in post.items() if len(v) <= TASK_COMMON}

    def best(self, text):
        c = collections.Counter(n for h in set(grams(text, 1)) for n in self.post.get(h, ()))
        if not c:
            return '', 0, 0
        n, shared = c.most_common(1)[0]
        return self.ids[n], shared, self.size[n]


def init(kind, out):
    global BENCH, TASKS
    BENCH = set(np.load(f'{INDEX}/ngrams.npy').tolist())
    if kind in ('agenttrove', 'nebius'):
        TASKS = TaskIndex([(d['pool'], d['task'], d['text']) for d in map(json.loads, open(f'{out}/tasks/tasks.jsonl'))])


def flag_unit(job):
    kind, path, rg, dest = job
    rows, pairs, cache = [], set(), {}
    for r in pq.ParquetFile(path).read_row_group(rg).to_pylist():
        extra = dict(task='', original_source='', instance_id='')
        if kind == 'i3':
            msgs = list(r.get('prompt') or []) + list(r.get('completion') or [])
        elif kind == 'agenttrove':
            msgs = r.get('conversations') or []
            extra.update(task=r.get('task') or '', original_source=r.get('original_source') or '')
        else:
            msgs = r.get('trajectory') or []
            extra['instance_id'] = r.get('instance_id') or ''
        prompt = '\n'.join(msg_text(m) for m in msgs if m.get('role') != 'assistant')
        answer = '\n'.join(msg_text(m) for m in msgs if m.get('role') == 'assistant')
        pk = prompt_key(first_user(msgs))
        ck = hashlib.blake2b(prompt.encode('utf-8', 'ignore'), digest_size=16).digest()
        if ck not in cache:
            cache = cache if len(cache) < 4096 else {}
            cache[ck] = bench_hits(prompt)
        ph, ah = cache[ck], bench_hits(answer)
        for h in set(ph) | set(ah):
            pairs.add((h, pk))
        task, shared, size = '', 0, 0
        if TASKS is not None:
            instr = (r.get('instruction') if kind == 'agenttrove' else None) or first_user(msgs)
            task, shared, size = TASKS.best(instr)
        rows.append(dict(prompt_key=pk, prompt_hashes=ph, assistant_hashes=ah, task_match=task, task_shared=shared,
                         task_ngrams=size, **extra))
    schema = pa.schema([('prompt_key', pa.uint64()), ('prompt_hashes', pa.list_(pa.uint64())),
                        ('assistant_hashes', pa.list_(pa.uint64())), ('task_match', pa.string()),
                        ('task_shared', pa.int32()), ('task_ngrams', pa.int32()), ('task', pa.string()),
                        ('original_source', pa.string()), ('instance_id', pa.string())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), dest + '.tmp')
    os.replace(dest + '.tmp', dest)
    return dest, sorted(pairs)


def source_files(root, source):  # same order as census_sft_v2.py
    if source.startswith('i3:'):
        return 'i3', sorted(glob.glob(f'{root}/PrimeIntellect/INTELLECT-3-SFT/{source[3:]}/*.parquet'))
    if source == 'agenttrove':
        return 'agenttrove', sorted(glob.glob(f'{root}/open-thoughts/AgentTrove/**/*.parquet', recursive=True))
    return 'nebius', sorted(glob.glob(f'{root}/nebius/SWE-rebench-openhands-trajectories/*.parquet'))


def flags(a):
    write_tasks(a.out)
    for source in ([a.source] if a.source else SOURCES):
        kind, files = source_files(a.root, source)
        d = f'{a.out}/work/{name_of(source)}'
        if os.path.exists(f'{d}/pairs.npy'):
            print('=== done', source, flush=True)
            continue
        os.makedirs(d, exist_ok=True)
        units = [(kind, f, i, f'{d}/{k:03d}-{i:05d}.parquet') for k, f in enumerate(files)
                 for i in range(pq.ParquetFile(f).num_row_groups)]
        todo = [u for u in units if not os.path.exists(u[3])]
        print(f'=== {source}: {len(units)} row groups, {len(todo)} to do', flush=True)
        pairs = set()
        with Pool(a.procs, initializer=init, initargs=(kind, a.out)) as pool:
            for k, (_, p) in enumerate(pool.imap_unordered(flag_unit, todo, chunksize=1)):
                pairs.update(p)
                if k % 200 == 0:
                    print(f'  {k}/{len(todo)}', flush=True)
        for u in units:  # units finished by an earlier attempt: recover their pairs
            if u not in todo:
                t = pq.read_table(u[3], columns=['prompt_key', 'prompt_hashes', 'assistant_hashes']).to_pylist()
                pairs.update((h, r['prompt_key']) for r in t for h in set(r['prompt_hashes']) | set(r['assistant_hashes']))
        np.save(f'{d}/pairs.npy', np.array(sorted(pairs), np.uint64).reshape(-1, 2))


# ---------------------------------------------------------------- finalize

def finalize(a):
    census = f'{a.out}/../census'
    df = collections.Counter()
    for s in SOURCES:
        for h, _ in np.load(f'{a.out}/work/{name_of(s)}/pairs.npy').tolist():
            df[h] += 1
    # df = distinct prompts per hash, summed over sources (a prompt in two sources counts twice)
    generic = {h for h, n in df.items() if n > MAX_DF}
    refs_ = {}
    for f in sorted(glob.glob(f'{a.out}/refs/*.npy')):
        refs_[os.path.basename(f)[:-4].replace('__', '/')] = np.load(f)
    ref_groups = collections.defaultdict(list)
    for n, arr in refs_.items():
        ref_groups[n.split('/')[0]].append((n, arr))
    nebius_drop, at_drop = set(), set()
    for d in map(json.loads, open(f'{census}/swe_overlap_nebius_instances.jsonl')):
        nebius_drop.add(d['query'])
    for d in map(json.loads, open(f'{census}/swe_overlap_agenttrove_swe_tasks.jsonl')):
        at_drop.add((d['original_source'], d['query']))
    keys_by_source = {}
    summary = dict(_config=dict(index=INDEX, ngram=N, min_distinct_words=MIN_DISTINCT, max_df_distinct_prompts=MAX_DF,
                                generic_hashes=len(generic), matched_hashes=len(df), task_min_shared=TASK_MIN_SHARED,
                                task_common_cutoff=TASK_COMMON,
                                refs=json.load(open(f'{a.out}/refs/refs.json')),
                                tasks=json.load(open(f'{a.out}/tasks/tasks.json'))))
    os.makedirs(f'{a.out}/flags', exist_ok=True)
    for s in SOURCES:
        parts = sorted(glob.glob(f'{a.out}/work/{name_of(s)}/*-*.parquet'))
        t = pa.concat_tables(pq.read_table(p) for p in parts)
        c = pq.read_table(f'{census}/{name_of(s)}.parquet', columns=['tokens', 'assistant_tokens'])
        if c.num_rows != t.num_rows:
            raise SystemExit(f'{s}: {t.num_rows} flag rows vs {c.num_rows} census rows')
        tok = np.asarray(c['tokens'])
        pk = np.asarray(t['prompt_key'], np.uint64)
        keys_by_source[s] = np.unique(pk)
        ph = [sum(h not in generic for h in x) for x in t['prompt_hashes'].to_pylist()]
        ah = [sum(h not in generic for h in x) for x in t['assistant_hashes'].to_pylist()]
        ph, ah = np.asarray(ph), np.asarray(ah)
        bench = (ph > 0) | (ah > 0)
        shared, size = np.asarray(t['task_shared']), np.asarray(t['task_ngrams'])
        task_hit = (shared >= TASK_MIN_SHARED) | ((size > 0) & (2 * shared >= size) & (shared >= 3))
        task_match = np.asarray(t['task_match'].to_pylist(), object)
        is_eval = np.asarray([x.split(':')[0] in EVAL_TASKS for x in task_match])
        rl_pool = task_hit & ~is_eval
        agent_eval = task_hit & ~rl_pool
        if s == 'nebius':
            swe = np.isin(np.asarray(t['instance_id'].to_pylist(), object), list(nebius_drop))
        elif s == 'agenttrove':
            swe = np.asarray([(o, k) in at_drop for o, k in zip(t['original_source'].to_pylist(), t['task'].to_pylist())])
        else:
            swe = np.zeros(len(pk), bool)
        out = dict(row=np.arange(len(pk)), prompt_key=pk, bench_prompt_hits=ph, bench_assistant_hits=ah,
                   benchmark_free=~bench, rl_pool_overlap=rl_pool, agent_eval_overlap=agent_eval, swe_overlap=swe,
                   task_match=task_match, task_shared=shared)
        dup = {}
        for g, lst in ref_groups.items():
            hit = np.zeros(len(pk), bool)
            for n, arr in lst:
                hit |= np.isin(pk, arr)
            dup[g] = hit
            out[f'prompt_in_{g}'] = hit
        drop = bench | rl_pool | agent_eval | swe
        out['decontam_drop'] = drop
        pq.write_table(pa.table({k: (list(v) if v.dtype == object else v) for k, v in out.items()}),
                       f'{a.out}/flags/{name_of(s)}.parquet')

        def cnt(m):
            return dict(rows=int(m.sum()), tokens=int(tok[m].sum()))
        summary[s] = dict(rows=len(pk), tokens=int(tok.sum()), distinct_prompts=len(keys_by_source[s]),
                          benchmark_prompt=cnt(ph > 0), benchmark_assistant_only=cnt((ph == 0) & (ah > 0)),
                          benchmark_any=cnt(bench), rl_pool=cnt(rl_pool), agent_eval=cnt(agent_eval), swe_overlap=cnt(swe),
                          decontam_drop=cnt(drop), **{f'prompt_in_{g}': cnt(m) for g, m in dup.items()},
                          task_pools=dict(collections.Counter(x.split(':')[0] for x in task_match[task_hit])))
        print(s, json.dumps(summary[s]), flush=True)
    # exact prompt shared across v2 sources (counted per source pair; the mix keeps one copy)
    names = list(keys_by_source)
    summary['_cross_source_prompts'] = {f'{x}|{y}': int(np.intersect1d(keys_by_source[x], keys_by_source[y]).size)
                                        for i, x in enumerate(names) for y in names[i + 1:]
                                        if np.intersect1d(keys_by_source[x], keys_by_source[y]).size}
    json.dump(summary, open(f'{a.out}/summary.json', 'w'), indent=1)
    print(json.dumps(summary['_cross_source_prompts'], indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['refs', 'flags', 'finalize'])
    p.add_argument('--root', default='/p/data1/mmlaion/marianna/sft_sources')
    p.add_argument('--out', required=True)
    p.add_argument('--source', help='flags: one source (default: all)')
    p.add_argument('--procs', type=int, default=os.cpu_count())
    a = p.parse_args()
    dict(refs=refs, flags=flags, finalize=finalize)[a.step](a)


if __name__ == '__main__':
    main()
