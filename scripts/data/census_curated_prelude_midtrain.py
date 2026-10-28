#!/usr/bin/env python3
"""Step 4: exact token/mask census of the verified canonical Prelude corpus.

Counts every survivor anew. Documents use the verified LF barrier; conversations
use the unchanged step-2 renderer and its semantic tool masks. No token-ID release
or model training is performed. Stage counts describe explicit virtual chunks for
step 5 to implement and check against these counts.
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
import shutil
import sys
import time

DEPS = '/p/project1/laionize/marianna/dc_agent/.prelude_census_deps'
sys.path.insert(0, DEPS)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pyarrow as pa
import pyarrow.parquet as pq
from prelude_midtrain import CASCADE, SPECIALIZED, STACK, TERMINAL, ULTRA, dump, render_conversation
from prelude_lf_counting import LFCountingTokenizer

CODE = ['census_curated_prelude_midtrain.py', 'prelude_midtrain.py', 'prelude_lf_counting.py']
STAGES = [(16384, 8192), (32768, 16384), (65536, 32768)]
OVERLAP = 256
POLICY = {
    'document': 'exact original text, native BOS 1 / EOS 2, BOS masked',
    'conversation': 'unchanged step-2 render_conversation, all ordinary targets, complete tool groups masked',
    'replay': 'rendered tokens <=4096 including BOS/EOS; same cut at every stage',
    'long_pdf': 'rendered original tokens >=32768 including BOS/EOS',
    'categories': 'one category per canonical row; Specialized remains raw document format',
    'stages': [{'context': c, 'minimum_original_tokens': m} for c, m in STAGES],
    'continuation': {'overlap_tokens': OVERLAP, 'added_bos': 1, 'overlap_and_added_bos_loss': 0,
                     'new_content': 'consume every original token exactly once; no intermediate EOS',
                     'tails': 'keep final short chunk, same source/split/group as parent',
                     'implementation': 'virtual census only; step-5 token-ID export must match'},
    'short_sources': 'replay is the sole minimum-length exception; report missing stage pools explicitly',
}
FIELDS = [(x, pa.string()) for x in ['id', 'mix_source', 'pool', 'category', 'kind', 'split', 'split_group',
                                    'language', 'eligibility_reason']]
FIELDS += [(x, pa.int64()) for x in ['tokens', 'trainable_tokens', 'masked_tokens', 'chars']]
FIELDS += [('eligible', pa.bool_())]
for c, _ in STAGES:
    FIELDS += [(f'stage_{c}_{x}', pa.int64()) for x in ['chunks', 'seen_tokens', 'trainable_tokens']]
METRICS = pa.schema(FIELDS)
_MANIFEST = _OUT = _HF = _COUNTER = None


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(16 * 1024**2):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(dump(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def state(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def classify(mix_source, kind, provenance=None):
    """Explicit registry: never silently assign an unknown source to a category."""
    source, subset = mix_source.split('/', 1)
    language = ''
    if source == CASCADE:
        category = {'math': 'reasoning', 'science': 'reasoning', 'swe': 'code_agent',
                    'conversational_agent': 'code_agent', 'chat': 'chat_if',
                    'instruction_following': 'chat_if'}[subset]
        if kind != 'conversation':
            raise ValueError('Cascade format changed')
        pool = mix_source
    elif source == TERMINAL:
        category, pool = 'code_agent', TERMINAL
        if kind != 'conversation':
            raise ValueError('Terminal format changed')
    else:
        if kind != 'document':
            raise ValueError('document source format changed')
        pool = mix_source
        if source == SPECIALIZED:
            category = {'Nemotron-Pretraining-InfiniByte-Reasoning': 'reasoning',
                        'Nemotron-Pretraining-Math-Textbooks': 'reasoning',
                        'Nemotron-Pretraining-RQA': 'reasoning',
                        'Nemotron-Pretraining-STEM-SFT': 'reasoning',
                        'Nemotron-Pretraining-Scientific-Coding': 'code_agent'}[subset]
        elif source == STACK:
            category = {'data/code': 'code_agent', 'data/long_documents': 'long_documents'}[subset]
        elif source == ULTRA:
            if subset.split('/')[-1] not in {'cpp', 'csharp', 'go', 'java', 'javascript', 'php',
                                            'python', 'r', 'ruby', 'rust', 'shell'}:
                raise ValueError('unknown Ultra language')
            category, language = 'code_agent', subset.split('/')[-1]
        elif source == 'FinePDFs-English-long' and subset == 'long_documents':
            category, language = 'long_documents', 'eng_Latn'
        elif source in {'dclm-baseline-1.0', 'Nemotron-CC-original-high', 'FinePDFs-English-replay',
                        'FinePDFs-EU-replay'}:
            expected = {'dclm-baseline-1.0': 'dclm_english', 'Nemotron-CC-original-high': 'nemotron_english',
                        'FinePDFs-English-replay': 'pdf_english', 'FinePDFs-EU-replay': 'pdf_multilingual'}
            if subset != expected[source]:
                raise ValueError('unexpected replay family')
            category = 'replay'
            language = (provenance or {}).get('language', '')
            if not language:
                raise ValueError('replay language missing')
            if source == 'FinePDFs-EU-replay':
                pool += '/' + language
        else:
            raise ValueError('unknown mixture source: ' + mix_source)
    return category, pool, language


def stage_counts(tokens, trainable, context, overlap=OVERLAP):
    if context <= overlap + 1 or tokens < 1 or not 0 <= trainable < tokens:
        raise ValueError('invalid chunk accounting')
    n = 1 + max(0, (tokens - context + context - overlap - 2) // (context - overlap - 1))
    return n, tokens + (n - 1) * (overlap + 1), trainable


def metric(row, tokens, trainable, chars):
    provenance = json.loads(row['provenance_json']) if row['mix_source'].split('/')[0] in {
        'FinePDFs-EU-replay', 'FinePDFs-English-replay', 'dclm-baseline-1.0', 'Nemotron-CC-original-high'} else None
    category, pool, language = classify(row['mix_source'], row['kind'], provenance)
    if row['split'] not in {'train', 'validation'} or not 0 < trainable < tokens:
        raise ValueError('invalid split or no target tokens')
    reason = ('replay_over_4096' if category == 'replay' and tokens > 4096 else
              'long_pdf_under_32768' if row['mix_source'].startswith('FinePDFs-English-long/') and tokens < 32768 else '')
    result = {k: row[k] for k in ['id', 'mix_source', 'kind', 'split', 'split_group']}
    result.update(pool=pool, category=category, language=language, eligibility_reason=reason,
                  tokens=tokens, trainable_tokens=trainable, masked_tokens=tokens-trainable,
                  chars=chars, eligible=not reason)
    for context, minimum in STAGES:
        values = stage_counts(tokens, trainable, context) if not reason and (category == 'replay' or tokens >= minimum) else (0, 0, 0)
        result.update({f'stage_{context}_{k}': v for k, v in zip(['chunks', 'seen_tokens', 'trainable_tokens'], values)})
    return result


def prepare(args):
    out, step3, step2 = Path(args.out), Path(args.step3).resolve(), Path(args.step2).resolve()
    curation = json.loads((step3/'curation_manifest.json').read_text())
    if curation['status'] != 'step3-complete':
        raise ValueError('complete Step 3 verification is required')
    contract = json.loads((step2/'tokenizer_contract.json').read_text())
    for name, expected in contract['tokenizer_files_sha256'].items():
        if sha(step2/'tokenizer'/name) != expected:
            raise ValueError('tokenizer changed')
    adapter = json.loads((step2/'adapter_manifest.json').read_text())
    for name in CODE[1:]:
        if sha(Path(__file__).with_name(name)) != adapter['code_sha256']['scripts/data/'+name]:
            raise ValueError('step-2 counting dependency changed')
    tasks = [{'key': Path(p).parent.name, 'path': str(step3/p), 'sha256': checksum,
              'state': state(step3/p)} for p, checksum in sorted(curation['curated_parts_sha256'].items())]
    m = {'schema_version': 1, 'step3': str(step3), 'step2': str(step2),
         'curation_manifest_sha256': sha(step3/'curation_manifest.json'),
         'curation_build_id': curation['build_id'], 'decision_build_id': curation['decision_build_id'],
         'expected_rows': curation['retained_rows'], 'expected_counts': curation['retained_counts'],
         'tokenizer_contract_sha256': sha(step2/'tokenizer_contract.json'),
         'tokenizer_files_sha256': contract['tokenizer_files_sha256'], 'policy': POLICY,
         'code_sha256': {n: sha(Path(__file__).with_name(n)) for n in CODE}, 'tasks': tasks,
         'dependencies': {n: importlib.metadata.version(n) for n in ['pyarrow', 'tokenizers', 'transformers', 'numpy']},
         'python': sys.version, 'status': 'inputs-frozen'}
    m['build_id'] = digest(m)
    path = out/'census_input_manifest.json'
    if path.exists() and json.loads(path.read_text()) != m:
        raise ValueError('different census build; use another output directory')
    write(path, m)
    tools = out/'tools'
    tools.mkdir(exist_ok=True)
    for name in CODE:
        src, dst = Path(__file__).with_name(name), tools/name
        if not dst.exists():
            shutil.copy2(src, dst)
        if sha(dst) != m['code_sha256'][name]:
            raise ValueError('frozen code differs')
    print(dump({'build_id': m['build_id'], 'tasks': len(tasks), 'rows': m['expected_rows']}), flush=True)


def validate(out):
    m = json.loads((Path(out)/'census_input_manifest.json').read_text())
    identity = {k: v for k, v in m.items() if k != 'build_id'}
    if digest(identity) != m['build_id'] or m['policy'] != POLICY:
        raise ValueError('census manifest identity/policy changed')
    for name, expected in m['code_sha256'].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('frozen census code changed: '+name)
    step2 = Path(m['step2'])
    for path, expected in [(Path(m['step3'])/'curation_manifest.json', m['curation_manifest_sha256']),
                           (step2/'tokenizer_contract.json', m['tokenizer_contract_sha256'])]:
        if sha(path) != expected:
            raise ValueError('frozen upstream manifest changed')
    for name, expected in m['tokenizer_files_sha256'].items():
        if sha(step2/'tokenizer'/name) != expected:
            raise ValueError('frozen tokenizer changed')
    for name, expected in m['dependencies'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('census dependency changed: '+name)
    return m


def init_worker(out, m):
    global _MANIFEST, _OUT, _HF, _COUNTER
    from transformers import AutoTokenizer
    _MANIFEST, _OUT = m, Path(out)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    _HF = AutoTokenizer.from_pretrained(Path(m['step2'])/'tokenizer', local_files_only=True)
    _HF.model_max_length = 10**12
    _COUNTER = LFCountingTokenizer(_HF.backend_tokenizer, Path(m['step2'])/'tokenizer/tokenizer.json')


def process(spec):
    task, limit = spec
    root = _OUT/('pilot' if limit else 'census')/task['key']
    done = root/'stats.json'
    if done.exists():
        previous = json.loads(done.read_text())
        if previous['build_id'] != _MANIFEST['build_id'] or previous['limit'] != limit:
            raise ValueError('resume configuration changed')
        if sha(root/'metrics.parquet') != previous['metrics_sha256']:
            raise ValueError('completed metrics changed')
        return dict(previous, resumed=True)
    started = time.monotonic()
    if state(task['path']) != task['state'] or sha(task['path']) != task['sha256']:
        raise ValueError('canonical input changed: '+task['key'])
    root.mkdir(parents=True, exist_ok=True)
    counts, rows, pending, pending_chars = Counter(), [], [], 0
    columns = ['id', 'kind', 'mix_source', 'text', 'messages_json', 'tools_json',
               'provenance_json', 'split', 'split_group', 'curation_build_id']
    pf = pq.ParquetFile(task['path'])
    path = root/'metrics.parquet'
    with pq.ParquetWriter(str(path)+'.tmp', METRICS, compression='zstd') as writer:
        def add(value):
            rows.append(value)
            counts['rows'] += 1
            counts['tokens'] += value['tokens']
            counts['trainable_tokens'] += value['trainable_tokens']
            counts['eligible_rows'] += int(value['eligible'])
            if value['eligibility_reason']:
                counts[value['eligibility_reason']] += 1
            if len(rows) >= 4096:
                writer.write_table(pa.Table.from_pylist(rows, schema=METRICS))
                rows.clear()

        def flush_documents():
            nonlocal pending_chars
            if pending:
                lengths = _COUNTER.encode_batch([r['text'] for r in pending])
                for r, encoding in zip(pending, lengths):
                    n = len(encoding.ids) + 2
                    add(metric(r, n, n-1, len(r['text'])))
                pending.clear()
                pending_chars = 0

        seen = 0
        for batch in pf.iter_batches(batch_size=16, columns=columns, use_threads=False):
            for row in batch.to_pylist():
                if limit and seen >= limit:
                    break
                seen += 1
                if row['curation_build_id'] != _MANIFEST['decision_build_id']:
                    raise ValueError('canonical row curation identity changed')
                if row['kind'] == 'document':
                    pending.append(row)
                    pending_chars += len(row['text'])
                    if len(pending) >= 128 or pending_chars >= 2*1024**2:
                        flush_documents()
                elif row['kind'] == 'conversation':
                    messages = json.loads(row['messages_json'])
                    tools = json.loads(row['tools_json']) if row['tools_json'] else None
                    ids, loss = render_conversation(_HF, messages, tools)
                    add(metric(row, len(ids), sum(loss), len(row['messages_json'])))
                else:
                    raise ValueError('unknown canonical kind')
            if limit and seen >= limit:
                break
        flush_documents()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=METRICS))
    if seen != (min(limit, pf.metadata.num_rows) if limit else pf.metadata.num_rows):
        raise ValueError('canonical row coverage differs')
    if state(task['path']) != task['state']:
        raise ValueError('canonical input changed during census')
    Path(str(path)+'.tmp').replace(path)
    result = {'build_id': _MANIFEST['build_id'], 'key': task['key'], 'limit': limit,
              'input_sha256': task['sha256'], 'metrics_sha256': sha(path), 'counts': dict(counts),
              'seconds': time.monotonic()-started}
    write(done, result)
    return result


def partition(tasks, world):
    bins, weights = [[] for _ in range(world)], [0]*world
    for t in sorted(tasks, key=lambda t: (-t['state']['bytes'], t['key'])):
        rank = min(range(world), key=lambda i: (weights[i], i))
        bins[rank].append(t)
        weights[rank] += t['state']['bytes']
    return bins


def run(args):
    m = validate(args.out)
    tasks = m['tasks']
    if args.keys:
        wanted = set(args.keys.split(','))
        tasks = [t for t in tasks if t['key'] in wanted]
        if len(tasks) != len(wanted):
            raise ValueError('unknown task key')
    if not 0 <= args.rank < args.world:
        raise ValueError('invalid rank')
    tasks = partition(tasks, args.world)[args.rank]
    print(dump({'rank': args.rank, 'world': args.world, 'tasks': len(tasks), 'workers': args.workers}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=init_worker, initargs=(args.out, m)) as pool:
        futures = {pool.submit(process, (t, args.limit)): t['key'] for t in tasks}
        for f in as_completed(futures):
            r = f.result()
            print(dump({k: r[k] for k in ['key', 'counts', 'seconds']} | {'resumed': r.get('resumed', False)}), flush=True)


def summarize(args):
    """All metrics are checksum-verified; SQL aggregates lengths, never payloads."""
    import duckdb
    out = Path(args.out)
    m = validate(out)
    totals, checksums = Counter(), {}
    for task in m['tasks']:
        root = out/'census'/task['key']
        r = json.loads((root/'stats.json').read_text())
        if r['build_id'] != m['build_id'] or r['limit'] or r['input_sha256'] != task['sha256']:
            raise ValueError('census task identity differs')
        if sha(root/'metrics.parquet') != r['metrics_sha256']:
            raise ValueError('census metrics changed')
        if pq.ParquetFile(root/'metrics.parquet').metadata.num_rows != r['counts'].get('rows',0):
            raise ValueError('metrics row coverage differs')
        totals.update(r['counts'])
        checksums[task['key']] = r['metrics_sha256']
    if totals['rows'] != m['expected_rows']:
        raise ValueError('full census coverage differs')
    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute("SET memory_limit='128GB'")
    con.execute("SET parquet_metadata_cache=true")
    con.read_parquet(str(out/'census/*/metrics.parquet'), hive_partitioning=False).create_view('metrics')
    def query(sql):
        cur = con.execute(sql)
        keys = [x[0] for x in cur.description]
        return [dict(zip(keys, row)) for row in cur.fetchall()]
    coverage = query('SELECT mix_source,kind,split,count(*) AS rows FROM metrics GROUP BY ALL ORDER BY ALL')
    expected = sorted(m['expected_counts'], key=lambda r: (r['mix_source'], r['kind'], r['split']))
    if coverage != expected:
        raise ValueError('per-source/split coverage differs from Step 3')
    base = 'pool,category,kind,split,language,eligible,eligibility_reason'
    sums = ','.join(f'sum({n}) AS {n}' for n in ['tokens','trainable_tokens','masked_tokens']+
                    [f'stage_{c}_{n}' for c,_ in STAGES for n in ['chunks','seen_tokens','trainable_tokens']])
    groups = query(f'SELECT {base},count(*) AS rows,{sums},min(tokens) min_tokens,max(tokens) max_tokens '
                   f'FROM metrics GROUP BY {base} ORDER BY {base}')
    bins = query('SELECT pool,category,kind,split,eligible,CASE WHEN tokens<=4096 THEN \'0-4K\' '
                 'WHEN tokens<8192 THEN \'4-8K\' WHEN tokens<16384 THEN \'8-16K\' '
                 'WHEN tokens<32768 THEN \'16-32K\' WHEN tokens<=65536 THEN \'32-64K\' '
                 'ELSE \'>64K\' END length_bin,count(*) AS rows,sum(tokens) tokens,'
                 'sum(trainable_tokens) trainable_tokens FROM metrics GROUP BY ALL ORDER BY ALL')
    report = {'schema_version': 1, 'status': 'retained-census-complete', 'census_build_id': m['build_id'],
              'census_input_manifest_sha256': sha(out/'census_input_manifest.json'),
              'curation_manifest_sha256': m['curation_manifest_sha256'], 'counts': dict(totals),
              'policy': POLICY, 'groups': groups, 'length_bins': bins, 'source_row_coverage': coverage,
              'metrics_sha256': checksums,
              'checks': {'row_counts': True, 'source_split_coverage': True, 'checksums': True},
              'training_has_run': False, 'token_id_release_built': False}
    report['build_id'] = digest(report)
    write(out/'retained_census.json', report)
    print(dump({k: report[k] for k in ['status','build_id','counts']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare','run','summarize'])
    p.add_argument('--out', required=True)
    p.add_argument('--step3')
    p.add_argument('--step2')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world', type=int, default=1)
    p.add_argument('--keys')
    p.add_argument('--limit', type=int, default=0, help='pilot rows per part, isolated from full census')
    p.add_argument('--threads', type=int, default=64)
    a = p.parse_args()
    {'prepare':prepare, 'run':run, 'summarize':summarize}[a.command](a)


if __name__ == '__main__':
    main()
