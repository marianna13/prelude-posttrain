#!/usr/bin/env python3
"""Export the frozen Prelude parents as portable, masked, finite token shards.

prepare runs on a CPU node: locate selected IDs without reading text, verify the
upstream release, pin code/runtime, and partition work by bounded row-group ranges.
run owns disjoint tasks and resumes checksum-verified completed shards. finalize
independently reads every token/mask array and reconciles the frozen mixture.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, '/p/project1/laionize/marianna/dc_agent/.prelude_census_deps')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from census_curated_prelude_midtrain import digest, sha, state, write, stage_counts
from finish_prelude_retained_census import fast_render_conversation
from prelude_lf_counting import chunks, verify_barrier

CODE = ['export_prelude_midtrain.py', 'census_curated_prelude_midtrain.py',
        'finish_prelude_retained_census.py', 'prelude_midtrain.py', 'prelude_lf_counting.py']
POLICY = {
    'schema': 'prelude_ids_loss_v1', 'ids': 'int32', 'loss': 'int8, target-aligned before causal shift',
    'documents': 'original text tokenized at proven LF barriers; native BOS 1 / EOS 2',
    'conversations': 'frozen template; full ordinary targets; complete semantic tool groups masked',
    'continuation': {'overlap': 256, 'bos': 1, 'prefix_loss': 0, 'intermediate_eos': False,
                     'tails': 'keep all tokens, including windows with zero target tokens'},
    'selection': 'exact frozen step-4 training parents; no repetition, dropping, or resampling',
    'validation': 'held-out canonical partition remains frozen in step 3; not selected for this training export',
    'order': 'canonical task/row/chunk order on disk; finite global shuffle is a loader operation',
}
SCHEMA = pa.schema([(n, pa.string()) for n in
    ['id', 'parent_id', 'mix_source', 'pool', 'category', 'kind', 'split', 'split_group',
     'canonical_part', 'provenance_json', 'parent_ids_sha256', 'parent_loss_sha256']] +
    [(n, pa.int64()) for n in ['context', 'chunk_index', 'original_tokens', 'new_start', 'new_end',
                               'overlap_start', 'overlap_end']] +
    [('ids', pa.list_(pa.int32())), ('loss', pa.list_(pa.int8()))])
VERSIONS = ['pyarrow', 'tokenizers', 'transformers', 'numpy', 'duckdb', 'jinja2']
_M = _OUT = _TOK = None


def self_check(m):
    if m['build_id'] != digest({k: v for k, v in m.items() if k != 'build_id'}):
        raise ValueError('manifest identity changed')


def document_ids(tok, text):
    """Same native IDs as a whole document, with bounded BPE heaps at LF."""
    result = [np.array([1], dtype='<i4')]
    pending, chars = [], 0
    def flush():
        nonlocal chars
        if pending:
            result.extend(np.asarray(e.ids, dtype='<i4') for e in
                          tok.backend_tokenizer.encode_batch_fast(pending, add_special_tokens=False))
            pending.clear()
            chars = 0
    for piece in chunks(text):
        pending.append(piece)
        chars += len(piece)
        if len(pending) >= 128 or chars >= 8 * 1024**2:
            flush()
    flush()
    result.append(np.array([2], dtype='<i4'))
    ids = np.concatenate(result)
    loss = np.ones(len(ids), dtype=np.int8)
    loss[0] = 0
    return ids, loss


def token_chunks(ids, loss, context, overlap=256):
    """Consume every original token once; duplicated context and new BOS mask 0."""
    if context <= overlap + 1 or len(ids) != len(loss) or not len(ids):
        raise ValueError('invalid chunk input')
    if ids[0] != 1 or loss[0] != 0:
        raise ValueError('original must start with masked BOS 1')
    start, index = 0, 0
    while start < len(ids):
        if index == 0:
            end = min(context, len(ids))
            a = b = 0
            window_ids, window_loss = ids[:end], loss[:end]
        else:
            end = min(start + context - overlap - 1, len(ids))
            a, b = start - overlap, start
            window_ids = np.concatenate((np.array([1], dtype='<i4'), ids[a:b], ids[start:end]))
            window_loss = np.concatenate((np.zeros(overlap + 1, dtype=np.int8), loss[start:end]))
        yield index, start, end, a, b, window_ids, window_loss
        start, index = end, index + 1


def check_upstream(m):
    for p, expected in m['upstream_sha256'].items():
        if sha(p) != expected:
            raise ValueError('upstream artifact changed: ' + p)
    for name, expected in m['code_sha256'].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('export code changed: ' + name)
    for name, expected in m['dependencies'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('export runtime changed: ' + name)
    if sys.version != m['python']:
        raise ValueError('Python version changed')


def prepare(args):
    import duckdb
    base, out = Path(args.base).resolve(), Path(args.out).resolve()
    mixture_dir = Path(args.mixture).resolve() if args.mixture else base / 'step4/mixture_v3'
    mixture = json.loads((mixture_dir / 'mixture_manifest.json').read_text())
    census = json.loads((base / 'step4/census_input_manifest.json').read_text())
    retained = json.loads((base / 'step4/retained_census.json').read_text())
    curation = json.loads((base / 'step3/curation_manifest.json').read_text())
    for m in [mixture, census, retained, curation]:
        self_check(m)
    if mixture['status'] != 'step4-complete' or curation['status'] != 'step3-complete':
        raise ValueError('completed upstream steps required')
    if sha(base / 'step3/curation_manifest.json') != mixture['curation_manifest_sha256'] or \
       sha(base / 'step4/retained_census.json') != mixture['retained_census_sha256'] or \
       sha(base / 'step4/census_input_manifest.json') != retained['census_input_manifest_sha256']:
        raise ValueError('upstream lineage differs')
    contract_path = base / 'step2/tokenizer_contract.json'
    contract = json.loads(contract_path.read_text())
    upstream = {str(base / p): sha(base / p) for p in [
        'step3/curation_manifest.json', 'step4/census_input_manifest.json',
        'step4/retained_census.json', 'step2/tokenizer_contract.json', 'step1/source_selection.json',
        'step2/adapter_manifest.json']}
    upstream[str(mixture_dir / 'mixture_manifest.json')] = sha(mixture_dir / 'mixture_manifest.json')
    if sha(contract_path) != census['tokenizer_contract_sha256']:
        raise ValueError('tokenizer contract differs from census')
    for n, expected in contract['tokenizer_files_sha256'].items():
        p = base / 'step2/tokenizer' / n
        if sha(p) != expected:
            raise ValueError('tokenizer changed')
        upstream[str(p)] = expected
    for stage in mixture['stages']:
        p = mixture_dir / stage['selection']
        if sha(p) != stage['selection_sha256']:
            raise ValueError('selection changed')
        upstream[str(p)] = stage['selection_sha256']
    identity = dict(schema_version=1, base=str(base), policy=POLICY,
        upstream_sha256=upstream, mixture_build_id=mixture['build_id'],
        decision_build_id=census['decision_build_id'], groups_per_task=args.groups_per_task,
        stages=mixture['stages'], code_sha256={n: sha(Path(__file__).with_name(n)) for n in CODE},
        dependencies={n: importlib.metadata.version(n) for n in VERSIONS}, python=sys.version)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'export_input_manifest.json').exists():
        old = json.loads((out / 'export_input_manifest.json').read_text())
        if old['identity'] != identity:
            raise ValueError('different export build; choose another output directory')
        validate(out)
        print(json.dumps({'prepared': True, 'resumed': True, 'build_id': old['build_id']}), flush=True)
        return
    # Verify the complete canonical input once before tokenization; workers also
    # check file state before/after reading. No source data is rewritten.
    def verify_task(t):
        if state(t['path']) != t['state'] or sha(t['path']) != t['sha256']:
            raise ValueError('canonical input changed: ' + t['key'])
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        list(pool.map(verify_task, census['tasks']))
    print(json.dumps({'canonical_checksums_verified': len(census['tasks'])}), flush=True)
    con = duckdb.connect()
    con.execute(f'SET threads={args.threads}')
    con.execute("SET memory_limit='96GB'")
    con.read_parquet([str(mixture_dir / s['selection']) for s in mixture['stages']],
                     hive_partitioning=False).create_view('selected')
    count, unique = con.execute('SELECT count(*),count(DISTINCT id) FROM selected').fetchone()
    if count != unique or count != sum(s['source_counts'][p].get('parents', 0)
                                    for s in mixture['stages'] for p in s['source_counts']):
        raise ValueError('repeated or missing selected parent')
    con.read_parquet([t['path'] for t in census['tasks']], filename=True, file_row_number=True,
                     hive_partitioning=False).create_view('canonical')
    con.execute('''CREATE TABLE located AS SELECT s.*,c.filename,c.file_row_number
                   FROM selected s JOIN canonical c USING(id)''')
    if con.execute('SELECT count(*) FROM located').fetchone()[0] != count or \
       con.execute('SELECT count(DISTINCT id) FROM located').fetchone()[0] != count:
        raise ValueError('missing or duplicated canonical locator')
    con.execute('CREATE INDEX located_filename ON located(filename)')
    print(json.dumps({'located_parents': count}), flush=True)
    selections = out / 'selections'
    selections.mkdir(exist_ok=True)
    by_path = {t['path']: t for t in census['tasks']}
    tasks, mapping_hashes = [], {}
    for filename, in con.execute('SELECT DISTINCT filename FROM located ORDER BY filename').fetchall():
        original = by_path[filename]
        table = con.execute('SELECT * EXCLUDE(filename) FROM located WHERE filename=? ORDER BY file_row_number',
                            [filename]).fetch_arrow_table()
        pf = pq.ParquetFile(filename)
        offsets, total = [], 0
        for group in range(pf.num_row_groups):
            offsets.append(total)
            total += pf.metadata.row_group(group).num_rows
        buckets = defaultdict(list)
        for row in table.to_pylist():
            g = bisect_right(offsets, row['file_row_number']) - 1
            if g < 0 or row['file_row_number'] >= total:
                raise ValueError('invalid canonical row locator')
            buckets[g // args.groups_per_task].append((g, row))
        for bucket, values in sorted(buckets.items()):
            key = original['key'] + f'-g{bucket:05d}'
            rel = f'selections/{key}.parquet'
            pq.write_table(pa.Table.from_pylist([v for _, v in values], schema=table.schema),
                           out / rel, compression='zstd')
            mapping_hashes[rel] = sha(out / rel)
            tasks.append(dict(key=key, source=original, groups=sorted({g for g, _ in values}),
                selection=rel, parents=len(values), seen_tokens=sum(v['seen_tokens'] for _, v in values)))
    con.close()
    m = dict(identity=identity, selection_sha256=mapping_hashes, tasks=tasks,
             canonical_checksums_verified=len(census['tasks']))
    m['build_id'] = digest(m)
    tools = out / 'tools'
    tools.mkdir(exist_ok=True)
    for n in CODE:
        dst = tools / n
        if dst.exists() and sha(dst) != identity['code_sha256'][n]:
            raise ValueError('frozen tools changed')
        shutil.copy2(Path(__file__).with_name(n), dst)
    release = out / 'release'
    release.mkdir(exist_ok=True)
    shutil.copytree(base / 'step2/tokenizer', release / 'tokenizer', dirs_exist_ok=True)
    shutil.copy2(contract_path, release / 'tokenizer_contract.json')
    provenance = release / 'provenance'
    provenance.mkdir(exist_ok=True)
    for p in ['step1/source_selection.json', 'step2/adapter_manifest.json', 'step3/curation_manifest.json',
              'step4/retained_census.json']:
        shutil.copy2(base / p, provenance / Path(p).name)
    shutil.copy2(mixture_dir / 'mixture_manifest.json', provenance / 'mixture_manifest.json')
    for stage in mixture['stages']:
        shutil.copy2(mixture_dir / stage['selection'], provenance / stage['selection'])
    write(out / 'export_input_manifest.json', m)
    print(json.dumps({'build_id': m['build_id'], 'tasks': len(tasks), 'parents': count}), flush=True)


def validate(out):
    m = json.loads((Path(out) / 'export_input_manifest.json').read_text())
    self_check(m)
    if m['identity']['policy'] != POLICY:
        raise ValueError('export policy changed')
    check_upstream(m['identity'])
    return m


def init_worker(out, m):
    global _M, _OUT, _TOK
    from transformers import AutoTokenizer
    _M, _OUT = m, Path(out)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    folder = Path(m['identity']['base']) / 'step2/tokenizer'
    _TOK = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    _TOK.model_max_length = 10**12
    verify_barrier(_TOK.backend_tokenizer, folder / 'tokenizer.json')


def process(task):
    started = time.monotonic()
    done = _OUT / 'work' / task['key'] / 'stats.json'
    if done.exists():
        old = json.loads(done.read_text())
        if old['build_id'] != _M['build_id'] or old['key'] != task['key']:
            raise ValueError('resume identity differs')
        for rel, expected in old['artifacts_sha256'].items():
            if sha(_OUT / 'release' / rel) != expected:
                raise ValueError('completed export changed: ' + rel)
        return dict(old, resumed=True)
    source = task['source']
    if state(source['path']) != source['state']:
        raise ValueError('canonical input state changed')
    selection = _OUT / task['selection']
    if sha(selection) != _M['selection_sha256'][task['selection']]:
        raise ValueError('selection changed')
    selected = {r['id']: r for r in pq.read_table(selection).to_pylist()}
    writers, buffers, artifacts, counts = {}, defaultdict(list), {}, defaultdict(Counter)
    def flush(context):
        if buffers[context]:
            writers[context].write_table(pa.Table.from_pylist(buffers[context], schema=SCHEMA))
            buffers[context].clear()
    try:
        pf = pq.ParquetFile(source['path'])
        for batch in pf.iter_batches(batch_size=8, row_groups=task['groups'], use_threads=False):
            for row in batch.to_pylist():
                selection = selected.pop(row['id'], None)
                if selection is None:
                    continue
                if row['curation_build_id'] != _M['identity']['decision_build_id']:
                    raise ValueError('canonical decision identity differs')
                for key in ['kind', 'mix_source', 'split', 'split_group']:
                    if row[key] != selection[key]:
                        raise ValueError('canonical selection differs: ' + key)
                if row['split'] != 'train':
                    raise ValueError('held-out leakage')
                if row['kind'] == 'document':
                    ids, loss = document_ids(_TOK, row['text'])
                else:
                    ids, loss = fast_render_conversation(_TOK, json.loads(row['messages_json']),
                        json.loads(row['tools_json']) if row['tools_json'] else None)
                    ids, loss = np.asarray(ids, dtype='<i4'), np.asarray(loss, dtype=np.int8)
                if len(ids) != selection['original_tokens'] or int(loss.sum()) != selection['trainable_tokens']:
                    raise ValueError('tokenization differs from step-4 parent counts: ' + row['id'])
                if np.any(ids < 0) or np.any(ids >= 262151) or not np.isin(loss, [0, 1]).all():
                    raise ValueError('invalid token or mask')
                if row['kind'] == 'document' and ids[-1] != 2:
                    raise ValueError('document EOS differs')
                ids_hash = hashlib.sha256(ids.tobytes()).hexdigest()
                loss_hash = hashlib.sha256(loss.tobytes()).hexdigest()
                context = selection['context']
                if context not in writers:
                    rel = f'train/{context}/{task["key"]}.parquet'
                    path = _OUT / 'release' / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    artifacts[context] = rel
                    writers[context] = pq.ParquetWriter(str(path) + '.tmp', SCHEMA, compression='zstd')
                chunks_seen = tokens_seen = targets_seen = 0
                for index, start, end, a, b, window_ids, window_loss in token_chunks(ids, loss, context):
                    value = {k: selection[k] for k in ['mix_source', 'pool', 'category', 'kind', 'split', 'split_group']}
                    value.update(id=f'{row["id"]}:c{context}:{index}', parent_id=row['id'], context=context,
                        chunk_index=index, original_tokens=len(ids), new_start=start, new_end=end,
                        overlap_start=a, overlap_end=b, canonical_part=source['key'],
                        provenance_json=row['provenance_json'], parent_ids_sha256=ids_hash,
                        parent_loss_sha256=loss_hash, ids=window_ids, loss=window_loss)
                    buffers[context].append(value)
                    chunks_seen += 1
                    tokens_seen += len(window_ids)
                    targets_seen += int(window_loss.sum())
                    group = counts[f'{context}|{selection["pool"]}']
                    group['chunks'] += 1
                    group['seen_tokens'] += len(window_ids)
                    group['trainable_tokens'] += int(window_loss.sum())
                    group['zero_target_chunks'] += int(not window_loss.any())
                    group[f'length_{min(16, (len(window_ids)-1).bit_length())}'] += 1
                    if len(buffers[context]) >= 64 or sum(len(v['ids']) for v in buffers[context]) >= 262144:
                        flush(context)
                if (chunks_seen, tokens_seen, targets_seen) != tuple(selection[k] for k in ['chunks', 'seen_tokens', 'trainable_tokens']):
                    raise ValueError('chunk counts differ from frozen selection')
                counts[f'{context}|{selection["pool"]}']['parents'] += 1
                counts[f'{context}|{selection["pool"]}']['original_tokens'] += len(ids)
        if selected:
            raise ValueError('selected parents not covered')
        for context in writers:
            flush(context)
            writers[context].close()
        if state(source['path']) != source['state']:
            raise ValueError('canonical input changed during export')
        for rel in artifacts.values():
            path = _OUT / 'release' / rel
            Path(str(path) + '.tmp').replace(path)
        result = dict(build_id=_M['build_id'], key=task['key'],
            counts={k: dict(v) for k, v in counts.items()},
            artifacts_sha256={p: sha(_OUT / 'release' / p) for p in artifacts.values()},
            seconds=time.monotonic() - started)
        write(done, result)
        return result
    except BaseException:
        for writer in writers.values():
            writer.close()
        raise


def run(args):
    m = validate(args.out)
    if not 0 <= args.rank < args.world:
        raise ValueError('invalid rank')
    tasks = m['tasks']
    if args.keys:
        wanted = set(args.keys.split(','))
        tasks = [t for t in tasks if t['key'] in wanted]
        if {t['key'] for t in tasks} != wanted:
            raise ValueError('unknown task keys')
    bins, sizes = [[] for _ in range(args.world)], [0] * args.world
    for t in sorted(tasks, key=lambda t: (-t['seen_tokens'], t['key'])):
        i = min(range(args.world), key=lambda i: (sizes[i], i))
        bins[i].append(t)
        sizes[i] += t['seen_tokens']
    tasks = bins[args.rank]
    print(json.dumps({'rank': args.rank, 'tasks': len(tasks), 'seen_tokens': sizes[args.rank]}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=init_worker, initargs=(args.out, m)) as pool:
        futures = {pool.submit(process, t): t['key'] for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if i % 25 == 0 or i == len(tasks):
                print(json.dumps({'done': i, 'total': len(tasks), 'key': result['key'],
                                  'seconds': result['seconds'], 'resumed': result.get('resumed', False)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--base', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--threads', type=int, default=32)
    a.add_argument('--groups-per-task', type=int, default=128)
    a.add_argument('--mixture', help='mixture directory (default BASE/step4/mixture_v3)')
    a.set_defaults(func=prepare)
    a = sub.add_parser('run')
    a.add_argument('--out', required=True)
    a.add_argument('--workers', type=int, default=32)
    a.add_argument('--rank', type=int, default=0)
    a.add_argument('--world', type=int, default=1)
    a.add_argument('--keys')
    a.set_defaults(func=run)
    args = p.parse_args()
    if getattr(args, 'groups_per_task', 1) < 1 or getattr(args, 'threads', 1) < 1:
        p.error('positive resources required')
    args.func(args)


if __name__ == '__main__':
    main()
