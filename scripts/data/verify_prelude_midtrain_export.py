#!/usr/bin/env python3
"""Independent full read-back of Prelude IDs, masks and finite chunk coverage."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys

sys.path.insert(0, '/p/project1/laionize/marianna/dc_agent/.prelude_census_deps')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pyarrow.parquet as pq
import export_prelude_midtrain as e


def audit_task(spec):
    out, m, task = spec
    out = Path(out)
    stats_path = out / 'work' / task['key'] / 'stats.json'
    stats = json.loads(stats_path.read_text())
    if stats['build_id'] != m['build_id'] or stats['key'] != task['key']:
        raise ValueError('export task identity differs')
    selected_path = out / task['selection']
    if e.sha(selected_path) != m['selection_sha256'][task['selection']]:
        raise ValueError('selection changed')
    selected = {r['id']: r for r in pq.read_table(selected_path).to_pylist()}
    if len(selected) != task['parents']:
        raise ValueError('selection parent count differs')
    counts, smoke_rows = defaultdict(Counter), []
    for rel, checksum in stats['artifacts_sha256'].items():
        path = out / 'release' / rel
        if e.sha(path) != checksum:
            raise ValueError('export checksum differs')
        pf = pq.ParquetFile(path)
        if pf.schema_arrow != e.SCHEMA:
            raise ValueError('export schema differs')
        current = None
        candidate_pools, candidate_continuation, candidate_zero, row_index = set(), False, False, 0
        def finish_parent():
            if current is None:
                return
            original = current['selection']
            if current['end'] != original['original_tokens'] or \
               current['index'] != original['chunks'] or \
               current['seen'] != original['seen_tokens'] or \
               current['trained'] != original['trainable_tokens'] or \
               current['ids_hash'].hexdigest() != current['expected_ids_hash'] or \
               current['loss_hash'].hexdigest() != current['expected_loss_hash']:
                raise ValueError('reconstructed parent differs: ' + original['id'])
            group = counts[f'{original["context"]}|{original["pool"]}']
            group['parents'] += 1
            group['original_tokens'] += original['original_tokens']
        for batch in pf.iter_batches(batch_size=8, use_threads=False):
            for row in batch.to_pylist():
                parent = row['parent_id']
                if current is None or parent != current['selection']['id']:
                    finish_parent()
                    if parent not in selected:
                        raise ValueError('repeated or unselected parent')
                    original = selected.pop(parent)
                    current = dict(selection=original, index=0, end=0, seen=0, trained=0,
                        ids_hash=hashlib.sha256(), loss_hash=hashlib.sha256(), tail=None,
                        expected_ids_hash=row['parent_ids_sha256'], expected_loss_hash=row['parent_loss_sha256'])
                original = current['selection']
                context, index = original['context'], current['index']
                if any(row[k] != original[k] for k in ['mix_source', 'pool', 'category', 'kind', 'split',
                                                       'split_group', 'context', 'original_tokens']):
                    raise ValueError('chunk metadata differs from selected parent')
                if row['id'] != f'{parent}:c{context}:{index}' or row['chunk_index'] != index or \
                   row['canonical_part'] != task['source']['key'] or row['split'] != 'train':
                    raise ValueError('chunk identity differs')
                ids, loss = np.asarray(row['ids'], dtype='<i4'), np.asarray(row['loss'], dtype=np.int8)
                if ids.shape != loss.shape or not 0 < len(ids) <= context or ids[0] != 1 or loss[0] != 0 or \
                   ids.min() < 0 or ids.max() >= 262151 or not np.isin(loss, [0, 1]).all():
                    raise ValueError('invalid token IDs, shape or target mask')
                start = current['end']
                end = min(start + context - (257 if index else 0), original['original_tokens'])
                if (row['new_start'], row['new_end'], row['overlap_start'], row['overlap_end']) != \
                   (start, end, start - 256 if index else 0, start if index else 0):
                    raise ValueError('new-token interval or overlap differs')
                prefix = 257 if index else 0
                if len(ids) != prefix + end - start:
                    raise ValueError('chunk length differs')
                if index and (not np.array_equal(ids[1:257], current['tail']) or loss[:257].any()):
                    raise ValueError('continuation prefix IDs or mask differ')
                if row['parent_ids_sha256'] != current['expected_ids_hash'] or \
                   row['parent_loss_sha256'] != current['expected_loss_hash']:
                    raise ValueError('parent digest changed between chunks')
                current['ids_hash'].update(ids[prefix:].tobytes())
                current['loss_hash'].update(loss[prefix:].tobytes())
                current['tail'] = ids[-256:]
                current['end'], current['index'] = end, index + 1
                current['seen'] += len(ids)
                current['trained'] += int(loss.sum())
                if end == original['original_tokens'] and row['kind'] == 'document' and ids[-1] != 2:
                    raise ValueError('native document EOS differs')
                group = counts[f'{context}|{row["pool"]}']
                group['chunks'] += 1
                group['seen_tokens'] += len(ids)
                group['trainable_tokens'] += int(loss.sum())
                group['zero_target_chunks'] += int(not loss.any())
                group[f'length_{min(16, (len(ids)-1).bit_length())}'] += 1
                zero = not loss.any()
                if row['pool'] not in candidate_pools or (index and not candidate_continuation) or (zero and not candidate_zero):
                    smoke_rows.append(dict(file=rel, row=row_index, id=row['id'], context=context,
                        pool=row['pool'], continuation=bool(index), zero_target=bool(zero)))
                    candidate_pools.add(row['pool'])
                    candidate_continuation |= bool(index)
                    candidate_zero |= bool(zero)
                row_index += 1
        finish_parent()
    if selected or {k: dict(v) for k, v in counts.items()} != stats['counts']:
        raise ValueError('read-back counts differ from export')
    result = dict(status='passed', build_id=m['build_id'], key=task['key'],
                  counts={k: dict(v) for k, v in counts.items()},
                  stats_sha256=e.sha(stats_path), artifacts_sha256=stats['artifacts_sha256'],
                  verifier_sha256=e.sha(__file__), smoke_rows=smoke_rows)
    e.write(out / 'work' / task['key'] / 'audit.json', result)
    return result


def audit(args):
    m = e.validate(args.out)
    tasks = m['tasks']
    if args.keys:
        wanted = set(args.keys.split(','))
        tasks = [t for t in tasks if t['key'] in wanted]
        if len(tasks) != len(wanted):
            raise ValueError('unknown task key')
    if not 0 <= args.rank < args.world:
        raise ValueError('invalid rank')
    bins, weights = [[] for _ in range(args.world)], [0] * args.world
    for t in sorted(tasks, key=lambda t: (-t['seen_tokens'], t['key'])):
        i = min(range(args.world), key=lambda i: (weights[i], i))
        bins[i].append(t)
        weights[i] += t['seen_tokens']
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(audit_task, (args.out, m, task)) for task in bins[args.rank]]
        for i, f in enumerate(as_completed(futures), 1):
            result = f.result()
            if i % 100 == 0 or i == len(futures):
                print(json.dumps({'rank': args.rank, 'verified': i, 'total': len(futures), 'key': result['key']}), flush=True)


def finalize(args):
    out = Path(args.out)
    m = e.validate(out)
    counts, artifacts = defaultdict(Counter), {}
    smoke_rows, smoke_pools, smoke_continuations, smoke_zeros = [], defaultdict(set), set(), set()
    for task in m['tasks']:
        root = out / 'work' / task['key']
        result = json.loads((root / 'audit.json').read_text())
        if result['build_id'] != m['build_id'] or result['status'] != 'passed' or \
           result['key'] != task['key'] or result['verifier_sha256'] != e.sha(__file__) or \
           result['stats_sha256'] != e.sha(root / 'stats.json'):
            raise ValueError('missing or stale task audit')
        if artifacts.keys() & result['artifacts_sha256'].keys():
            raise ValueError('repeated artifact')
        artifacts.update(result['artifacts_sha256'])
        for k, v in result['counts'].items():
            counts[k].update(v)
        for row in result['smoke_rows']:
            c = row['context']
            if row['pool'] not in smoke_pools[c] or (row['continuation'] and c not in smoke_continuations) or \
               (row['zero_target'] and c not in smoke_zeros):
                smoke_rows.append(row)
                smoke_pools[c].add(row['pool'])
                if row['continuation']:
                    smoke_continuations.add(c)
                if row['zero_target']:
                    smoke_zeros.add(c)
    stages = []
    for stage in m['identity']['stages']:
        context = stage['context']
        actual = {p: dict(counts.get(f'{context}|{p}', Counter())) for p in stage['source_counts']}
        for p, expected in stage['source_counts'].items():
            for k in ['parents', 'chunks', 'original_tokens', 'seen_tokens', 'trainable_tokens']:
                if actual[p].get(k, 0) != expected.get(k, 0):
                    raise ValueError(f'frozen source exposure differs: {context} {p} {k}')
        stages.append(dict(context=context, source_counts=actual,
                           **{k: sum(v.get(k, 0) for v in actual.values()) for k in
                              ['parents', 'chunks', 'original_tokens', 'seen_tokens', 'trainable_tokens', 'zero_target_chunks']}))
    expected_keys = {f'{s["context"]}|{p}' for s in m['identity']['stages'] for p in s['source_counts']}
    if counts.keys() - expected_keys:
        raise ValueError('unrecognized source or stage')
    release = out / 'release'
    disk_shards = {str(p.relative_to(release)) for p in (release / 'train').rglob('*.parquet')}
    if disk_shards != set(artifacts):
        raise ValueError('untracked or missing shards in release')
    shutil.copy2(out / 'export_input_manifest.json', release / 'provenance/export_input_manifest.json')
    tools = release / 'tools'
    tools.mkdir(exist_ok=True)
    for name in e.CODE:
        shutil.copy2(Path(e.__file__).with_name(name), tools / name)
    shutil.copy2(__file__, tools / Path(__file__).name)
    # The native Levanter import/runtime is bundled separately after its CPU smoke.
    for folder in ['tokenizer', 'provenance', 'tools']:
        for p in (release / folder).rglob('*'):
            if p.is_file():
                artifacts[str(p.relative_to(release))] = e.sha(p)
    artifacts['tokenizer_contract.json'] = e.sha(release / 'tokenizer_contract.json')
    total = {k: sum(s[k] for s in stages) for k in
             ['parents', 'chunks', 'original_tokens', 'seen_tokens', 'trainable_tokens', 'zero_target_chunks']}
    result = dict(schema_version=1, status='token-export-verified', export_build_id=m['build_id'],
        mixture_build_id=m['identity']['mixture_build_id'], policy=e.POLICY, stages=stages, totals=total,
        artifacts_sha256=artifacts, artifact_bytes=sum((release / p).stat().st_size for p in artifacts),
        dependencies=m['identity']['dependencies'], python=m['identity']['python'], loader_smoke_rows=smoke_rows,
        checks={'all_parent_selections_exact': True, 'read_back_every_token_and_mask': True,
                'parent_ids_and_loss_reconstructed': True, 'continuation_overlap_and_mask': True,
                'all_source_stage_counts_match': True, 'no_implicit_truncation_or_resampling': True},
        loader_cpu_smoke=False, mn5_gpu_smoke=False, training_has_run=False)
    result['build_id'] = e.digest(result)
    e.write(release / 'release_manifest.json', result)
    shas = dict(artifacts, **{'release_manifest.json': e.sha(release / 'release_manifest.json')})
    (release / 'SHA256SUMS').write_text(''.join(f'{v}  {p}\n' for p, v in sorted(shas.items())))
    print(json.dumps({'status': result['status'], 'build_id': result['build_id'], 'totals': total,
                      'artifact_bytes': result['artifact_bytes']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--out', required=True)
    a.add_argument('--rank', type=int, default=0)
    a.add_argument('--world', type=int, default=1)
    a.add_argument('--workers', type=int, default=16)
    a.add_argument('--keys')
    a.set_defaults(func=audit)
    a = sub.add_parser('finalize')
    a.add_argument('--out', required=True)
    a.set_defaults(func=finalize)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
