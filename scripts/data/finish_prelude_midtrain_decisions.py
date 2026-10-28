#!/usr/bin/env python3
"""Finish checksums for an already exported Prelude decision build in parallel.

The frozen reducer must have completed its SQL assertions, split export and
review-policy copy. Recount every exported row and reproduce its exact manifest.
This can run alongside the reducer's serial checksum tail: both write identical
manifest bytes, and existing manifests must compare equal before reuse.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curate_prelude_midtrain import digest, dump, sha, validate, write
from reduce_prelude_midtrain import check_complete


def inspect_part(path):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    path = Path(path)
    if path.stat().st_size <= 32 * 1024**2:
        payload = path.read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        source = pa.BufferReader(payload)
    else:
        checksum = sha(path)
        source = path
    table = pq.ParquetFile(source).read(columns=['id', 'source', 'subset', 'reason', 'split'])
    if table['id'].null_count:
        raise RuntimeError('null decision ID')
    keys = ['source', 'subset', 'reason', 'split']
    counts = table.group_by(keys, use_threads=False).aggregate([('id', 'count')]).to_pylist()
    allowed = {'keep', 'reserved_sft_task', 'benchmark_match', 'covered_by_stack', 'exact_duplicate'}
    for row in counts:
        if row['reason'] not in allowed or row['split'] not in ('train', 'validation'):
            raise RuntimeError('invalid exported decision')
        row['rows'] = row.pop('id_count')
    if sum(row['rows'] for row in counts) != table.num_rows:
        raise RuntimeError('decision count differs from exported rows')
    return str(path), checksum, counts


def finish(out, *, workers=16, pilot=False):
    out = Path(out)
    manifest = validate(out)
    _, reports, totals, pins = check_complete(out, manifest, pilot, False)
    root = out / ('pilot_reduction' if pilot else 'reduction')
    summary = json.loads((root / 'scan_summary.json').read_text())
    reducer_hash = sha(Path(__file__).with_name('reduce_prelude_midtrain.py'))
    if summary['input_manifest_sha256'] != sha(out / 'input_manifest.json'):
        raise RuntimeError('input manifest changed since review')
    if summary['task_report_sha256'] != pins or summary['reducer_sha256'] != reducer_hash:
        raise RuntimeError('scan or reducer changed since review')
    # This copy is written only after the original reducer's SQL invariants and
    # complete split export. Its presence alone is insufficient: reconcile rows.
    policy_path = root / 'benchmark_exclusions.json'
    policy = json.loads(policy_path.read_text())
    if policy['scan_build_id'] != manifest['build_id'] or policy['status'] != 'reviewed':
        raise RuntimeError('different or unreviewed exclusion policy')
    if policy.get('scope') != ('pilot' if pilot else 'full') or policy.get('frequency_cutoff') is not None:
        raise RuntimeError('different review scope or automatic frequency exclusion')
    expected = {key for key, report in reports.items()
                if not key.startswith('reference-') and report['counts'].get('accepted_rows', 0)}
    directories = {p.name.removeprefix('task_key=') for p in (root / 'decisions').iterdir() if p.is_dir()}
    if directories != expected:
        raise RuntimeError('decision partitions do not cover accepted source units')
    files = sorted((root / 'decisions').rglob('*.parquet'))
    file_hashes = {}
    counts = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(inspect_part, path) for path in files]
        for i, future in enumerate(as_completed(pending), 1):
            path, checksum, rows = future.result()
            file_hashes[str(Path(path).relative_to(root))] = checksum
            for row in rows:
                counts[tuple(row[k] for k in ('source', 'subset', 'reason', 'split'))] += row['rows']
            if i % 10000 == 0:
                print(json.dumps({'verified_files': i, 'total_files': len(files)}), flush=True)
    if sum(counts.values()) != totals['scan']['accepted_rows']:
        raise RuntimeError('exported decisions differ from accepted scan rows')
    retained = sum(n for key, n in counts.items() if key[2] == 'keep')
    if pq.ParquetFile(root / 'split_assignments.parquet').metadata.num_rows != retained:
        raise RuntimeError('split assignment count differs from retained decisions')
    overlap = [dict(zip(('source', 'subset', 'reason', 'split'), key), rows=n)
               for key, n in sorted(counts.items())]
    result = {'schema_version': 1, 'status': 'decisions-frozen',
              'scan_build_id': manifest['build_id'],
              'scan_summary_sha256': sha(root / 'scan_summary.json'),
              'exclusions_sha256': sha(policy_path), 'reducer_sha256': reducer_hash,
              'policy': manifest['policy'], 'counts_by_source': overlap,
              'task_components': pq.ParquetFile(root / 'task_components.parquet').metadata.num_rows,
              'scope': 'pilot' if pilot else 'full', 'decision_files_sha256': file_hashes,
              'artifacts_sha256': {name: sha(root / name) for name in
                  ('task_components.parquet', 'content_id_conflicts.parquet',
                   'split_assignments.parquet', 'benchmark_exclusions.json')}}
    result['build_id'] = digest(dump(result))
    destination = root / 'decision_manifest.json'
    if destination.exists() and json.loads(destination.read_text()) != result:
        raise RuntimeError('parallel completion differs from existing decision manifest')
    write(destination, result)
    write(root / 'parallel_completion.json', {
        'status': 'decision manifest verified', 'build_id': result['build_id'],
        'helper_sha256': sha(__file__), 'reducer_sha256': reducer_hash,
        'workers': workers, 'verified_files': len(files),
        'accepted_rows': sum(counts.values()), 'retained_rows': retained,
        'decision_manifest_sha256': sha(destination)})
    print(json.dumps({'decision_build_id': result['build_id'], 'retained_rows': retained,
                      'verified_files': len(files)}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    finish(args.out, workers=args.workers, pilot=args.pilot)
