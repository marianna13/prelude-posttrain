#!/usr/bin/env python3
"""Build the published data recipe: every training item of mid-training stages 1-4 and SFT v2 as a pointer to its
upstream record (repository, revision, file, row or byte offset, record hash). No upstream text is copied.

Runs where the original build lives (JURECA). Inputs are the frozen artifacts of the build:
  mid-training  step4/mixture_v3/selection-{16384,32768,65536}.parquet, stage4_128k/mixture/selection-131072.parquet,
                step3/curated/*/rows.parquet (id, provenance_json)
  SFT v2        selection/<source>.npy (census rows), normalized/ + rendered/ parts, loop_scan/full/flagged_turns.jsonl,
                the SFT v1 normalized parts (thinking-off rows)

Writes OUT/midtrain/stage{1..4}_*.parquet, OUT/sft_v2/<mix_source>.parquet and OUT/manifests/.
  python build_recipe.py --out OUT [--procs 64] [--only midtrain|sft]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MID = '/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1'
SFT_ROOT = '/p/data1/mmlaion/marianna/sft_sources'
V2 = '/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v2'
V1 = '/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v1'
STAGES = [('stage1_16k', f'{MID}/step4/mixture_v3/selection-16384.parquet'),
          ('stage2_32k', f'{MID}/step4/mixture_v3/selection-32768.parquet'),
          ('stage3_64k', f'{MID}/step4/mixture_v3/selection-65536.parquet'),
          ('stage4_128k', f'{MID}/stage4_128k/mixture/selection-131072.parquet')]
PROV = ['source', 'hf_repo', 'revision', 'relative_path', 'sha256', 'family', 'language', 'locator_kind', 'locator',
        'record_sha256']
WANTED = None


def scan_part(path):
    """(id, provenance fields, files json, provenance json) for the wanted ids of one curated part."""
    t = pq.read_table(path, columns=['id'])
    ids = t.column(0).to_pylist()
    hit = [i for i, x in enumerate(ids) if x in WANTED]
    if not hit:
        return []
    prov = pq.read_table(path, columns=['provenance_json']).column(0).take(pa.array(hit)).to_pylist()
    out = []
    for i, p in zip(hit, prov):
        d = json.loads(p)
        files = [{k: f.get(k) for k in ('file_path', 'content_id', 'license_type')} for f in d.get('files') or []]
        out.append((ids[i], {k: d.get(k) for k in PROV}, json.dumps(files) if files else None, p))
    return out


RELEASES = [f'{MID}/step5/release/train/*/*.parquet', f'{MID}/stage4_128k/export/release/train/*/*.parquet']


def scan_release(path):
    """(parent id, context) -> (ids sha256, loss sha256) of the parent's full token sequence, from the export."""
    t = pq.read_table(path, columns=['parent_id', 'context', 'chunk_index', 'parent_ids_sha256', 'parent_loss_sha256'])
    d = t.to_pydict()
    return [((p, c), (a, b)) for p, c, k, a, b in zip(d['parent_id'], d['context'], d['chunk_index'],
                                                         d['parent_ids_sha256'], d['parent_loss_sha256']) if k == 0]


def midtrain(out, procs):
    global WANTED
    sels = {name: pq.read_table(path) for name, path in STAGES}
    WANTED = set().union(*(set(t.column('id').to_pylist()) for t in sels.values()))
    parts = sorted(glob.glob(f'{MID}/step3/curated/*/rows.parquet'))
    print(f'{len(WANTED)} selected parents, scanning {len(parts)} curated parts', flush=True)
    found = {}
    with Pool(procs) as pool:
        for n, rows in enumerate(pool.imap_unordered(scan_part, parts, chunksize=4)):
            for i, prov, files, raw in rows:
                found[i] = (prov, files, raw)
            if n % 500 == 0:
                print(f'  {n}/{len(parts)} parts, {len(found)} found', flush=True)
    hashes = {}
    rel = sorted(f for g in RELEASES for f in glob.glob(g))
    print(f'reading token hashes from {len(rel)} release files', flush=True)
    with Pool(procs) as pool:
        for rows in pool.imap_unordered(scan_release, rel, chunksize=8):
            hashes.update(rows)
    missing = WANTED - found.keys()
    if missing:
        raise SystemExit(f'{len(missing)} selected ids not in step3/curated, e.g. {sorted(missing)[:3]}')
    os.makedirs(f'{out}/midtrain', exist_ok=True)
    for name, t in sels.items():
        ids = t.column('id').to_pylist()
        cols = {c: t.column(c) for c in t.schema.names}
        for k in PROV:
            col = 'file_sha256' if k == 'sha256' else k
            cols[col] = pa.array([found[i][0][k] for i in ids],
                                 pa.int64() if k == 'locator' else pa.string())
        ctx = t.column('context').to_pylist()
        h = [hashes.get((i, c)) for i, c in zip(ids, ctx)]
        if any(x is None for x in h):
            raise SystemExit(f'{name}: {sum(x is None for x in h)} parents without release token hashes')
        cols['ids_sha256'] = pa.array([x[0] for x in h], pa.string())
        cols['loss_sha256'] = pa.array([x[1] for x in h], pa.string())
        cols['repo_files_json'] = pa.array([found[i][1] for i in ids], pa.string())
        cols['provenance_json'] = pa.array([found[i][2] for i in ids], pa.string())
        pq.write_table(pa.table(cols), f'{out}/midtrain/{name}.parquet', compression='zstd')
        print(f'{name}: {len(ids)} parents', flush=True)


def sft_files(source):
    """Same file order as census_sft_v2.py / materialize_prelude_sft_v2.py source_files()."""
    if source.startswith('i3:'):
        return 'PrimeIntellect/INTELLECT-3-SFT', sorted(glob.glob(f'{SFT_ROOT}/PrimeIntellect/INTELLECT-3-SFT/{source[3:]}/*.parquet'))
    if source == 'agenttrove':
        return 'open-thoughts/AgentTrove', sorted(glob.glob(f'{SFT_ROOT}/open-thoughts/AgentTrove/**/*.parquet', recursive=True))
    return 'nebius/SWE-rebench-openhands-trajectories', sorted(glob.glob(f'{SFT_ROOT}/nebius/SWE-rebench-openhands-trajectories/*.parquet'))


def revision(repo):
    return open(f'{SFT_ROOT}/{repo}/PINNED_REVISION').read().strip()


def locate(source):
    """census row k -> (relative path, row in file) for every row of the source."""
    repo, files = sft_files(source)
    bounds, names = [0], []
    for f in files:
        bounds.append(bounds[-1] + pq.ParquetFile(f).metadata.num_rows)
        names.append(os.path.relpath(f, f'{SFT_ROOT}/{repo}'))
    return repo, revision(repo), np.array(bounds), names


def scan_rendered(rp):
    """Rows of one rendered part: (mix, normalized uuid, part stem, 1-based row, ids sha256, loss sha256)."""
    mix, stem = rp.split('/')[-2], os.path.basename(rp)[:-len('.parquet')]
    uuids = pq.read_table(f'{V2}/normalized/{mix}/{stem}.parquet', columns=['uuid']).column(0).to_pylist()
    out = []
    for b in pq.ParquetFile(rp).iter_batches(batch_size=2000, columns=['id', 'ids', 'loss']):
        d = b.to_pydict()
        for rid, ids, loss in zip(d['id'], d['ids'], d['loss']):
            n = int(rid.rsplit('-', 1)[1])  # 1-based row of the normalized part
            out.append((mix, uuids[n - 1], stem, n, hashlib.sha256(np.asarray(ids, '<i4').tobytes()).hexdigest(),
                        hashlib.sha256(np.asarray(loss, np.int8).tobytes()).hexdigest()))
    return out


def trained_rows(procs):
    """mix_source -> {normalized uuid: (ids sha256, loss sha256, trained, part stem, part row)}: rows that survived rendering; trained is
    False for the loop-scan rows dropped at packing."""
    flagged = {json.loads(line)['uuid'] for line in open(f'{V2}/loop_scan/full/flagged_turns.jsonl')}
    out = defaultdict(dict)
    parts = sorted(glob.glob(f'{V2}/rendered/*/*.parquet'))
    with Pool(procs) as pool:
        for rows in pool.imap_unordered(scan_rendered, parts):
            for mix, u, stem, n, hi, hl in rows:
                out[mix][u] = (hi, hl, u not in flagged, stem, n)
    return out


def sft(out, procs):
    sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'data', 'sft_v2')
    import sys
    sys.path.insert(0, sys_path)
    from materialize_prelude_sft_v2 import MIX, NOTHINK, name_of
    kept = trained_rows(procs)
    os.makedirs(f'{out}/sft_v2', exist_ok=True)
    report = {}
    for source, mix in MIX.items():
        repo, rev, bounds, names = locate(source)
        rows = np.load(f'{V2}/selection/{name_of(source)}.npy')
        f = np.searchsorted(bounds, rows, side='right') - 1
        uu = [f'{source}:{k}' for k in rows]
        info = [kept[mix].get(u) for u in uu]
        t = pa.table({'uuid': uu, 'mix_source': [mix] * len(rows), 'hf_repo': [repo] * len(rows),
                      'revision': [rev] * len(rows), 'relative_path': [names[i] for i in f],
                      'row': (rows - bounds[f]).astype(np.int64), 'census_row': rows.astype(np.int64),
                      'rendered': [x is not None for x in info], 'trained': [bool(x and x[2]) for x in info],
                      'ids_sha256': [x and x[0] for x in info], 'loss_sha256': [x and x[1] for x in info],
                      'part': [x and x[3] for x in info], 'part_row': pa.array([x and x[4] for x in info], pa.int64())})
        pq.write_table(t, f'{out}/sft_v2/{mix}.parquet', compression='zstd')
        report[mix] = dict(selected=len(rows), rendered=sum(t.column('rendered').to_pylist()),
                           trained=sum(t.column('trained').to_pylist()))
        print(mix, report[mix], flush=True)
    for sub in NOTHINK:
        mix = f'v1_{sub}_nothink'
        recs = []
        for p in sorted(glob.glob(f'{V2}/normalized/{mix}/*.parquet')):
            recs += pq.read_table(p, columns=['uuid']).column(0).to_pylist()
        cache, up = {}, []
        for u in recs:  # v1:<sub>:<v1 part file>:<row>
            _, s, part, k = u.split(':')
            if part not in cache:
                cache[part] = pq.read_table(f'{V1}/normalized/{s}/{part}',
                                            columns=['uuid', 'hf_repo', 'hf_revision', 'file']).to_pylist()
            up.append(cache[part][int(k)])
        info = [kept[mix].get(u) for u in recs]
        t = pa.table({'uuid': recs, 'mix_source': [mix] * len(recs), 'hf_repo': [r['hf_repo'] for r in up],
                      'revision': [r['hf_revision'] for r in up], 'relative_path': [r['file'] for r in up],
                      'upstream_uuid': [r['uuid'] for r in up],
                      'rendered': [x is not None for x in info], 'trained': [bool(x and x[2]) for x in info],
                      'ids_sha256': [x and x[0] for x in info], 'loss_sha256': [x and x[1] for x in info],
                      'part': [x and x[3] for x in info], 'part_row': pa.array([x and x[4] for x in info], pa.int64())})
        pq.write_table(t, f'{out}/sft_v2/{mix}.parquet', compression='zstd')
        report[mix] = dict(selected=len(recs), rendered=sum(t.column('rendered').to_pylist()),
                           trained=sum(t.column('trained').to_pylist()))
        print(mix, report[mix], flush=True)
    json.dump(report, open(f'{out}/sft_v2/counts.json', 'w'), indent=1)


def manifests(out):
    m = f'{out}/manifests'
    copies = {
        'midtrain/source_selection.json': f'{MID}/step1/source_selection.json',
        'midtrain/curation_manifest.json': f'{MID}/step3/curation_manifest.json',
        'midtrain/benchmark_exclusions.json': f'{MID}/step3/benchmark_exclusions.json',
        'midtrain/mixture_manifest.json': f'{MID}/step4/mixture_v3/mixture_manifest.json',
        'midtrain/release_manifest.json': f'{MID}/step5/release/release_manifest.json',
        'midtrain/stage4_mixture_manifest.json': f'{MID}/stage4_128k/mixture/mixture_manifest.json',
        'sft_v2/selection.json': f'{V2}/selection/selection.json',
        'sft_v2/decontam_summary.json': f'{V2}/decontam/summary.json',
        'sft_v2/normalize_stats.json': f'{V2}/normalized/stats.json',
        'sft_v2/loop_scan_flagged.jsonl': f'{V2}/loop_scan/full/flagged_turns.jsonl',
        'sft_v2/swe_overlap.json': f'{V2}/census/swe_overlap.json',
    }
    # Adapter entries (source, subset/family, language, text field, row limit, pinned file hash) without local paths:
    # rebuild_midtrain.py needs them to call prelude_midtrain.adapt() exactly as the build did.
    files = json.load(open(f'{MID}/step2/adapter_manifest.json'))['files']
    os.makedirs(f'{m}/midtrain', exist_ok=True)
    json.dump([{k: v for k, v in e.items() if k not in ('path', 'state')} for e in files],
              open(f'{m}/midtrain/source_files.json', 'w'), indent=1)
    # The frozen Prelude chat tokenizer (openeurollm/prelude tokenizer + ChatML/think/tool tokens, Apache-2.0 base).
    tok = f'{MID}/step2/tokenizer'
    os.makedirs(f'{out}/tokenizer', exist_ok=True)
    for f in os.listdir(tok):
        if os.path.isfile(f'{tok}/{f}'):
            shutil.copy(f'{tok}/{f}', f'{out}/tokenizer/{f}')
    shutil.copy(f'{MID}/step2/tokenizer_contract.json', f'{m}/midtrain/tokenizer_contract.json')
    for dst, src in copies.items():
        if os.path.exists(src):
            os.makedirs(os.path.dirname(f'{m}/{dst}'), exist_ok=True)
            shutil.copy(src, f'{m}/{dst}')
        else:
            print('missing manifest', src, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=64)
    ap.add_argument('--only', choices=['midtrain', 'sft'])
    a = ap.parse_args()
    if a.only != 'sft':
        midtrain(a.out, a.procs)
    if a.only != 'midtrain':
        sft(a.out, a.procs)
    manifests(a.out)


if __name__ == '__main__':
    main()
