#!/usr/bin/env python3
"""Rebuild Prelude mid-training token chunks from the published recipe and the upstream datasets.

For every parent of a stage the recipe gives the upstream file (repository, revision, path, SHA256) and the record
(row index or byte offset). This script fetches the file, reads the record, and runs the original pipeline code on
it: prelude_midtrain.adapt() (normalization and canonical ID), the exporter's document_ids() /
fast_render_conversation() (tokens and loss mask) and token_chunks() (context windows with 256-token continuations).
It checks the canonical ID, the raw-record hash, token counts and the token/loss SHA256 against the recipe, so a
rebuilt parent is byte-identical to the one we trained on or the script stops.

Output: OUT/release/train/<context>/part-NNNNN.parquet {id, parent_id, mix_source, pool, category, kind, split,
split_group, context, chunk_index, ids, loss}, the input of scripts/data/pack_prelude_stage_megatron.py --release OUT.

  python recipe/rebuild_midtrain.py --recipe RECIPE_DIR --stage stage1_16k --tokenizer TOKENIZER_DIR \
      --cache UPSTREAM_DIR --out OUT [--sources-dir LOCAL_MIRROR] [--limit N] [--workers 16]

RECIPE_DIR is a snapshot of the Hugging Face dataset marianna13/prelude-9b-posttrain-recipe. TOKENIZER_DIR is the
Prelude chat tokenizer (sft/prelude/prepare_midtrain_tokenizer.py; hashes in manifests/midtrain/release_manifest.json).
Upstream files are downloaded into --cache from the Hugging Face Hub at the pinned revision, except Nemotron-CC,
which comes from data.commoncrawl.org. --sources-dir is checked first, laid out as <hf_repo>/<relative_path>.
"""
import argparse
import hashlib
import io
import json
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parent.parent / 'scripts' / 'data'
sys.path.insert(0, str(DATA))
from prelude_midtrain import adapt  # noqa: E402
from export_prelude_midtrain import document_ids, token_chunks  # noqa: E402
from finish_prelude_retained_census import fast_render_conversation  # noqa: E402

COMMONCRAWL = 'https://data.commoncrawl.org/'
SCHEMA = pa.schema([(n, pa.string()) for n in ['id', 'parent_id', 'mix_source', 'pool', 'category', 'kind', 'split',
                                               'split_group']] +
                   [('context', pa.int64()), ('chunk_index', pa.int64()),
                    ('ids', pa.list_(pa.int32())), ('loss', pa.list_(pa.int8()))])


def file_sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def fetch(entry, cache, sources_dir, verify):
    """Local path of the pinned upstream file, downloading it when needed."""
    rel = entry['relative_path']
    for root in [sources_dir, cache]:
        if root and (p := Path(root) / entry['hf_repo'] / rel).exists():
            break
    else:
        p = Path(cache) / entry['hf_repo'] / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if entry['hf_repo'] == 'nvidia/Nemotron-CC':
            tmp = p.with_suffix(p.suffix + '.part')
            urllib.request.urlretrieve(COMMONCRAWL + rel, tmp)
            tmp.rename(p)
        else:
            from huggingface_hub import hf_hub_download
            got = hf_hub_download(entry['hf_repo'], rel, revision=entry['revision'], repo_type='dataset',
                                  local_dir=Path(cache) / entry['hf_repo'])
            p = Path(got)
    if verify and file_sha256(p) != entry['sha256']:
        raise SystemExit(f'{p}: SHA256 differs from the pinned {entry["sha256"]}')
    return p


def read_records(path, kind, locators):
    """locator -> raw record. row_index: decoded row of a Parquet or JSONL(.zst/.gz) file; byte_offset: JSONL line start."""
    want, out = set(locators), {}
    name = str(path)
    if kind == 'byte_offset':
        with open(path, 'rb') as f:
            for off in sorted(want):
                f.seek(off)
                out[off] = json.loads(f.readline())
        return out
    if name.endswith('.parquet'):
        pf, start = pq.ParquetFile(path), 0
        for g in range(pf.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            hit = [x - start for x in want if start <= x < start + n]
            if hit:
                rows = pf.read_row_group(g).take(pa.array(sorted(hit))).to_pylist()
                out.update({start + i: r for i, r in zip(sorted(hit), rows)})
            start += n
        return out
    if name.endswith(('.zst', '.zstd')):
        import zstandard
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, 'rb')))
    elif name.endswith('.gz'):
        import gzip
        reader = gzip.open(path, 'rt')
    else:
        reader = open(path)
    last = max(want)
    with reader:
        for i, line in enumerate(reader):
            if i in want:
                out[i] = json.loads(line)
            if i >= last:
                break
    return out


_TOK = None


def init(tokenizer):
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tokenizer)


def build_file(job):
    """Rebuild every selected parent of one upstream file; returns the chunk rows."""
    entry, path, parents = job
    entry = dict(entry, path=str(path))
    records = read_records(path, parents[0]['locator_kind'], [p['locator'] for p in parents])
    rows = []
    for p in parents:
        prov = json.loads(p['provenance_json'])
        row = adapt(records[p['locator']], entry, p['locator'], selected_repo=prov.get('repo_selection'))
        if row['id'] != p['id']:
            raise ValueError(f'{p["id"]}: rebuilt canonical id {row["id"]}')
        if row['provenance']['record_sha256'] != p['record_sha256']:
            raise ValueError(f'{p["id"]}: upstream record changed')
        if row['kind'] == 'document':
            ids, loss = document_ids(_TOK, row['text'])
        else:
            ids, loss = fast_render_conversation(_TOK, row['messages'], row['tools'])
            ids, loss = np.asarray(ids, dtype='<i4'), np.asarray(loss, dtype=np.int8)
        if hashlib.sha256(ids.tobytes()).hexdigest() != p['ids_sha256'] or \
           hashlib.sha256(loss.tobytes()).hexdigest() != p['loss_sha256']:
            raise ValueError(f'{p["id"]}: tokens differ from the release (tokenizer or library version?)')
        for index, _, _, _, _, w_ids, w_loss in token_chunks(ids, loss, p['context']):
            rows.append({k: p[k] for k in ['mix_source', 'pool', 'category', 'kind', 'split', 'split_group']} |
                        dict(id=f'{p["id"]}:c{p["context"]}:{index}', parent_id=p['id'], context=p['context'],
                             chunk_index=index, ids=w_ids, loss=w_loss))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--stage', required=True, help='stage1_16k | stage2_32k | stage3_64k | stage4_128k')
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sources-dir')
    ap.add_argument('--limit', type=int, default=0, help='first N parents only (a smoke check)')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--skip-file-hash', action='store_true', help='do not re-hash whole upstream files')
    a = ap.parse_args()
    recipe = Path(a.recipe)
    table = pq.read_table(recipe / 'midtrain' / f'{a.stage}.parquet')
    if a.limit:
        table = table.slice(0, a.limit)
    entries = {(e['source'], e['relative_path']): e
               for e in json.load(open(recipe / 'manifests' / 'midtrain' / 'source_files.json'))}
    groups = defaultdict(list)
    for p in table.to_pylist():
        groups[(p['source'], p['relative_path'])].append(p)
    print(f'{a.stage}: {table.num_rows} parents from {len(groups)} upstream files', flush=True)
    jobs = []
    for key, parents in groups.items():
        entry = entries[key]
        jobs.append((entry, fetch(entry, a.cache, a.sources_dir, not a.skip_file_hash), parents))
    out = Path(a.out) / 'release' / 'train'
    writers, n_parts, done = {}, defaultdict(int), 0
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.tokenizer,)) as pool:
        for fut in as_completed([pool.submit(build_file, j) for j in jobs]):
            by_ctx = defaultdict(list)
            for r in fut.result():
                by_ctx[r['context']].append(r)
            for ctx, rows in by_ctx.items():
                d = out / str(ctx)
                d.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), d / f'part-{n_parts[ctx]:05d}.parquet',
                               compression='zstd')
                n_parts[ctx] += 1
            done += 1
            if done % 20 == 0:
                print(f'  {done}/{len(jobs)} files', flush=True)
    print(f'done: {table.num_rows} parents verified, {sum(n_parts.values())} part files under {out}')


if __name__ == '__main__':
    main()
