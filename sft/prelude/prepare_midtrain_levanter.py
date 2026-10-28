#!/usr/bin/env python3
"""Import portable Prelude IDs/masks into a CPU Levanter cache; never tokenize.

Run in the pinned Levanter image on JURECA or MN5. The same finite ChatDataset
factory is used by the release smoke. A full training launcher must respect its
finite length and pad its final batch, rather than repeat examples to fill it.
"""
from __future__ import annotations

import argparse
import asyncio
from bisect import bisect_right
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'levanter'))
import numpy as np
import pyarrow.parquet as pq
from haliax import Axis
import jax
from levanter.data.text.datasets import ChatDataset
from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache
from levanter.tokenizers import load_tokenizer
from midtrain_format import PreludeMidtrainFormat


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(16 * 1024**2):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def verified_manifest(release):
    m = json.loads((release / 'release_manifest.json').read_text())
    identity = {k: v for k, v in m.items() if k != 'build_id'}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if m['status'] != 'token-export-verified' or m['build_id'] != digest:
        raise ValueError('unverified or changed release manifest')
    for rel, expected in m['artifacts_sha256'].items():
        if not rel.startswith('train/') and sha(release / rel) != expected:
            raise ValueError('release metadata/tokenizer changed: ' + rel)
    return m


def iter_rows(release, m, context):
    for rel, checksum in sorted(m['artifacts_sha256'].items()):
        if not rel.startswith(f'train/{context}/'):
            continue
        if sha(release / rel) != checksum:
            raise ValueError('token shard checksum changed')
        pf = pq.ParquetFile(release / rel)
        for batch in pf.iter_batches(batch_size=16, columns=['id', 'ids', 'loss', 'context', 'split'], use_threads=False):
            # Keep numeric Arrow buffers as arrays. Expanding billions of token
            # IDs/masks into Python integers dominates a full stage import.
            columns = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}
            scalars = {name: columns[name].to_pylist() for name in ('id', 'context', 'split')}
            arrays = {name: (columns[name].offsets.to_numpy(),
                             columns[name].values.to_numpy()) for name in ('ids', 'loss')}
            for i in range(batch.num_rows):
                row = {name: values[i] for name, values in scalars.items()}
                for name, (offsets, values) in arrays.items():
                    row[name] = values[offsets[i]:offsets[i + 1]]
                yield row


def stage_dataset(cache_path, tokenizer_path, context, *, shuffle_seed=20261017):
    """Finite packed examples, each visited once by a bijective permutation."""
    tok = load_tokenizer(str(tokenizer_path))
    fmt = PreludeMidtrainFormat(max_tokens=context)
    proc = fmt.build_preprocessor(tok)
    cache = TreeCache.load(str(Path(cache_path) / 'train'), proc.output_exemplar,
                           CacheMetadata(preprocessor_metadata=proc.metadata))
    ds = ChatDataset(cache, Axis('position', context), max_segments_per_example=64,
                     slice_strategy='raise', mask_user_turns=True, block_cross_document_attention=True)
    if shuffle_seed is not None:
        ds = ds.shuffle(jax.random.PRNGKey(shuffle_seed))
    if not ds.is_finite():
        raise ValueError('Prelude stage must be finite')
    return ds


def make_cache(rows, destination, tokenizer_path, context, *, batch_tokens=32 * 1024**2):
    tok = load_tokenizer(str(tokenizer_path))
    fmt = PreludeMidtrainFormat(max_tokens=context)
    proc = fmt.build_preprocessor(tok)
    if destination.exists():
        raise ValueError('cache output already exists; use a new path')
    temp = Path(str(destination) + '.building')
    if temp.exists():
        raise ValueError('unfinished cache exists; inspect it and use a new cache path')
    counts = dict(rows=0, seen_tokens=0, trainable_tokens=0, zero_target_rows=0)
    with SerialCacheWriter(str(temp / 'train'), proc.output_exemplar,
                           metadata=CacheMetadata(preprocessor_metadata=proc.metadata)) as writer:
        batch, tokens = [], 0
        for row in rows:
            if row['context'] != context or row['split'] != 'train':
                raise ValueError('wrong stage or split')
            batch.append(row)
            tokens += len(row['ids'])
            counts['rows'] += 1
            counts['seen_tokens'] += len(row['ids'])
            counts['trainable_tokens'] += int(np.sum(row['loss'], dtype=np.int64))
            counts['zero_target_rows'] += int(not np.any(row['loss']))
            # Zarr3 stores 128M-element shards with 256K inner chunks. Tiny
            # synchronous appends repeatedly rewrite the same large shard.
            # Bound memory while matching Marin's bulk row batching instead.
            if len(batch) >= 16384 or tokens >= batch_tokens:
                writer.write_batch(proc(batch))
                batch, tokens = [], 0
        if batch:
            writer.write_batch(proc(batch))
    if not counts['rows']:
        raise ValueError('empty cache')
    # Exercise relocation: the cache is opened only after its root has moved.
    temp.replace(destination)
    write(destination / 'train/.stats.json', {'total_elements': counts['rows'], 'total_tokens': counts['seen_tokens']})
    write(destination / 'import_counts.json', counts)
    return counts


def smoke(release, m, context, out):
    stage = next(s for s in m['stages'] if s['context'] == context)
    wanted = {p for p, c in stage['source_counts'].items() if c.get('parents', 0)}
    chosen, covered = [], set()
    continuation = zero = False
    verified_files = set()
    for ref in m['loader_smoke_rows']:
        if ref['context'] != context:
            continue
        path = release / ref['file']
        if ref['file'] not in verified_files:
            if sha(path) != m['artifacts_sha256'][ref['file']]:
                raise ValueError('smoke shard checksum differs')
            verified_files.add(ref['file'])
        pf = pq.ParquetFile(path)
        offsets, count = [], 0
        for g in range(pf.num_row_groups):
            offsets.append(count)
            count += pf.metadata.row_group(g).num_rows
        group = bisect_right(offsets, ref['row']) - 1
        row = pf.read_row_group(group).slice(ref['row'] - offsets[group], 1).to_pylist()[0]
        if row['id'] != ref['id'] or row['context'] != context or row['pool'] != ref['pool']:
            raise ValueError('smoke reference differs from token row')
        is_zero = not any(row['loss'])
        if row['pool'] not in covered or (row['chunk_index'] and not continuation) or (is_zero and not zero):
            chosen.append(row)
            covered.add(row['pool'])
            continuation |= bool(row['chunk_index'])
            zero |= is_zero
    if covered != wanted or (stage['zero_target_chunks'] and not zero):
        raise ValueError('smoke source/zero-mask coverage incomplete')
    out.mkdir(parents=True, exist_ok=True)
    counts = make_cache(chosen, out / 'cache', release / 'tokenizer', context)
    ds = stage_dataset(out / 'cache', release / 'tokenizer', context, shuffle_seed=None)
    row_indices = {row['id']: i for i, row in enumerate(chosen)}
    expected, pending, tokens = [], [], 0
    for row in chosen:
        if pending and (tokens + len(row['ids']) > context or len(pending) == 64):
            expected.append(pending)
            pending, tokens = [], 0
        pending.append(row)
        tokens += len(row['ids'])
    if pending:
        expected.append(pending)
    async def check():
        if await ds.async_len() != len(expected):
            raise ValueError('finite packed sequence count differs')
        targets = 0
        for i, rows in enumerate(expected):
            ex = await ds.getitem_async(i)
            ids = [t for r in rows for t in r['ids']]
            loss = [t for r in rows for t in r['loss']]
            segments = [row_indices[r['id']] for r in rows for _ in r['ids']]
            padding = context - len(ids)
            ids += [0] * padding
            loss += [0] * padding
            segments += [-1] * padding
            weights = np.roll(np.asarray(loss, dtype=np.float32), -1)
            weights[-1] = 0
            np.testing.assert_array_equal(np.asarray(ex.tokens), ids)
            np.testing.assert_array_equal(np.asarray(ex.loss_weight), weights)
            np.testing.assert_array_equal(np.asarray(ex.attn_mask.segment_ids[0]), segments)
            targets += int(np.asarray(ex.loss_weight).sum())
        if targets != counts['trainable_tokens']:
            raise ValueError('causal loader lost target tokens')
        shuffled = stage_dataset(out / 'cache', release / 'tokenizer', context)
        if await shuffled.async_len() != len(expected) or not shuffled.is_finite():
            raise ValueError('shuffle changed finite length')
        # Check every shuffled pack occurs exactly once, without replacement.
        original, actual = Counter(), Counter()
        for i in range(len(expected)):
            original[hashlib.sha256(np.asarray((await ds.getitem_async(i)).tokens).tobytes()).hexdigest()] += 1
            actual[hashlib.sha256(np.asarray((await shuffled.getitem_async(i)).tokens).tobytes()).hexdigest()] += 1
        if actual != original:
            raise ValueError('finite shuffle repeated or dropped packs')
        return len(expected)
    n = asyncio.run(check())
    result = dict(status='passed', context=context, release_build_id=m['build_id'],
        source_pools=sorted(covered), rows=len(chosen), packed_sequences=n, counts=counts,
        continuation_covered=continuation, zero_target_chunk_covered=zero,
        checks=['exact IDs/masks', 'one causal shift', 'padding', 'segment boundaries',
                'all nonzero source pools', 'finite packing and bijective shuffle', 'cache relocation'],
        versions={n: importlib.metadata.version(n) for n in ['jax', 'pyarrow', 'transformers', 'tokenizers']})
    write(out / 'smoke.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['build', 'smoke'])
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--context', type=int, choices=[16384, 32768, 65536], required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    m = verified_manifest(args.release)
    if args.command == 'smoke':
        result = smoke(args.release, m, args.context, args.out)
    else:
        result = make_cache(iter_rows(args.release, m, args.context), args.out,
                            args.release / 'tokenizer', args.context)
        expected = next(s for s in m['stages'] if s['context'] == args.context)
        if (result['rows'], result['seen_tokens'], result['trainable_tokens']) != \
           (expected['chunks'], expected['seen_tokens'], expected['trainable_tokens']):
            raise ValueError('cache counts differ from release')
        ds = stage_dataset(args.out, args.release / 'tokenizer', args.context)
        result.update(packed_sequences=asyncio.run(ds.async_len()), finite=ds.is_finite(),
                      release_build_id=m['build_id'], shuffle_seed=20261017,
                      exhaustion='stop; final training batch must be padded, never repeat parents')
        write(args.out / 'stage_loader.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
