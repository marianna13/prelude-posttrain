#!/usr/bin/env python3
"""Pack an exported Prelude stage (export_prelude_midtrain.py run output), or rendered SFT parts
(prerender_levanter.py, --inputs), into Megatron arrays.

Same layout as megatron_tp/export_stage.py, which megatron_midtrain.py reads:
  tokens.npy      int32 [N, L]   pad id 0 after the last segment
  loss_mask.npy   uint8 [N, L]   position t weights the prediction of token t+1 (export loss is target-aligned)
  cu_seqlens.npy  int32 [N, D+1] segment starts plus the end of the last segment, padded with -1
  manifest.json

Chunks of at least --short tokens are packed best-fit decreasing, shorter ones worst-fit in seeded random order
(at most D = --max-docs segments per row); rows are then shuffled with the same seed.
With --pad-multiple M each segment is padded to a multiple of M inside its own segment (pad id 0, loss 0), as
packed THD context parallelism needs (M = 2 x CP).
--exclude-rows drops rendered rows listed as {part: '<source dir>/<part stem>', row: N} (loop_scan_sft_v2.py output);
the rendered id is '<mix_source>-<N>', N the 1-based row of that part.
"""
import argparse
import bisect
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', help='export output dir (contains release/train/CONTEXT/)')
    p.add_argument('--inputs', help='instead of --release: glob of rendered SFT parts {id, ids, loss} (prerender_levanter.py)')
    p.add_argument('--budget-tokens', type=float, default=0, help='--inputs only: seeded random subset of about this many tokens')
    p.add_argument('--exclude-rows', help='--inputs only: JSONL of {part, row} to drop (loop_scan_sft_v2.py flagged_turns.jsonl)')
    p.add_argument('--context', type=int, default=131072)
    p.add_argument('--out', required=True)
    p.add_argument('--pad-multiple', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261022)
    p.add_argument('--max-docs', type=int, default=64)
    p.add_argument('--short', type=int, default=4096, help='chunks below this are spread worst-fit')
    a = p.parse_args()
    L, M, MAX_DOCS = a.context, a.pad_multiple, a.max_docs
    if bool(a.release) == bool(a.inputs):
        raise SystemExit('pass exactly one of --release / --inputs')
    files = sorted(glob.glob(a.inputs if a.inputs else f'{a.release}/release/train/{L}/*.parquet'))
    if not files:
        raise SystemExit('no chunks')
    excl = set()
    if a.exclude_rows:
        excl = {(x['part'], int(x['row'])) for x in map(json.loads, open(a.exclude_rows))}
    ids, loss, ids_list = [], [], []
    excluded = 0
    keep_frac = 1.0
    if a.budget_tokens:  # rendered rows: sample each row with p = budget / total tokens, keyed on its id
        total = sum(pc.sum(pc.list_value_length(pq.read_table(f, columns=['ids']).column('ids'))).as_py() for f in files)
        keep_frac = min(1.0, a.budget_tokens / total)
    for f in files:
        part = f"{Path(f).parent.name}/{Path(f).stem}"
        cols = ['id', 'ids', 'loss'] + ([] if a.inputs else ['context', 'split'])
        for b in pq.ParquetFile(f).iter_batches(columns=cols, batch_size=256):
            d = b.to_pydict()
            for n, (i, x, y) in enumerate(zip(d['id'], d['ids'], d['loss'])):
                if not a.inputs and (d['context'][n] != L or d['split'][n] != 'train'):
                    raise ValueError(f'bad chunk {i}')
                if len(x) != len(y) or not 0 < len(x) <= L:
                    raise ValueError(f'bad chunk {i}')
                if excl and (part, int(i.rsplit('-', 1)[1])) in excl:
                    excluded += 1
                    continue
                if keep_frac < 1 and int.from_bytes(hashlib.blake2b(f'{a.seed}|{i}'.encode(), digest_size=8).digest(),
                                                     'big') >= keep_frac * 2**64:
                    continue
                ids.append(np.asarray(x, np.int32)); loss.append(np.asarray(y, np.int8)); ids_list.append(i)
    n_chunks = len(ids)
    padded = [-(-len(x) // M) * M for x in ids]
    if max(padded) > L:
        raise ValueError('a padded chunk exceeds the context')

    # Chunks >= SHORT tokens: best-fit decreasing (rows kept sorted by remaining capacity). Shorter chunks, in
    # seeded random order: worst-fit (row with the most room), which spreads them evenly so rows of tiny
    # documents do not reach the segment cap half empty.
    rng = np.random.default_rng(a.seed)
    short = [i for i in range(n_chunks) if padded[i] < a.short]
    order = sorted((i for i in range(n_chunks) if padded[i] >= a.short), key=lambda i: (-padded[i], ids_list[i]))
    rows, keys = [], []  # keys: sorted (remaining, row index) for rows with < MAX_DOCS segments
    for i in order + [short[j] for j in rng.permutation(len(short))]:
        need = padded[i]
        if need >= a.short:
            j = bisect.bisect_left(keys, (need, -1))
        else:
            j = len(keys) - 1 if keys and keys[-1][0] >= need else len(keys)
        if j < len(keys):
            rem, r = keys.pop(j)
            rows[r].append(i)
            rem -= need
        else:
            r, rem = len(rows), L - need
            rows.append([i])
        if len(rows[r]) < MAX_DOCS and rem > 0:
            bisect.insort(keys, (rem, r))
    perm = rng.permutation(len(rows))
    for r in rows:
        r.sort(key=lambda i: ids_list[i])  # deterministic segment order inside a row

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    N = len(rows)
    T = np.lib.format.open_memmap(out / 'tokens.npy', 'w+', np.int32, (N, L))
    LM = np.lib.format.open_memmap(out / 'loss_mask.npy', 'w+', np.uint8, (N, L))
    CU = np.lib.format.open_memmap(out / 'cu_seqlens.npy', 'w+', np.int32, (N, MAX_DOCS + 1))
    seen = trained = pad_inside = 0
    for k, r in enumerate(perm):
        tok, lm, cu, pos = np.zeros(L, np.int32), np.zeros(L, np.uint8), [0], 0
        for i in rows[r]:
            x, y, n = ids[i], loss[i], len(ids[i])
            tok[pos:pos + n] = x
            lm[pos:pos + n - 1] = y[1:]  # weight of predicting token t+1; 0 at the segment end and its padding
            pos += padded[i]
            cu.append(pos)
            seen += n; trained += int(y[1:].sum()); pad_inside += padded[i] - n
        T[k], LM[k] = tok, lm
        CU[k] = cu + [-1] * (MAX_DOCS + 1 - len(cu))
    for x in (T, LM, CU):
        x.flush()
    del T, LM, CU
    # every chunk starts with masked BOS 1, so token 0 of each segment is never a target
    m = dict(description='Prelude stage packed for Megatron (pack_prelude_stage_megatron.py)',
             release=str(Path(a.release).resolve()) if a.release else None, inputs=a.inputs,
             budget_tokens=a.budget_tokens or None, keep_fraction=keep_frac,
             exclude_rows=str(Path(a.exclude_rows).resolve()) if a.exclude_rows else None, excluded_rows=excluded, context=L, pad_multiple=M, max_docs=MAX_DOCS, seed=a.seed,
             conventions=dict(tokens='int32 [N, L]; pad id 0', loss_mask='uint8 [N, L]; position t weights the '
                              'prediction of token t+1', cu_seqlens=f'int32 [N, {MAX_DOCS + 1}]; segment starts then '
                              'end of the last segment, -1 padded'),
             counts=dict(rows=N, chunks=n_chunks, seen_tokens=seen, trained_positions=trained,
                         segment_padding=pad_inside, fill=seen / (N * L)),
             source_files=len(files),
             sha256={f: sha256(out / f) for f in ('tokens.npy', 'loss_mask.npy', 'cu_seqlens.npy')})
    (out / 'manifest.json').write_text(json.dumps(m, indent=1) + '\n')
    print(json.dumps(m['counts']))


if __name__ == '__main__':
    main()
