#!/usr/bin/env python3
"""Rebuild the Prelude SFT v2 rendered rows from the published recipe and the upstream datasets.

The recipe lists every selected row of each mix source with its upstream file (repository, revision, path) and row,
plus the part file and 1-based row it had in our build and the SHA256 of its token IDs and loss mask. This script
fetches the upstream rows, normalizes them with the original code (materialize_prelude_sft_v2.convert(); SFT v1
IF/chat rows first through materialize_prelude_sft_v1.convert()), renders them with
prerender_levanter.render_row() (assistant-only loss, chat template with thinking unset) and checks the token and
loss hashes. Rows dropped by the loop scan (trained == False) are left out, as packing did.

Output: OUT/rendered/<mix_source>/<part>.parquet {id = '<mix_source>-<row>', ids, loss, mix_source}, the same files
and row ids as our build, so scripts/data/pack_prelude_stage_megatron.py --inputs 'OUT/rendered/*/*.parquet'
--context 65536 reproduces the packed SFT v2 arrays.

  python recipe/rebuild_sft_v2.py --recipe RECIPE_DIR --tokenizer TOKENIZER_DIR --cache UPSTREAM_DIR --out OUT \
      [--mix i3_math,agenttrove] [--limit-parts N] [--workers 16]
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'scripts' / 'data' / 'sft_v2'), str(ROOT / 'scripts' / 'data'), str(ROOT / 'scripts' / 'sft_gen')]
import materialize_prelude_sft_v1 as v1  # noqa: E402
from materialize_prelude_sft_v2 import MAX_TOKENS, MIX, convert  # noqa: E402
from prerender_levanter import SCHEMA, render_row  # noqa: E402

SOURCE = {mix: source for source, mix in MIX.items()}


def fetch(repo, revision, rel, cache, sources_dir):
    for root in [sources_dir, cache]:
        if root and (p := Path(root) / repo / rel).exists():
            return p
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, rel, revision=revision, repo_type='dataset', local_dir=Path(cache) / repo))


def upstream_rows(path, rows, v1_sub=None):
    """Selected upstream records: by in-file row for Parquet sources, by upstream uuid for the SFT v1 JSONL files."""
    if v1_sub is None:
        pf, want, out, start = pq.ParquetFile(path), sorted(set(rows)), {}, 0
        for g in range(pf.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            hit = [x - start for x in want if start <= x < start + n]
            if hit:
                out.update({start + i: r for i, r in zip(hit, pf.read_row_group(g).take(pa.array(hit)).to_pylist())})
            start += n
        return out
    want, out = set(rows), {}
    for rec in v1.iter_records(str(path), 'jsonl', 0):
        if rec.get('uuid') in want:
            out[rec['uuid']] = rec
            if len(out) == len(want):
                break
    return out


_TOK = None


def init(tokenizer):
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tokenizer)


def build_part(job):
    """One rendered part: every trained row of it, in our row order, checked against the recipe hashes."""
    mix, part, rows, path = job
    sub = mix[len('v1_'):-len('_nothink')] if mix.startswith('v1_') else None
    recs = upstream_rows(path, [r['upstream_uuid'] if sub else r['row'] for r in rows], sub)
    out = []
    for r in sorted(rows, key=lambda r: r['part_row']):
        if sub:
            conv1, tools1, _, _ = v1.convert(sub, recs[r['upstream_uuid']])
            conv, _, tools = convert('v1', {'conversations': conv1, 'tools_json': tools1})
        else:
            conv, _, tools = convert(SOURCE[mix], recs[r['row']])
        ids, loss = render_row(_TOK, conv, {}, json.loads(tools) if tools else None)
        if len(ids) > MAX_TOKENS or ids[0] != _TOK.bos_token_id or not any(loss):
            raise ValueError(f'{r["uuid"]}: rendering differs from the build (length, BOS or no target)')
        if hashlib.sha256(np.asarray(ids, '<i4').tobytes()).hexdigest() != r['ids_sha256'] or \
           hashlib.sha256(np.asarray(loss, np.int8).tobytes()).hexdigest() != r['loss_sha256']:
            raise ValueError(f'{r["uuid"]}: tokens differ from the build (tokenizer or library version?)')
        out.append({'id': f'{mix}-{r["part_row"]}', 'ids': ids, 'loss': loss, 'mix_source': mix})
    return mix, part, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sources-dir')
    ap.add_argument('--mix', help='comma-separated mix sources (default: all)')
    ap.add_argument('--limit-parts', type=int, default=0, help='first N parts per mix source (a smoke check)')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    files = sorted((Path(a.recipe) / 'sft_v2').glob('*.parquet'))
    if a.mix:
        files = [f for f in files if f.stem in a.mix.split(',')]
    jobs = []
    for f in files:
        t = pq.read_table(f).to_pylist()
        parts = defaultdict(list)
        for r in t:
            if r['trained']:
                parts[r['part']].append(r)
        for part in sorted(parts)[:a.limit_parts or None]:
            rows = parts[part]
            r0 = rows[0]
            if {(r['hf_repo'], r['relative_path']) for r in rows} != {(r0['hf_repo'], r0['relative_path'])}:
                raise SystemExit(f'{f.stem}/{part}: rows from more than one upstream file')
            rel = r0['relative_path']
            if f.stem.startswith('v1_'):  # SFT v1 recorded the file name only
                rel = v1.SUBSETS[f.stem[len('v1_'):-len('_nothink')]][1]
            jobs.append((f.stem, part, rows, fetch(r0['hf_repo'], r0['revision'], rel, a.cache, a.sources_dir)))
    print(f'{len(jobs)} parts, {sum(len(j[2]) for j in jobs)} rows', flush=True)
    done = 0
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.tokenizer,)) as pool:
        for fut in as_completed([pool.submit(build_part, j) for j in jobs]):
            mix, part, rows = fut.result()
            d = Path(a.out) / 'rendered' / mix
            d.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), d / f'{part}.parquet', compression='zstd')
            done += 1
            if done % 20 == 0:
                print(f'  {done}/{len(jobs)} parts', flush=True)
    print(f'done: {len(jobs)} parts verified under {a.out}/rendered')


if __name__ == '__main__':
    main()
