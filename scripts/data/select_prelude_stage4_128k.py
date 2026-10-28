#!/usr/bin/env python3
"""Select the Prelude 128K mid-training stage (stage 4, issue 0054) from the step-4 census.

Same category weights as stages 1-3 and the stage-3 per-pool weights (`requested_weight` of the 64K stage in
mixture_v3). Parents already selected for any earlier stage are excluded. Preferred original length:
>= 64K for code/agent and long documents, >= 32K for reasoning (no reasoning pool except Cascade math has
enough >= 64K originals); replay (<= 4K) and chat/IF keep their short exceptions. A pool short of its quota
gives the remainder to the other pools of its category, in proportion to their weights.

Parents are ordered within each pool by sha256(seed || id) and taken whole against seen-token quotas
(131072-token chunks, 256-token masked overlap + BOS per continuation), skipping parents above the
remaining quota. Writes OUT/selection-131072.parquet and OUT/mixture_manifest.json in the mixture_v3 layout
so export_prelude_midtrain.py --mixture OUT can tokenize it.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, '/p/project1/laionize/marianna/dc_agent/.prelude_census_deps')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from census_curated_prelude_midtrain import digest, sha, stage_counts, write
from freeze_prelude_midtrain_mixture import CATEGORY_WEIGHTS, SELECTION_SCHEMA

CONTEXT = 131072
SEED = 'prelude-midtrain-v1-stage4-128k-20261022'
MINIMUM = {'code_agent': 65536, 'long_documents': 65536, 'reasoning': 32768, 'replay': 0, 'chat_if': 0}
SLACK = 4_000_000  # extra ordered candidates fetched per pool for the whole-parent skip rule


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True, help='prelude_midtrain_v1 root (contains step4/)')
    p.add_argument('--out', required=True)
    p.add_argument('--budget', type=int, default=1_000_000_000)
    p.add_argument('--threads', type=int, default=32)
    a = p.parse_args()
    base, out = Path(a.base), Path(a.out)
    v3 = base / 'step4/mixture_v3'
    mixture = json.loads((v3 / 'mixture_manifest.json').read_text())
    stage3 = next(s for s in mixture['stages'] if s['context'] == 65536)
    weights = {pool: c['requested_weight'] for pool, c in stage3['source_counts'].items() if c['requested_weight'] > 0}

    con = duckdb.connect()
    con.execute(f'SET threads={a.threads}')
    con.execute(f"CREATE VIEW metrics AS SELECT * FROM read_parquet('{base}/step4/census/*/metrics.parquet')")
    con.execute(f"""CREATE VIEW used AS SELECT id FROM read_parquet([{','.join(
        repr(str(v3 / s['selection'])) for s in mixture['stages'])}])""")
    minimum = ' '.join(f"WHEN '{c}' THEN {m}" for c, m in MINIMUM.items())
    con.execute(f"""CREATE TABLE cand AS SELECT id, mix_source, pool, category, kind, split, split_group, tokens,
        trainable_tokens, sha256('{SEED}' || id) AS h FROM metrics
        WHERE eligible AND split = 'train' AND tokens >= CASE category {minimum} END
          AND NOT EXISTS (SELECT 1 FROM used u WHERE u.id = metrics.id)""")
    # seen tokens at 128K, exactly as census stage_counts
    n = f'1 + greatest(0, (tokens - 258) // {CONTEXT - 257})'
    con.execute('ALTER TABLE cand ADD COLUMN seen BIGINT')
    con.execute(f'UPDATE cand SET seen = tokens + ({n} - 1) * 257')
    avail = dict(con.execute('SELECT pool, sum(seen) FROM cand GROUP BY pool').fetchall())

    # quotas with capacity caps, redistributed within each category
    category = dict(con.execute('SELECT DISTINCT pool, category FROM metrics').fetchall())
    quota = {}
    for cat, cw in CATEGORY_WEIGHTS.items():
        pools = {p: w for p, w in weights.items() if category.get(p) == cat}
        target, fixed = cw * a.budget, {}
        while True:
            free = {p: w for p, w in pools.items() if p not in fixed}
            left = target - sum(fixed.values())
            if not free:
                raise SystemExit(f'category {cat} exhausted: {target - sum(fixed.values()):.0f} tokens short')
            share = {p: left * w / sum(free.values()) for p, w in free.items()}
            over = {p for p, q in share.items() if q > avail.get(p, 0)}
            if not over:
                quota.update(fixed); quota.update(share)
                break
            fixed.update({p: avail.get(p, 0) for p in over})

    rows, counts = [], {}
    for pool, q in sorted(quota.items()):
        q = int(q)
        if q <= 0:
            counts[pool] = dict(requested_weight=weights[pool], quota=0, parents=0, seen_tokens=0, available=avail.get(pool, 0))
            continue
        cand = con.execute(f"""SELECT * EXCLUDE(h) FROM (SELECT *, sum(seen) OVER (ORDER BY h, id) AS cum FROM cand
                               WHERE pool = ?) WHERE cum - seen <= ? ORDER BY h, id""", [pool, q + SLACK]).fetch_arrow_table()
        left, taken, skipped = q, 0, 0
        for r in cand.to_pylist():
            if r['seen'] > left:
                skipped += 1
                continue
            chunks, seen, trainable = stage_counts(r['tokens'], r['trainable_tokens'], CONTEXT)
            assert seen == r['seen']
            rows.append(dict(id=r['id'], mix_source=r['mix_source'], pool=pool, category=r['category'], kind=r['kind'],
                             split=r['split'], split_group=r['split_group'], context=CONTEXT,
                             original_tokens=r['tokens'], seen_tokens=seen, trainable_tokens=trainable, chunks=chunks))
            left -= seen
            taken += 1
        counts[pool] = dict(requested_weight=weights[pool], quota=q, parents=taken, seen_tokens=q - left,
                            remaining_quota=left, skipped_above_remaining_quota=skipped, available=avail.get(pool, 0),
                            chunks=sum(r['chunks'] for r in rows if r['pool'] == pool),
                            original_tokens=sum(r['original_tokens'] for r in rows if r['pool'] == pool),
                            trainable_tokens=sum(r['trainable_tokens'] for r in rows if r['pool'] == pool))

    out.mkdir(parents=True, exist_ok=True)
    sel = out / f'selection-{CONTEXT}.parquet'
    pq.write_table(pa.Table.from_pylist(rows, schema=SELECTION_SCHEMA), sel, compression='zstd')
    seen = sum(r['seen_tokens'] for r in rows)
    cat_seen = {c: sum(r['seen_tokens'] for r in rows if r['category'] == c) / seen for c in CATEGORY_WEIGHTS}
    stage = dict(context=CONTEXT, budget=a.budget, budget_shortfall=a.budget - seen, seen_tokens=seen,
                 trainable_tokens=sum(r['trainable_tokens'] for r in rows), category_seen_shares=cat_seen,
                 document_seen_share=sum(r['seen_tokens'] for r in rows if r['kind'] == 'document') / seen,
                 selection=sel.name, selection_sha256=sha(sel), source_counts=counts)
    m = {k: mixture[k] for k in ('curation_manifest_sha256', 'retained_census_sha256', 'census_build_id')}
    m.update(schema_version=1, status='step4-complete', scope='prelude-midtrain-v1 stage 4 (128K)',
             parent_mixture_build_id=mixture['build_id'], seed=SEED, minimum_original_tokens=MINIMUM,
             category_weights=CATEGORY_WEIGHTS, stages=[stage],
             policy=dict(exclusion='all parents of mixture_v3 stages', weights='mixture_v3 64K requested_weight',
                         redistribution='capacity-capped pools pass their remainder to the rest of their category'))
    m['build_id'] = digest(m)
    write(out / 'mixture_manifest.json', m)
    print(json.dumps(dict(parents=len(rows), seen_tokens=seen, categories=cat_seen,
                          capped={p: c['quota'] for p, c in counts.items() if c['quota'] and c['quota'] >= c['available']})))


if __name__ == '__main__':
    main()
