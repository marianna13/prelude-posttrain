#!/usr/bin/env python3
"""Finalize all retained-token census parts, including empty canonical work units.

This separate finalizer is versioned independently from the frozen token-counting
workers. It checks every metric checksum/footer, per-source split coverage and
token accounting before exposing counts to the mixture builder.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0,'/p/project1/laionize/marianna/dc_agent/.prelude_census_deps')
sys.path.insert(0,str(Path(__file__).resolve().parent))
import duckdb
import pyarrow.parquet as pq
from census_curated_prelude_midtrain import STAGES, POLICY, digest, sha, write, validate


def finalize(args):
    out = Path(args.out)
    m = validate(out)
    def check(task):
        root = out/'census'/task['key']
        r = json.loads((root/'stats.json').read_text())
        if r['build_id']!=m['build_id'] or r['limit'] or r['input_sha256']!=task['sha256']:
            raise ValueError('census task identity differs')
        if sha(root/'metrics.parquet')!=r['metrics_sha256']:
            raise ValueError('census metrics changed')
        n = pq.ParquetFile(root/'metrics.parquet').metadata.num_rows
        if n!=r['counts'].get('rows',0):
            raise ValueError('metrics footer row coverage differs')
        return task['key'],r
    totals,checksums = Counter(),{}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for key,r in pool.map(check,m['tasks']):
            totals.update(r['counts'])
            checksums[key] = r['metrics_sha256']
    if totals['rows']!=m['expected_rows']:
        raise ValueError('full census coverage differs')
    con = duckdb.connect()
    con.execute(f'SET threads={args.threads}')
    con.execute("SET memory_limit='192GB'")
    con.execute('SET parquet_metadata_cache=true')
    con.read_parquet(str(out/'census/*/metrics.parquet'),hive_partitioning=False).create_view('metrics')
    def query(sql):
        cur = con.execute(sql)
        columns = [x[0] for x in cur.description]
        return [dict(zip(columns,row)) for row in cur.fetchall()]
    coverage = query('SELECT mix_source,kind,split,count(*) AS rows FROM metrics GROUP BY ALL ORDER BY ALL')
    expected = sorted(m['expected_counts'],key=lambda r:(r['mix_source'],r['kind'],r['split']))
    if coverage!=expected:
        raise ValueError('per-source/split coverage differs from Step 3')
    invalid = con.execute('''SELECT count(*) FROM metrics WHERE id IS NULL OR tokens<=trainable_tokens
        OR trainable_tokens<=0 OR masked_tokens!=tokens-trainable_tokens
        OR split NOT IN ('train','validation') OR eligible!=(eligibility_reason='')''').fetchone()[0]
    if invalid:
        raise ValueError('invalid token or eligibility accounting')
    for context,minimum in STAGES:
        invalid = con.execute(f'''SELECT count(*) FROM metrics WHERE
          (stage_{context}_chunks>0 AND (NOT eligible OR (category!='replay' AND tokens<{minimum})))
          OR (stage_{context}_chunks>0 AND stage_{context}_trainable_tokens!=trainable_tokens)
          OR stage_{context}_seen_tokens!=CASE WHEN stage_{context}_chunks>0
             THEN tokens+(stage_{context}_chunks-1)*257 ELSE 0 END''').fetchone()[0]
        if invalid:
            raise ValueError('invalid stage accounting')
    sql_totals = con.execute('SELECT count(*),sum(tokens),sum(trainable_tokens) FROM metrics').fetchone()
    if sql_totals!=tuple(totals[k] for k in ['rows','tokens','trainable_tokens']):
        raise ValueError('SQL totals differ from worker totals')
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
    report = {'schema_version':1,'status':'retained-census-complete','census_build_id':m['build_id'],
              'census_input_manifest_sha256':sha(out/'census_input_manifest.json'),
              'curation_manifest_sha256':m['curation_manifest_sha256'],'counts':dict(totals),
              'policy':POLICY,'groups':groups,'length_bins':bins,'source_row_coverage':coverage,
              'metrics_sha256':checksums,'finalizer_sha256':sha(__file__),
              'checks':{'row_counts':True,'source_split_coverage':True,'checksums':True,
                        'token_accounting':True,'stage_accounting':True,'sql_worker_totals_agree':True},
              'training_has_run':False,'token_id_release_built':False}
    if (out/'parallel_completion.json').exists():
        completion = json.loads((out/'parallel_completion.json').read_text())
        plan = json.loads((out/'parallel_completion_plan.json').read_text())
        parity = json.loads((out/'bounded_parity.json').read_text())
        if completion['census_build_id']!=m['build_id'] or plan['census_build_id']!=m['build_id'] or \
           parity['census_build_id']!=m['build_id'] or parity['status']!='passed' or \
           not parity['exact_counts_and_parquet_bytes'] or \
           len({completion['helper_sha256'],plan['helper_sha256'],parity['helper_sha256']})!=1:
            raise ValueError('bounded completion lacks matching exact parity evidence')
        report['completion_evidence'] = {p:sha(out/p) for p in
            ['parallel_completion_plan.json','parallel_completion.json','bounded_parity.json']}
        report['completion_evidence']['helper_sha256'] = completion['helper_sha256']
    report['build_id'] = digest(report)
    write(out/'retained_census.json',report)
    print(json.dumps({k:report[k] for k in ['status','build_id','counts']}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--threads',type=int,default=64)
    finalize(p.parse_args())


if __name__=='__main__':
    main()
