#!/usr/bin/env python3
"""Verify every curated part and global split/ID counts before declaring step 3 complete."""
import argparse
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parent))
from curate_prelude_midtrain import digest,dump,sha,validate,write
from reduce_prelude_midtrain import check_complete,q


def verify_source_rows(out,m,reports):
    """Reconcile every full unit with frozen footers/row cuts and the step 1 census."""
    contract=json.loads((Path(m['step2'])/'tokenizer_contract.json').read_text())
    step1=Path(contract['step1']);files={(e['namespace'],e['file_id']):e for e in m['files']}
    selected={}
    for r in pq.read_table(m['repo_selection_path'],columns=['file_id','row']).to_pylist():
        selected.setdefault(r['file_id'],[]).append(r['row'])
    for rows in selected.values():rows.sort()
    totals=Counter()
    for t in m['tasks']:
        e=files[(t['namespace'],t['file_id'])]
        if e['source']=='stack-v3-train':
            rows=selected.get(e['file_id'],[])
            expected=bisect_left(rows,t['row_start']+t['rows'])-bisect_left(rows,t['row_start'])
        elif t['kind']=='jsonl':
            previous=json.loads((step1/'census_v2/tasks'/f'{t["task_id"]:06d}.json').read_text())
            expected=previous['counts']['rows_in']
        elif t['kind']=='parquet':
            expected=t['rows']
            if e.get('row_limit'):expected=min(expected,max(0,e['row_limit']-t['row_start']))
        else:
            previous=json.loads((step1/'replay_v1'/f'{e["file_id"]:04d}.json').read_text())
            expected=previous['counts']['rows']
        actual=reports[t['task_key']]['counts'].get('scanned_rows',0)
        if actual!=expected:raise RuntimeError(f'input row coverage differs in {t["task_key"]}: {actual} != {expected}')
        totals[e['source']]+=actual
    return dict(totals)


def finalize(out,pilot=False,memory='128GB',threads=16):
    out=Path(out);m=validate(out)
    if digest(dump({k:v for k,v in m.items() if k!='build_id'}))!=m['build_id']:
        raise RuntimeError('input manifest identity changed')
    _,reports,scan_counts,pins=check_complete(out,m,pilot,False)
    source_coverage=verify_source_rows(out,m,reports) if not pilot else None
    reduction=out/('pilot_reduction' if pilot else 'reduction')
    curated=out/('pilot_curated' if pilot else 'curated')
    decision=json.loads((reduction/'decision_manifest.json').read_text())
    if decision['scan_build_id']!=m['build_id']:raise RuntimeError('different scan/decision builds')
    if decision['scope']!=('pilot' if pilot else 'full'):raise RuntimeError('different decision scope')
    identity={k:v for k,v in decision.items() if k!='build_id'}
    if digest(dump(identity))!=decision['build_id']:raise RuntimeError('decision manifest identity changed')
    if decision['scan_summary_sha256']!=sha(reduction/'scan_summary.json'):raise RuntimeError('scan summary changed')
    summary=json.loads((reduction/'scan_summary.json').read_text())
    if summary['input_manifest_sha256']!=sha(out/'input_manifest.json'):
        raise RuntimeError('input manifest changed since scan review')
    if pins!=summary['task_report_sha256']:raise RuntimeError('scan changed since decisions')
    if decision['reducer_sha256']!=sha(Path(__file__).with_name('reduce_prelude_midtrain.py')):
        raise RuntimeError('frozen reducer changed')
    decision_artifacts={**decision['decision_files_sha256'],**decision['artifacts_sha256']}
    def verify_decision(item):
        name,expected_hash=item
        if sha(reduction/name)!=expected_hash:raise RuntimeError('frozen decision artifact changed: '+name)
    with ThreadPoolExecutor(max_workers=min(16,threads)) as pool:
        list(pool.map(verify_decision,decision_artifacts.items()))
    expected={t['task_key'] for t in m['tasks']}
    if pilot:expected={k for k in reports if not k.startswith('reference-')}
    completed={p.parent.name for p in curated.glob('*/stats.json')}
    if completed!=expected:raise RuntimeError(f'materialization incomplete: {len(expected-completed)} missing, {len(completed-expected)} unexpected')
    def verify_part(key):
        r=json.loads((curated/key/'stats.json').read_text())
        if r['decision_build_id']!=decision['build_id']:raise RuntimeError('mixed decision builds in curated rows')
        path=curated/key/'rows.parquet'
        if sha(path)!=r['artifact_sha256']['rows.parquet']:raise RuntimeError('curated payload changed')
        if pq.ParquetFile(path).metadata.num_rows!=r['rows']:raise RuntimeError('curated row count mismatch')
        return str(path.relative_to(out)),r['artifact_sha256']['rows.parquet'],r['rows']
    parts={};rows=0
    with ThreadPoolExecutor(max_workers=min(16,threads)) as pool:
        for path,h,n in pool.map(verify_part,sorted(completed)):
            parts[path]=h;rows+=n
    retained=sum(r['rows'] for r in decision['counts_by_source'] if r['reason']=='keep')
    if rows!=retained:raise RuntimeError('curated count differs from retained decisions')
    sys.path.insert(0,m['dependencies']);import duckdb
    con=duckdb.connect();con.execute('SET memory_limit='+q(memory));con.execute('SET threads='+str(int(threads)))
    con.execute('SET parquet_metadata_cache=true')
    temp=Path('/p/scratch/synthlaion/dc-agent-shared/prelude_midtrain_v1')/('finalize-'+digest(str(out.resolve()))[:16])
    temp.mkdir(parents=True,exist_ok=True)
    con.execute('SET temp_directory='+q(temp));con.execute('SET preserve_insertion_order=false')
    con.execute('CREATE VIEW curated AS SELECT * FROM read_parquet('+q(curated/'*/rows.parquet')+')')
    con.execute('CREATE VIEW decisions AS SELECT * FROM read_parquet('+q(reduction/'decisions/*/*.parquet')+',hive_partitioning=true)')
    if con.execute('SELECT count(*)-count(DISTINCT id) FROM curated').fetchone()[0]:raise RuntimeError('duplicate retained ID')
    if con.execute('SELECT count(*) FROM (SELECT split_group FROM curated GROUP BY split_group HAVING count(DISTINCT split)>1)').fetchone()[0]:raise RuntimeError('train/validation group overlap')
    if con.execute("SELECT count(*) FROM curated c LEFT JOIN decisions d USING(id) WHERE d.id IS NULL OR d.reason!='keep' OR c.split!=d.split OR c.split_group!=d.split_group").fetchone()[0]:raise RuntimeError('curated rows disagree with frozen decisions')
    if con.execute("SELECT count(*) FROM curated WHERE kind IS NULL OR kind NOT IN ('document','conversation') OR (kind='document' AND (text IS NULL OR messages_json IS NOT NULL)) OR (kind='conversation' AND (text IS NOT NULL OR messages_json IS NULL)) OR split_group IS NULL OR split NOT IN ('train','validation') OR split IS NULL").fetchone()[0]:raise RuntimeError('canonical payload schema violated')
    if con.execute('SELECT count(*) FROM curated WHERE curation_build_id IS NULL OR curation_build_id!='+q(decision['build_id'])).fetchone()[0]:raise RuntimeError('mixed payload curation versions')
    counts=con.execute('SELECT mix_source,kind,split,count(*) AS rows FROM curated GROUP BY ALL ORDER BY ALL').fetch_arrow_table().to_pylist()
    con.close()
    result={'schema_version':1,'status':'step3-pilot-complete' if pilot else 'step3-complete',
            'scan_build_id':m['build_id'],'decision_build_id':decision['build_id'],'scope':'bounded pilot' if pilot else 'full selected source corpus',
            'policy':decision['policy'],'scan_counts':scan_counts,'retained_rows':rows,'retained_counts':counts,
            'source_row_coverage':source_coverage,
            'decision_counts':decision['counts_by_source'],'input_manifest_sha256':sha(out/'input_manifest.json'),
            'scan_summary_sha256':sha(reduction/'scan_summary.json'),'decision_manifest_sha256':sha(reduction/'decision_manifest.json'),
            'finalizer_sha256':sha(__file__),'curated_parts_sha256':parts,
            'verification':{'memory_limit':memory,'threads':threads,'checksum_workers':min(16,threads),
                            'parquet_metadata_cache':True},
            'deduplication_completed':not pilot,'benchmark_decontamination_completed':not pilot,
            'full_source_materialization_completed':not pilot,
            'validation_partition_frozen':True,'near_deduplication_completed':False,
            'final_mixture_weights_frozen':False,'training_has_run':False}
    if (out/'runtime.json').exists():
        result['runtime']=json.loads((out/'runtime.json').read_text())
    result['build_id']=digest(dump(result))
    write(out/('pilot_curation_manifest.json' if pilot else 'curation_manifest.json'),result)
    print(json.dumps({'status':result['status'],'build_id':result['build_id'],'parts':len(parts),'retained_rows':rows}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--pilot',action='store_true')
    p.add_argument('--memory',default='128GB');p.add_argument('--threads',type=int,default=16);a=p.parse_args()
    finalize(a.out,a.pilot,a.memory,a.threads)


if __name__=='__main__':main()
