#!/usr/bin/env python3
"""Global step 3 reduction and local payload filtering, independent of cluster ranks.

summarize retains raw/low-diversity-filtered hit statistics and examples for review.
decide applies explicitly reviewed exclusions, reserves SFT tasks, deduplicates,
then freezes task/repository validation assignments before any packing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parent))
from curate_prelude_midtrain import (STACK,ULTRA,CURATED,adapt,check,digest,dump,finish,init_worker,
                                    pack_normalized,schedule,selected_repos,sha,source_rows,validate,write)
import curate_prelude_midtrain as scan


def q(value):return "'"+str(value).replace("'","''")+"'"


def union_components(edges):
    """Only explicit upstream task aliases add edges; generic follow-up turns do not."""
    parent={}
    def root(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for prompt,alias in edges:
        a,b=root('p:'+prompt),root('t:'+alias)
        if a!=b:
            small,large=sorted((a,b));parent[large]=small
    return {key[2:]:root(key)[2:] for key in parent if key.startswith('p:')}


def split_for(group,seed='prelude-midtrain-v1-val-20261017'):
    return 'validation' if int(digest(seed+':'+group)[:16],16)%10000<100 else 'train'


def check_complete(out,m,pilot=False,verify_bytes=False):
    root=Path(out)/('pilot' if pilot else 'scan')
    expected={t['task_key'] for t in m['tasks']+m['references']}
    reports={p.parent.name:json.loads(p.read_text()) for p in root.glob('*/stats.json')}
    if not pilot and set(reports)!=expected:
        raise RuntimeError(f'incomplete scan: {len(expected-set(reports))} missing; {len(set(reports)-expected)} unexpected')
    totals=Counter();reference=Counter();pins={}
    for key,r in reports.items():
        if r['build_id']!=m['build_id']:raise RuntimeError('mixed scan builds')
        if not pilot and r['limit']!=0:raise RuntimeError('bounded work unit in production scan')
        for name,h in r['artifact_sha256'].items():
            if verify_bytes and sha(root/key/name)!=h:raise RuntimeError('scan artifact changed')
        c=r['counts'];assert c.get('scanned_rows',0)==c.get('accepted_rows',0)+sum(v for k,v in c.items() if k.startswith('rejected_'))
        (reference if key.startswith('reference-') else totals).update(c)
        pins[key]=sha(root/key/'stats.json')
    return root,reports,{'scan':dict(totals),'reference':dict(reference)},pins


def database(out,m,*,pilot=False,memory='128GB',threads=16):
    sys.path.insert(0,m['dependencies']);import duckdb
    out=Path(out);root=out/('pilot' if pilot else 'scan')
    reduction=out/('pilot_reduction' if pilot else 'reduction');reduction.mkdir(exist_ok=True)
    db=duckdb.connect(str(reduction/'global.duckdb'))
    db.execute('SET memory_limit = '+q(memory));db.execute('SET threads = '+str(int(threads)))
    temp=Path('/p/scratch/synthlaion/dc-agent-shared/prelude_midtrain_v1')/('reduce-'+digest(str(out.resolve()))[:16])
    temp.mkdir(parents=True,exist_ok=True)
    db.execute('SET temp_directory = '+q(temp));db.execute('SET preserve_insertion_order=false')
    db.execute('SET partitioned_write_max_open_files=64')
    db.execute('CREATE OR REPLACE VIEW records AS SELECT *,regexp_extract(filename,\'/([^/]+)/records[.]parquet$\',1) AS task_key FROM read_parquet('+q(root/'*/records.parquet')+',filename=true)')
    db.execute('CREATE OR REPLACE VIEW reference AS SELECT * FROM read_parquet('+q(root/'reference-*/reference.parquet')+')')
    db.execute('CREATE OR REPLACE VIEW repo_files AS SELECT * FROM read_parquet('+q(root/'*/files.parquet')+')')
    db.execute('CREATE OR REPLACE VIEW samples AS SELECT * FROM read_parquet('+q(root/'*/samples.parquet')+')')
    return db,reduction


def summarize(args):
    m=validate(args.out)
    root,reports,counts,pins=check_complete(args.out,m,args.pilot,args.verify_bytes)
    db,dest=database(args.out,m,pilot=args.pilot,memory=args.memory,threads=args.threads)
    if db.execute('SELECT count(*)-count(DISTINCT id) FROM records').fetchone()[0]:raise RuntimeError('record IDs are not unique')
    db.execute('CREATE OR REPLACE TABLE raw_df AS SELECT h,count(*) AS raw_df FROM (SELECT unnest(raw_hits) AS h FROM records) GROUP BY h')
    db.execute('CREATE OR REPLACE TABLE eligible_df AS SELECT h,count(*) AS eligible_df FROM (SELECT unnest(eligible_hits) AS h FROM records) GROUP BY h')
    db.execute('CREATE OR REPLACE TABLE hit_examples AS SELECT h,min(ngram) AS ngram,min(id) AS example_id,min(source) AS example_source,min(field) AS example_field FROM samples GROUP BY h')
    db.execute('CREATE OR REPLACE TABLE hit_review AS SELECT r.*,COALESCE(e.eligible_df,0) AS eligible_df,x.ngram,x.example_id,x.example_source,x.example_field FROM raw_df r LEFT JOIN eligible_df e USING(h) LEFT JOIN hit_examples x USING(h)')
    db.execute('COPY hit_review TO '+q(dest/'hit_review.parquet')+' (FORMAT PARQUET,COMPRESSION ZSTD)')
    # Persist every hash in Parquet; JSON is only a bounded review convenience.
    review=db.execute('SELECT * FROM hit_review ORDER BY eligible_df DESC,raw_df DESC,h LIMIT 1000').fetch_arrow_table().to_pylist()
    write(dest/'boilerplate_review.json',{'status':'review candidates; frequency does not exclude hashes',
          'scan_build_id':m['build_id'],'rows':review,'scope':'pilot' if args.pilot else 'full selected corpus'})
    ref_pins={}
    for r in reports.values():ref_pins.update(r.get('input_sha256',{}))
    verified={'build_id':m['build_id'],'input_manifest_sha256':sha(Path(args.out)/'input_manifest.json'),
              'reducer_sha256':sha(__file__),
              'completed_units':len(reports),'task_report_sha256':pins,'reference_input_sha256':ref_pins,
              'counts':counts,'artifact_sha256':{'hit_review.parquet':sha(dest/'hit_review.parquet'),
                                               'boilerplate_review.json':sha(dest/'boilerplate_review.json')},
              'scope':'pilot' if args.pilot else 'full','reference_bytes_verified':True,
              'scanner_artifacts_rehashed':args.verify_bytes}
    write(dest/'scan_summary.json',verified);db.close()
    print(json.dumps({'completed_units':len(reports),'counts':counts}),flush=True)


def decide(args):
    m=validate(args.out)
    root,reports,counts,pins=check_complete(args.out,m,args.pilot,args.verify_bytes)
    db,dest=database(args.out,m,pilot=args.pilot,memory=args.memory,threads=args.threads)
    summary=json.loads((dest/'scan_summary.json').read_text())
    if summary['task_report_sha256']!=pins:raise RuntimeError('scan changed since review')
    if summary.get('reducer_sha256')!=sha(__file__):raise RuntimeError('reducer changed since summary; regenerate the review summary')
    exclusions=json.loads(Path(args.exclusions).read_text())
    if exclusions['scan_build_id']!=m['build_id'] or exclusions['status']!='reviewed':raise RuntimeError('unreviewed/different exclusion policy')
    if exclusions.get('scope')!=('pilot' if args.pilot else 'full'):raise RuntimeError('exclusions reviewed for a different corpus scope')
    if exclusions.get('frequency_cutoff') is not None:raise RuntimeError('automatic frequency exclusions are forbidden')
    excluded=exclusions['excluded_hashes']
    if any(not r.get('reason') or not r.get('ngram') for r in excluded):raise RuntimeError('exclusion needs text and reason')
    for r in excluded:
        from prelude_decontam import gram_hash
        if gram_hash(r['ngram'])!=int(r['h']):raise RuntimeError('reviewed ngram does not hash to excluded ID')
    ex_table=pa.table({'h':pa.array([int(r['h']) for r in excluded],type=pa.uint64())})
    db.register('excluded_arrow',ex_table);db.execute('CREATE OR REPLACE TABLE excluded AS SELECT * FROM excluded_arrow')
    edges=db.execute('SELECT DISTINCT task_hash,task_alias FROM records WHERE task_alias IS NOT NULL').fetchall()
    components=union_components(edges)
    ct=pa.table({'task_hash':pa.array(list(components),type=pa.string()),
                 'component_hash':pa.array(list(components.values()),type=pa.string())})
    db.register('components_arrow',ct);db.execute('CREATE OR REPLACE TABLE components AS SELECT * FROM components_arrow')
    db.execute('COPY components TO '+q(dest/'task_components.parquet')+' (FORMAT PARQUET,COMPRESSION ZSTD)')
    db.execute('CREATE OR REPLACE TABLE contaminated AS SELECT id,count(*) AS filtered_hits FROM (SELECT id,unnest(eligible_hits) AS h FROM records) h ANTI JOIN excluded USING(h) GROUP BY id')
    db.execute('CREATE OR REPLACE TABLE reference_tasks AS SELECT COALESCE(c.component_hash,r.task_hash) AS task_group,min(r.file||\':\'||r.row) AS reference_id FROM reference r LEFT JOIN components c USING(task_hash) GROUP BY task_group')
    db.execute('CREATE OR REPLACE VIEW grouped AS SELECT m.*,COALESCE(c.component_hash,m.task_hash) AS task_group FROM records m LEFT JOIN components c USING(task_hash)')
    db.execute('CREATE OR REPLACE VIEW annotated AS SELECT g.*,r.reference_id,COALESCE(b.filtered_hits,0) AS filtered_hits FROM grouped g LEFT JOIN reference_tasks r USING(task_group) LEFT JOIN contaminated b USING(id)')
    db.execute('CREATE OR REPLACE VIEW candidates AS SELECT * FROM annotated WHERE reference_id IS NULL AND filtered_hits=0')
    db.execute('CREATE OR REPLACE TABLE retained_stack AS SELECT id FROM candidates WHERE source='+q(STACK)+' QUALIFY row_number() OVER(PARTITION BY content_sha256 ORDER BY id)=1')
    db.execute('CREATE OR REPLACE TABLE stack_file_coverage AS SELECT f.content_sha256,min(parent_id) AS covered_by FROM repo_files f JOIN retained_stack s ON s.id=f.parent_id GROUP BY f.content_sha256')
    db.execute('CREATE OR REPLACE TABLE covered_ultra AS SELECT u.id,f.covered_by FROM candidates u JOIN stack_file_coverage f USING(content_sha256) WHERE u.source='+q(ULTRA))
    db.execute('CREATE OR REPLACE TABLE stack_content_ids AS SELECT DISTINCT content_id,content_sha256 FROM repo_files f JOIN retained_stack s ON s.id=f.parent_id')
    db.execute('COPY (SELECT u.content_id,count(*) AS observations FROM records u JOIN stack_content_ids s USING(content_id) WHERE u.source='+q(ULTRA)+' AND u.content_sha256!=s.content_sha256 GROUP BY u.content_id) TO '+q(dest/'content_id_conflicts.parquet')+' (FORMAT PARQUET,COMPRESSION ZSTD)')
    priority='CASE source WHEN '+q(STACK)+' THEN 0 WHEN '+q(ULTRA)+" THEN 1 WHEN 'Nemotron-Pretraining-Specialized-v1' THEN 2 ELSE 3 END"
    db.execute('CREATE OR REPLACE TABLE ranked AS SELECT c.id,first_value(c.id) OVER w AS winner FROM candidates c ANTI JOIN covered_ultra USING(id) WINDOW w AS (PARTITION BY CASE WHEN kind=\'conversation\' THEN \'task:\'||task_group ELSE \'doc:\'||content_sha256 END ORDER BY '+priority+',CASE WHEN isfinite(quality) THEN quality ELSE 0 END DESC,CASE WHEN isfinite(algo_rel) THEN algo_rel ELSE 0 END DESC,c.id)')
    group="CASE WHEN a.kind='conversation' THEN 'task:'||a.task_group WHEN COALESCE(a.repo_key,'')!='' THEN 'repo:'||a.repo_key ELSE 'doc:'||a.content_sha256 END"
    split="CASE WHEN ('0x'||substr(sha256("+q(m['policy']['validation_seed']+':')+'||split_group),1,16))::UBIGINT % 10000 < 100 THEN \'validation\' ELSE \'train\' END'
    db.execute('CREATE OR REPLACE TABLE decisions AS SELECT *, '+split+' AS split FROM (SELECT a.id,a.namespace,a.file_id,a.locator,a.source,a.subset,a.kind,a.mix_source,a.record_sha256,a.task_key,'+group+' AS split_group,a.reference_id,a.filtered_hits,u.covered_by,r.winner AS duplicate_of,CASE WHEN a.reference_id IS NOT NULL THEN \'reserved_sft_task\' WHEN a.filtered_hits>0 THEN \'benchmark_match\' WHEN u.id IS NOT NULL THEN \'covered_by_stack\' WHEN r.winner!=a.id THEN \'exact_duplicate\' ELSE \'keep\' END AS reason FROM annotated a LEFT JOIN covered_ultra u USING(id) LEFT JOIN ranked r USING(id))')
    destination=dest/'decisions'
    if destination.exists():raise RuntimeError('decisions exist; use a new decision build for a policy change')
    db.execute('COPY decisions TO '+q(destination)+' (FORMAT PARQUET,COMPRESSION ZSTD,PARTITION_BY(task_key))')
    overlap=db.execute("SELECT source,subset,reason,split,count(*) AS rows FROM decisions GROUP BY ALL ORDER BY ALL").fetch_arrow_table().to_pylist()
    # Check global invariants before these decisions may be materialized.
    if db.execute("SELECT count(*) FROM (SELECT split_group FROM decisions WHERE reason='keep' GROUP BY split_group HAVING count(DISTINCT split)>1)").fetchone()[0]:raise RuntimeError('split leakage')
    if db.execute("SELECT count(*) FROM decisions WHERE reason='keep' AND (reference_id IS NOT NULL OR filtered_hits>0 OR covered_by IS NOT NULL)").fetchone()[0]:raise RuntimeError('invalid retained row')
    db.execute("COPY (SELECT id,split,split_group FROM decisions WHERE reason='keep') TO "+q(dest/'split_assignments.parquet')+' (FORMAT PARQUET,COMPRESSION ZSTD)')
    shutil.copyfile(args.exclusions,dest/'benchmark_exclusions.json')
    decision_manifest={'schema_version':1,'status':'decisions-frozen','scan_build_id':m['build_id'],
                       'scan_summary_sha256':sha(dest/'scan_summary.json'),'exclusions_sha256':sha(args.exclusions),
                       'reducer_sha256':sha(__file__),'policy':m['policy'],'counts_by_source':overlap,
                       'task_components':len(components),'scope':'pilot' if args.pilot else 'full',
                       'decision_files_sha256':{str(p.relative_to(dest)):sha(p) for p in sorted(destination.rglob('*.parquet'))},
                       'artifacts_sha256':{f:sha(dest/f) for f in ('task_components.parquet','content_id_conflicts.parquet','split_assignments.parquet','benchmark_exclusions.json')}}
    decision_manifest['build_id']=digest(dump(decision_manifest));write(dest/'decision_manifest.json',decision_manifest)
    db.close();print(json.dumps({'decision_build_id':decision_manifest['build_id'],'counts_by_source':overlap}),flush=True)


def materialize_task(job):
    task,out,decision_manifest,pilot=job;started=time.monotonic()
    dest=Path(out)/('pilot_reduction' if pilot else 'reduction')
    root=Path(out)/('pilot_curated' if pilot else 'curated')/task['task_key'];root.mkdir(parents=True,exist_ok=True)
    completion=root/'stats.json'
    if completion.exists():
        r=json.loads(completion.read_text())
        if r['decision_build_id']!=decision_manifest['build_id']:raise RuntimeError('different decisions in materialization')
        for name,expected in r['artifact_sha256'].items():
            if sha(root/name)!=expected:raise RuntimeError('materialized artifact changed')
        return r
    directory=dest/'decisions'/('task_key='+task['task_key'])
    decisions=[]
    for p in sorted(directory.glob('*.parquet')):
        rel=str(p.relative_to(dest))
        if sha(p)!=decision_manifest['decision_files_sha256'][rel]:raise RuntimeError('decision partition changed')
        decisions.extend(pq.ParquetFile(p).read().to_pylist())
    keep={r['locator']:r for r in decisions if r['reason']=='keep'}
    entry=next(e for e in scan._M['files'] if e['namespace']==task['namespace'] and e['file_id']==task['file_id'])
    check(entry);counter=Counter();rows=[]
    with pq.ParquetWriter(root/'rows.parquet.tmp',CURATED,compression='zstd') as w:
        if keep:
            for locator,record,error in source_rows(entry,task):
                if locator not in keep:continue
                if error:raise RuntimeError('previously accepted record no longer parses')
                d=keep[locator]
                selected=selected_repos().get(f'{entry["file_id"]}:{locator}') if entry['source']==STACK else None
                row=adapt(record,entry,locator,counter,selected_repo=selected)
                if row['id']!=d['id'] or row['provenance']['record_sha256']!=d['record_sha256']:
                    raise RuntimeError('canonical source record changed between passes')
                rows.append({**pack_normalized(row),'split':d['split'],'split_group':d['split_group'],
                             'curation_build_id':decision_manifest['build_id']})
                counter['written_rows']+=1
                if len(rows)>=64 or len(row.get('text') or '')>8*1024**2:
                    w.write_table(pa.Table.from_pylist(rows,schema=CURATED));rows.clear()
            if rows:w.write_table(pa.Table.from_pylist(rows,schema=CURATED))
        if counter['written_rows']!=len(keep):raise RuntimeError('retained rows missing from second source pass')
    check(entry)
    report={'scan_build_id':scan._M['build_id'],'decision_build_id':decision_manifest['build_id'],'task_key':task['task_key'],
            'rows':len(keep),'counts':dict(counter),'seconds':time.monotonic()-started}
    return finish(root,report,['rows.parquet'])


def materialize(args):
    m=validate(args.out);dest=Path(args.out)/('pilot_reduction' if args.pilot else 'reduction')
    decision_manifest=json.loads((dest/'decision_manifest.json').read_text())
    if decision_manifest['scan_build_id']!=m['build_id'] or sha(__file__)!=decision_manifest['reducer_sha256']:
        raise RuntimeError('different scan/reducer implementation')
    if decision_manifest['scope']!=('pilot' if args.pilot else 'full'):raise RuntimeError('different decision scope')
    for name,expected in decision_manifest['artifacts_sha256'].items():
        if sha(dest/name)!=expected:raise RuntimeError('frozen decision artifact changed: '+name)
    tasks=m['tasks']
    if args.pilot:
        names={p.parent.name for p in (Path(args.out)/'pilot').glob('*/records.parquet')}
        tasks=[t for t in tasks if t['task_key'] in names]
    tasks=schedule(tasks,args.rank,args.world)
    with ProcessPoolExecutor(args.workers,initializer=init_worker,initargs=(m,args.out)) as pool:
        jobs=[pool.submit(materialize_task,(t,args.out,decision_manifest,args.pilot)) for t in tasks]
        for f in as_completed(jobs):
            r=f.result();print(json.dumps({'task':r['task_key'],'rows':r['rows'],'seconds':round(r['seconds'],2)}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['summarize','decide','materialize'])
    ap.add_argument('--out',required=True);ap.add_argument('--pilot',action='store_true');ap.add_argument('--verify-bytes',action='store_true')
    ap.add_argument('--memory',default='128GB');ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--exclusions');ap.add_argument('--workers',type=int,default=16);ap.add_argument('--rank',type=int,default=0);ap.add_argument('--world',type=int,default=1)
    a=ap.parse_args()
    if a.command=='summarize':summarize(a)
    elif a.command=='decide':decide(a)
    else:materialize(a)


if __name__=='__main__':main()
