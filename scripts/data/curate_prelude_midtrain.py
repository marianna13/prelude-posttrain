#!/usr/bin/env python3
"""Prelude step 3: bounded scans, global hash reduction, audited decisions, local filtering.

Raw payloads never cross a shuffle. Scans persist all benchmark hit hashes;
frequency alone never excludes a hash. Final materialization rereads source rows
by stable ID and preserves the full canonical record.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parent))
from prelude_midtrain import CASCADE, TERMINAL, STACK, ULTRA, Reject, adapt, dump, response_parts
from materialize_prelude_midtrain import NORMALIZED, pack_normalized, records, sha
from prelude_decontam import Scanner

ROOT=Path(__file__).resolve().parents[2]
CODE=['scripts/data/curate_prelude_midtrain.py','scripts/data/prelude_decontam.py','scripts/data/prelude_ngram_scan.cpp']
FIELDS=[('id',pa.string()),('namespace',pa.string()),('file_id',pa.int32()),('locator',pa.int64()),
        ('source',pa.string()),('subset',pa.string()),('kind',pa.string()),('mix_source',pa.string()),
        ('record_sha256',pa.string()),('task_hash',pa.string()),('task_alias',pa.string()),
        ('content_sha256',pa.string()),('repo_key',pa.string()),('content_id',pa.string()),
        ('quality',pa.float64()),('algo_rel',pa.float64()),('raw_hits',pa.list_(pa.uint64())),
        ('eligible_hits',pa.list_(pa.uint64()))]
META=pa.schema(FIELDS)
FILES=pa.schema([('parent_id',pa.string()),('repo_key',pa.string()),('content_id',pa.string()),
                 ('content_sha256',pa.string()),('file_path',pa.string())])
SAMPLES=pa.schema([('h',pa.uint64()),('ngram',pa.string()),('eligible',pa.bool_()),
                   ('id',pa.string()),('source',pa.string()),('subset',pa.string()),('field',pa.string())])
REF=pa.schema([('task_hash',pa.string()),('uuid',pa.string()),('file',pa.string()),('row',pa.int64())])
CURATED=pa.schema(list(NORMALIZED)+[('split',pa.string()),('split_group',pa.string()),('curation_build_id',pa.string())])
_M=_OUT=_SCANNER=_SELECTED=None


def digest(value): return hashlib.sha256(value.encode('utf-8')).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)


def state(path):
    s=Path(path).stat();return {'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def check(entry):
    if state(entry['path'])!=entry['state']:raise RuntimeError('frozen input changed: '+entry['path'])


def task_prompt(messages):
    """Exact first genuine user task; system, assistant and tool replies are excluded.

    No case folding, stripping, whitespace collapse, punctuation or Unicode
    normalization. Code in prompts consequently retains its byte identity.
    """
    for m in messages:
        if m['role']!='user':continue
        content=m.get('content')
        if not isinstance(content,str):raise Reject('nontext_task_prompt')
        if response_parts(content) is not None:continue
        if not content:continue
        return digest(content)
    raise Reject('missing_task_prompt')


def task_alias(record,entry):
    # Terminal task names repeat across upstream datasets, hence include source.
    if entry['source']==TERMINAL and isinstance(record.get('task'),str):
        return digest(dump([entry['source'],record.get('source'),record['task']]))
    for field in ('problem_id','question_id','task_id'):
        value=record.get(field)
        if isinstance(value,(str,int)) and str(value):
            return digest(dump([entry['hf_repo'],record.get('source'),field,value]))
    return None


def repo_key(value):return str(value or '').removeprefix('https://github.com/').strip('/').lower()


def scan_fields(row):
    if row['kind']=='document':yield 'document',row['text'];return
    for i,m in enumerate(row['messages']):
        for key in ('content','reasoning_content'):
            if m.get(key):yield f'{i}:{m["role"]}:{key}',m[key]
        if m.get('tool_calls'):yield f'{i}:assistant:tool_calls',dump(m['tool_calls'])
    if row.get('tools'):yield 'tools',dump(row['tools'])


def prepare(args):
    step2,out=Path(args.step2),Path(args.out)
    from materialize_prelude_midtrain import validate
    frozen=validate(step2)
    c=json.loads((step2/'tokenizer_contract.json').read_text())
    index=Path(args.index)
    im=json.loads((index/'manifest.json').read_text())
    if im['n']!=13:raise ValueError('expected frozen Prelude 13-gram index')
    source_index=str(index.resolve())
    source_index_sha256={f:sha(index/f) for f in ('ngrams.npy','manifest.json')}
    # The existing SFT index is a set serialized in arbitrary order. Preserve
    # its membership exactly; the native scanner needs sorted uint64 values.
    import numpy as np
    original=np.load(index/'ngrams.npy')
    ordered=np.unique(original)
    if len(ordered)!=im['total_ngrams'] or len(ordered)!=len(original):
        raise ValueError('index cardinality differs from manifest or contains duplicates')
    index=out/'index';index.mkdir(parents=True,exist_ok=True)
    index_meta={**im,'sorted_view_of':source_index,'source_sha256':source_index_sha256}
    if (index/'ngrams.npy').exists():
        if not np.array_equal(np.load(index/'ngrams.npy'),ordered):raise ValueError('different sorted index exists')
        if json.loads((index/'manifest.json').read_text())!=index_meta:raise ValueError('different sorted index provenance')
    else:
        np.save(index/'ngrams.npy',ordered);write(index/'manifest.json',index_meta)
    files=frozen['files']
    allowed={CASCADE,TERMINAL,STACK,ULTRA,'Nemotron-Pretraining-Specialized-v1'}
    if any(e['namespace']=='core' and e['source'] not in allowed for e in files):raise ValueError('unreserved core source')
    if any('terminal_agent' in e['relative_path'] for e in files):raise ValueError('duplicate Cascade terminal_agent')
    tasks=[dict(t) for t in frozen['tasks'] if t['namespace']=='core']
    for e in files:
        if e['namespace']!='replay':continue
        if e['path'].endswith('.parquet'):
            pf=pq.ParquetFile(e['path']);start=0
            for g in range(pf.num_row_groups):
                n=pf.metadata.row_group(g).num_rows
                if not e.get('row_limit') or start<e['row_limit']:
                    tasks.append({'task_key':f'replay-{e["file_id"]}-rg{g}','namespace':'replay','file_id':e['file_id'],
                                  'kind':'parquet','row_group':g,'row_start':start,'rows':n,
                                  'weight':pf.metadata.row_group(g).total_byte_size})
                start+=n
        else:tasks.append({'task_key':f'replay-{e["file_id"]}','namespace':'replay','file_id':e['file_id'],
                           'kind':'replay','weight':e['state']['bytes']})
    sft=Path(args.sft)
    refs=[]
    for cleaned in sorted((sft/'rendered_clean').glob('*/*.parquet')):
        relative=cleaned.relative_to(sft/'rendered_clean')
        normalized=sft/'normalized'/relative
        if not normalized.exists():raise ValueError('missing matching normalized SFT part')
        refs.append({'task_key':f'reference-{len(refs)}','kind':'reference','path':str(normalized),
                     'state':state(normalized),'clean_path':str(cleaned),'clean_state':state(cleaned),
                     'weight':normalized.stat().st_size})
    if not refs:raise ValueError('no SFT v1 reference parts')
    code={name:sha(ROOT/name) for name in CODE}
    code.update(frozen['code_sha256'])
    m={'schema_version':1,'step2':str(step2.resolve()),'adapter_manifest_sha256':sha(step2/'adapter_manifest.json'),
       'step1_sha256':frozen['step1_sha256'],'files':files,'tasks':tasks,'references':refs,
       'repo_selection_path':frozen['repo_selection_path'],'repo_selection_sha256':frozen['repo_selection_sha256'],
       'index':str(index.resolve()),'index_sha256':{f:sha(index/f) for f in ('ngrams.npy','manifest.json')},
       'source_index':source_index,'source_index_sha256':source_index_sha256,
       'native_library':str((out/'native/ngram_scan.so').resolve()),'native_library_sha256':sha(out/'native/ngram_scan.so'),
       'code_sha256':code,'dependencies':args.deps,
       'policy':{'task_identity':'first genuine nonempty user content, exact UTF-8 SHA256; tools excluded',
                 'task_aliases':'explicit source-scoped task/problem/question IDs; union with prompt identity',
                 'code_and_document_identity':'exact UTF-8 SHA256, no normalization',
                 'source_priority':[STACK,ULTRA,'Nemotron-Pretraining-Specialized-v1','replay'],
                 'ultra_representative':'quality DESC, algo_rel DESC, stable ID ASC',
                 'conversation_representative':'stable ID ASC after SFT exclusion and decontamination',
                 'stack_coverage':'Ultra exact file hash against files in retained selected repositories; audit content-ID conflicts',
                 'validation_fraction':0.01,'validation_seed':'prelude-midtrain-v1-val-20261017',
                 'validation_identity':'task component, canonical GitHub repository, or exact document hash',
                 'benchmark_fields':'all document text, every message content/reasoning/tool_calls, root tools; fields separately',
                 'ngram_n':13,'min_distinct_words':7,'frequency_exclusion':False,
                 'benchmark_exclusions':'explicit reviewed and versioned hashes only; raw hits retained without a cap',
                 'reserved_for_downstream_sft':['AgentTrove','RTS teacher traces','TMax teacher traces'],
                 'supersedes':'older AgentTrove mid-training proposal; Cascade terminal_agent excluded'},
       'status':'inputs frozen; SFT byte checksums finalized by reference scans'}
    m['build_id']=digest(dump(m))
    path=out/'input_manifest.json'
    if path.exists() and json.loads(path.read_text())!=m:raise ValueError('different build; choose a new output directory')
    write(path,m);print(json.dumps({'build_id':m['build_id'],'scan_tasks':len(tasks),'reference_tasks':len(refs)}),flush=True)


def validate(out):
    out=Path(out);m=json.loads((out/'input_manifest.json').read_text())
    for name,expected in m['code_sha256'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('frozen implementation changed: '+name)
    if sha(m['native_library'])!=m['native_library_sha256']:raise RuntimeError('native scanner changed')
    for name,expected in m['index_sha256'].items():
        if sha(Path(m['index'])/name)!=expected:raise RuntimeError('benchmark index changed')
    if sha(m['repo_selection_path'])!=m['repo_selection_sha256']:raise RuntimeError('repo selection changed')
    if sha(Path(m['step2'])/'adapter_manifest.json')!=m['adapter_manifest_sha256']:raise RuntimeError('step 2 manifest changed')
    return m


def init_worker(m,out):
    global _M,_OUT,_SCANNER,_SELECTED
    _M,_OUT=m,Path(out);_SELECTED=None
    sys.path.insert(0,m['dependencies'])
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    _SCANNER=Scanner(m['index'],m['native_library'])


def selected_repos():
    global _SELECTED
    if _SELECTED is None:_SELECTED={r['id']:r for r in pq.read_table(_M['repo_selection_path']).to_pylist()}
    return _SELECTED


def source_rows(entry,task):
    if entry['source']!=STACK:
        yield from records(entry,task);return
    # Avoid converting unselected, potentially multi-GB repository payloads to
    # Python. Only materialize selected Arrow scalars, one repository at a time.
    wanted={r['row'] for r in selected_repos().values() if r['file_id']==entry['file_id']}
    if 'row_group' in task:
        wanted={r for r in wanted if task['row_start']<=r<task['row_start']+task['rows']}
    if not wanted:return
    pf=pq.ParquetFile(entry['path']);start=0
    for g in range(pf.num_row_groups):
        n=pf.metadata.row_group(g).num_rows
        if any(start<=r<start+n for r in wanted):
            cursor=start
            for batch in pf.iter_batches(batch_size=1,row_groups=[g],use_threads=False):
                if cursor in wanted:yield cursor,batch.to_pylist()[0],None
                cursor+=1
        start+=n


def existing(root,m,limit):
    path=root/'stats.json'
    if not path.exists():return None
    report=json.loads(path.read_text())
    if report['build_id']!=m['build_id'] or report['limit']!=limit:raise RuntimeError('resume build/config mismatch')
    for name,expected in report['artifact_sha256'].items():
        if sha(root/name)!=expected:raise RuntimeError('completed artifact changed: '+str(root/name))
    return report


def finish(root,report,names):
    for name in names:(root/(name+'.tmp')).replace(root/name)
    report['artifact_sha256']={name:sha(root/name) for name in names}
    write(root/'stats.json',report)
    return report


def scan_task(job):
    task,limit,pilot=job;start=time.monotonic()
    root=_OUT/('pilot' if pilot else 'scan')/task['task_key']
    root.mkdir(parents=True,exist_ok=True)
    if old:=existing(root,_M,limit):return old
    if task['kind']=='reference':return scan_reference(task,root,limit,start)
    entry=next(e for e in _M['files'] if e['namespace']==task['namespace'] and e['file_id']==task['file_id'])
    check(entry)
    counter=Counter();meta=[];files=[];samples={}
    with pq.ParquetWriter(root/'records.parquet.tmp',META,compression='zstd') as mw, \
         pq.ParquetWriter(root/'files.parquet.tmp',FILES,compression='zstd') as fw, \
         open(root/'rejections.jsonl.tmp','w') as reject:
        def flush():
            if meta:mw.write_table(pa.Table.from_pylist(meta,schema=META));meta.clear()
            if files:fw.write_table(pa.Table.from_pylist(files,schema=FILES));files.clear()
        for locator,record,error in source_rows(entry,task):
            if entry.get('row_limit') and locator>=entry['row_limit']:break
            if limit and counter['scanned_rows']>=limit:break
            counter['scanned_rows']+=1
            try:
                if error:raise Reject(error)
                selected=selected_repos().get(f'{entry["file_id"]}:{locator}') if entry['source']==STACK else None
                row=adapt(record,entry,locator,counter,selected_repo=selected)
                raw,eligible=set(),set()
                for field,value in scan_fields(row):
                    hits,evaluated=_SCANNER.scan(value)
                    counter['ngrams_evaluated']+=evaluated
                    raw.update(hits);eligible.update(h for h,v in hits.items() if v['eligible'])
                    for h,v in hits.items():
                        samples.setdefault(h,{'h':h,'ngram':v['ngram'],'eligible':v['eligible'],
                                              'id':row['id'],'source':entry['source'],'subset':entry['subset'],'field':field})
                content_hash=digest(row['text']) if row['kind']=='document' else None
                prompt=task_prompt(row['messages']) if row['kind']=='conversation' else None
                key=repo_key(record.get('repo_path') if entry['source']==STACK else record.get('repo_name')) if entry['source'] in (STACK,ULTRA) else None
                cid=str(((record.get('source_meta') or {}).get('file_meta') or {}).get('content_id') or '') if entry['source']==ULTRA else None
                meta.append({'id':row['id'],'namespace':entry['namespace'],'file_id':entry['file_id'],'locator':locator,
                             'source':entry['source'],'subset':entry['subset'],'kind':row['kind'],'mix_source':row['mix_source'],
                             'record_sha256':row['provenance']['record_sha256'],'task_hash':prompt,
                             'task_alias':task_alias(record,entry) if prompt else None,'content_sha256':content_hash,
                             'repo_key':key,'content_id':cid,'quality':float(record.get('quality_score') or 0),
                             'algo_rel':float(record.get('algo_rel_score') or 0),'raw_hits':sorted(raw),'eligible_hits':sorted(eligible)})
                if entry['source']==STACK:
                    for f in record['files']:
                        if f['is_vendor'] or f['license_type'] not in ('permissive','no_license'):continue
                        files.append({'parent_id':row['id'],'repo_key':key,'content_id':f['content_id'],
                                      'content_sha256':digest(f['content']),'file_path':f['file_path']})
                counter['accepted_rows']+=1
                counter['raw_hit_rows']+=bool(raw);counter['eligible_hit_rows']+=bool(eligible)
            except (Reject,UnicodeEncodeError) as e:
                reason=str(e) if isinstance(e,Reject) else 'non_utf8_text'
                counter['rejected_'+reason]+=1
                reject.write(dump({'locator':locator,'reason':reason})+'\n')
            if len(meta)>=256 or len(files)>=20000:flush()
        flush()
    pq.write_table(pa.Table.from_pylist(list(samples.values()),schema=SAMPLES),root/'samples.parquet.tmp',compression='zstd')
    check(entry)
    report={'build_id':_M['build_id'],'task_key':task['task_key'],'source':entry['source'],'subset':entry['subset'],
            'limit':limit,'counts':dict(counter),'seconds':time.monotonic()-start,'input_state':entry['state']}
    return finish(root,report,['records.parquet','files.parquet','samples.parquet','rejections.jsonl'])


def scan_reference(task,root,limit,start):
    check(task)
    if state(task['clean_path'])!=task['clean_state']:raise RuntimeError('SFT clean part changed')
    pf=pq.ParquetFile(task['path']);keep=set()
    for b in pq.ParquetFile(task['clean_path']).iter_batches(batch_size=50000,columns=['id']):
        for ident in b.column(0).to_pylist():
            match=re.fullmatch(r'.+-(\d+)',ident)
            if not match:raise RuntimeError('unexpected SFT row ID')
            at=int(match[1])-1
            if not 0<=at<pf.metadata.num_rows or at in keep:raise RuntimeError('invalid/duplicate SFT row locator')
            keep.add(at)
    counter=Counter();rows=[];cursor=0
    with pq.ParquetWriter(root/'reference.parquet.tmp',REF,compression='zstd') as w,open(root/'rejections.jsonl.tmp','w') as reject:
        for b in pf.iter_batches(batch_size=128,columns=['conversations','uuid']):
            for r in b.to_pylist():
                at=cursor;cursor+=1
                if at not in keep:continue
                if limit and counter['scanned_rows']>=limit:break
                counter['scanned_rows']+=1
                try:key=task_prompt(r['conversations'])
                except Reject as e:
                    counter['rejected_'+str(e)]+=1;reject.write(dump({'row':at,'reason':str(e)})+'\n');continue
                rows.append({'task_hash':key,'uuid':r.get('uuid'),'file':task['path'],'row':at});counter['accepted_rows']+=1
            if rows:w.write_table(pa.Table.from_pylist(rows,schema=REF));rows.clear()
            if limit and counter['scanned_rows']>=limit:break
    report={'build_id':_M['build_id'],'task_key':task['task_key'],'source':'reserved_prelude_sft_v1','limit':limit,
            'counts':dict(counter),'seconds':time.monotonic()-start,
            'input_sha256':{task['path']:sha(task['path']),task['clean_path']:sha(task['clean_path'])}}
    check(task)
    if state(task['clean_path'])!=task['clean_state']:raise RuntimeError('SFT clean part changed')
    return finish(root,report,['reference.parquet','rejections.jsonl'])


def schedule(tasks,rank,world):
    # LPT assignment makes ranks disjoint on either cluster and repeatable.
    loads=[0]*world;bins=[[] for _ in range(world)]
    for t in sorted(tasks,key=lambda t:(-t.get('weight',0),t['task_key'])):
        r=min(range(world),key=lambda r:loads[r]);bins[r].append(t);loads[r]+=max(1,t.get('weight',1))
    return bins[rank]


def run(args):
    m=validate(args.out)
    tasks=m['references'] if args.stage=='reference' else m['tasks']
    if args.task:tasks=[t for t in tasks if t['task_key']==args.task]
    if args.pilot:
        # One work unit for every source/subset (all nine Cascade files), each
        # replay file, plus every SFT subset. Production still scans all units.
        entries={(e['namespace'],e['file_id']):e for e in m['files']};grouped={}
        for t in tasks:
            e=entries.get((t.get('namespace'),t.get('file_id')))
            if e:
                group=(e['source'],e['relative_path'] if e['source']==CASCADE else e['subset']) if e['namespace']=='core' else ('replay',e['file_id'])
            else:group=('reference',Path(t['path']).parent.name)
            grouped.setdefault(group,t)
        tasks=list(grouped.values())
    tasks=schedule(tasks,args.rank,args.world)
    pending=[]
    for t in tasks:
        root=Path(args.out)/('pilot' if args.pilot else 'scan')/t['task_key']
        if not (root/'stats.json').exists():pending.append(t)
        else:existing(root,m,args.limit if args.pilot else 0)
    print(json.dumps({'rank':args.rank,'world':args.world,'assigned':len(tasks),'pending':len(pending)}),flush=True)
    with ProcessPoolExecutor(args.workers,initializer=init_worker,initargs=(m,args.out)) as pool:
        jobs={pool.submit(scan_task,(t,args.limit if args.pilot else 0,args.pilot)):t for t in pending}
        for f in as_completed(jobs):
            r=f.result();print(json.dumps({'task':r['task_key'],'seconds':round(r['seconds'],2),'counts':r['counts']}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--step2',required=True);p.add_argument('--out',required=True)
    p.add_argument('--index',default='/p/data1/mmlaion/marianna/decontam/index_prelude_v1')
    p.add_argument('--sft',default='/p/data1/mmlaion/marianna/sft_mixes/prelude_sft_v1')
    p.add_argument('--deps',default='/p/project1/laionize/marianna/dc_agent/.prelude_census_deps')
    p=sub.add_parser('run');p.add_argument('--out',required=True);p.add_argument('--stage',choices=['reference','scan'],default='scan')
    p.add_argument('--workers',type=int,default=16);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--task');p.add_argument('--pilot',action='store_true');p.add_argument('--limit',type=int,default=100)
    a=ap.parse_args()
    if a.command=='prepare':prepare(a)
    elif a.command=='run':run(a)


if __name__=='__main__':main()
