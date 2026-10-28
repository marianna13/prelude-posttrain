#!/usr/bin/env python3
"""Finish uncounted canonical parts with bounded, disjoint row-group ranges.

Runs the frozen counting worker through a bounded Parquet view, using a guarded
equivalent linear tool-mask renderer, then merges metric rows in original order.
Completed canonical parts are preserved. Hashes, masks, lengths and the census
build remain the original worker's. The helper plan, parity evidence, code and
every child artifact are pinned.
Stop previous owners before preparing/running this completion plan.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import re
import sys
import time

sys.path.insert(0,'/p/project1/laionize/marianna/dc_agent/.prelude_census_deps')
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pyarrow as pa
import pyarrow.parquet as pq
import census_curated_prelude_midtrain as original
from prelude_midtrain import Reject,render_conversation as _ORIGINAL_RENDER

_NATIVE_PARQUET = pq.ParquetFile


def fast_render_conversation(tok,messages,tools=None):
    """Same template/tool spans; O(messages + tokens), with per-row text parity.

    The frozen template closes a tool group exactly at a prefix boundary. Two
    instrumentation markers therefore locate the same complete groups as all
    prefix renders. Markers are proven absent from the original rendered text,
    removed before tokenization, and the resulting text must equal the original
    byte for byte. Boundaries and token overlaps retain the original guards.
    """
    expected_groups = sum(m['role']=='tool' and (i==0 or messages[i-1]['role']!='tool')
                          for i,m in enumerate(messages))
    if not expected_groups:
        return _ORIGINAL_RENDER(tok,messages,tools)
    kwargs = {'tokenize':False,'add_generation_prompt':False}
    if tools:
        kwargs['tools'] = tools
    full = tok.apply_chat_template(messages,**kwargs)
    start,end = '\x00PRELUDE_CENSUS_TOOL_START\x00','\x00PRELUDE_CENSUS_TOOL_END\x00'
    while start in full or end in full:
        start += '!'
        end += '!'
    template = tok.chat_template
    first = "{%- if i == 0 or messages[i - 1].role != 'tool' %}{{- '<|im_start|>user' }}{%- endif %}"
    last = "{%- if i == messages | length - 1 or messages[i + 1].role != 'tool' %}{{- '<|im_end|>\\n' }}{%- endif %}"
    if template.count(first)!=1 or template.count(last)!=1:
        raise ValueError('unsupported template for exact tool-group instrumentation')
    instrumented = template.replace(first,first.replace("{{- '<|im_start|>user' }}",
        "{{- prelude_census_start }}{{- '<|im_start|>user' }}"))
    instrumented = instrumented.replace(last,last.replace("{{- '<|im_end|>\\n' }}",
        "{{- '<|im_end|>\\n' }}{{- prelude_census_end }}"))
    tagged = tok.apply_chat_template(messages,chat_template=instrumented,
        prelude_census_start=start,prelude_census_end=end,**kwargs)
    clean,spans,previous,cursor,opened = [],[],0,0,None
    for match in re.finditer(re.escape(start)+'|'+re.escape(end),tagged):
        piece = tagged[previous:match.start()]
        clean.append(piece)
        cursor += len(piece)
        if match.group()==start:
            if opened is not None:
                raise ValueError('nested instrumentation markers')
            opened = cursor
        else:
            if opened is None:
                raise ValueError('unpaired instrumentation marker')
            spans.append((opened,cursor))
            opened = None
        previous = match.end()
    clean.append(tagged[previous:])
    if opened is not None or len(spans)!=expected_groups or ''.join(clean)!=full:
        raise ValueError('instrumented text differs from frozen rendered text')
    enc = tok(full,add_special_tokens=False,return_offsets_mapping=True)
    ids,offsets = enc['input_ids'],enc['offset_mapping']
    starts,ends = {s for s,e in offsets if e>s},{e for s,e in offsets if e>s}
    if any(a not in starts or b not in ends for a,b in spans):
        raise Reject('unaligned_tool_group')
    loss,si,previous_s = [1]*len(ids),0,-1
    for k,(s,e) in enumerate(offsets):
        if s<previous_s:
            # Preserve original semantics for any unsupported offset ordering.
            return _ORIGINAL_RENDER(tok,messages,tools)
        previous_s = s
        while si<len(spans) and spans[si][1]<=s:
            si += 1
        if si<len(spans):
            a,b = spans[si]
            if s<b and e>a:
                if s<a or e>b:
                    raise Reject('token_crosses_mask_boundary')
                loss[k] = 0
    if not ids or ids[0]!=tok.bos_token_id:
        raise Reject('missing_bos')
    loss[0] = 0
    return ids,loss


class BoundedParquet:
    def __init__(self,path,groups):
        self.file = _NATIVE_PARQUET(path)
        self.groups = groups
        from types import SimpleNamespace
        self.metadata = SimpleNamespace(num_rows=sum(self.file.metadata.row_group(g).num_rows for g in groups))

    def iter_batches(self,*args,**kwargs):
        return self.file.iter_batches(*args,row_groups=self.groups,**kwargs)


def bounded_process(spec):
    task,groups = spec
    previous_renderer = original.render_conversation
    original.render_conversation = fast_render_conversation
    pq.ParquetFile = lambda path,*a,**kw: (BoundedParquet(path,groups) if str(path)==task['path']
                                           else _NATIVE_PARQUET(path,*a,**kw))
    try:
        return original.process((task,0))
    finally:
        pq.ParquetFile = _NATIVE_PARQUET
        original.render_conversation = previous_renderer


def group_ranges(pf,groups_per_child):
    if groups_per_child<1:
        raise ValueError('groups per child must be positive')
    # The frozen worker batches documents but emits conversations immediately.
    # Split homogeneous canonical parts only, preserving its exact emission order.
    column = pf.schema_arrow.get_field_index('kind')
    kinds = set()
    for g in range(pf.num_row_groups):
        if not pf.metadata.row_group(g).num_rows:
            continue
        stats = pf.metadata.row_group(g).column(column).statistics
        if stats is None or stats.min!=stats.max:
            return [list(range(pf.num_row_groups))]
        kinds.add(stats.min)
    if len(kinds)>1:
        return [list(range(pf.num_row_groups))]
    return [list(range(i,min(i+groups_per_child,pf.num_row_groups)))
            for i in range(0,pf.num_row_groups,groups_per_child)] or [[]]


def prepare(args):
    out = Path(args.out)
    m = original.validate(out)
    plan = {'schema_version':1,'census_build_id':m['build_id'],
            'census_input_manifest_sha256':original.sha(out/'census_input_manifest.json'),
            'helper_sha256':original.sha(__file__),'groups_per_child':args.groups,'parts':[]}
    for task in m['tasks']:
        root = out/'census'/task['key']
        if (root/'stats.json').exists():
            r = json.loads((root/'stats.json').read_text())
            if r['build_id']!=m['build_id'] or r['limit'] or original.sha(root/'metrics.parquet')!=r['metrics_sha256']:
                raise ValueError('completed part identity/checksum differs')
            continue
        pf = _NATIVE_PARQUET(task['path'])
        children = []
        # Even empty parents have one checksum-verified zero-row child.
        ranges = group_ranges(pf,args.groups)
        for i,groups in enumerate(ranges):
            child = dict(task,key=task['key']+f'__{i:05d}')
            children.append({'task':child,'groups':groups,
                'weight':sum(pf.metadata.row_group(g).total_byte_size for g in groups)})
        if [g for c in children for g in c['groups']]!=list(range(pf.num_row_groups)):
            raise ValueError('row groups not disjoint and exhaustive')
        plan['parts'].append({'task':task,'rows':pf.metadata.num_rows,'children':children})
    plan['build_id'] = original.digest(plan)
    path = out/'parallel_completion_plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('completion plan already exists; resume it instead')
    original.write(path,plan)
    print(json.dumps({'parts':len(plan['parts']),'children':sum(len(p['children']) for p in plan['parts']),
                      'build_id':plan['build_id']}),flush=True)


def load(out):
    out = Path(out)
    m = original.validate(out)
    plan = json.loads((out/'parallel_completion_plan.json').read_text())
    if plan['helper_sha256']!=original.sha(__file__) or plan['census_build_id']!=m['build_id'] or \
       plan['census_input_manifest_sha256']!=original.sha(out/'census_input_manifest.json') or \
       plan['build_id']!=original.digest({k:v for k,v in plan.items() if k!='build_id'}):
        raise ValueError('completion plan identity changed')
    return m,plan


def run(args):
    m,plan = load(args.out)
    children = [c for p in plan['parts'] for c in p['children']]
    bins,weights = [[] for _ in range(args.world)],[0]*args.world
    for c in sorted(children,key=lambda c:(-c['weight'],c['task']['key'])):
        rank = min(range(args.world),key=lambda i:(weights[i],i))
        bins[rank].append(c)
        weights[rank] += c['weight']
    if not 0<=args.rank<args.world:
        raise ValueError('invalid rank')
    print(json.dumps({'rank':args.rank,'children':len(bins[args.rank]),'workers':args.workers}),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
         initializer=original.init_worker,initargs=(str(Path(args.out)/'parallel_work'),m)) as pool:
        futures = [pool.submit(bounded_process,(c['task'],c['groups'])) for c in bins[args.rank]]
        for f in as_completed(futures):
            r = f.result()
            print(json.dumps({'key':r['key'],'counts':r['counts'],'seconds':r['seconds'],
                              'resumed':r.get('resumed',False)}),flush=True)


def merge_part(out,m,plan,part):
    out = Path(out)
    root = out/'census'/part['task']['key']
    root.mkdir(parents=True,exist_ok=True)
    children,counts = [],Counter()
    for child in part['children']:
        child_root = out/'parallel_work/census'/child['task']['key']
        r = json.loads((child_root/'stats.json').read_text())
        if r['build_id']!=m['build_id'] or r['limit'] or r['input_sha256']!=part['task']['sha256'] or \
           original.sha(child_root/'metrics.parquet')!=r['metrics_sha256']:
            raise ValueError('bounded child identity/checksum differs')
        counts.update(r['counts'])
        children.append((child_root/'metrics.parquet',r))
    if counts['rows']!=part['rows']:
        raise ValueError('bounded children do not cover parent row count')
    if (root/'stats.json').exists():
        old = json.loads((root/'stats.json').read_text())
        if old['build_id']!=m['build_id'] or old['counts']!=dict(counts) or \
           old['metrics_sha256']!=original.sha(root/'metrics.parquet'):
            raise ValueError('existing completed parent differs')
        return old
    path = root/'metrics.parquet'
    started = time.monotonic()
    with pq.ParquetWriter(str(path)+'.parallel.tmp',original.METRICS,compression='zstd') as writer:
        pending = pa.Table.from_pylist([],schema=original.METRICS)
        for child_path,_ in children:
            for batch in _NATIVE_PARQUET(child_path).iter_batches(batch_size=4096,use_threads=False):
                pending = pa.concat_tables([pending,pa.Table.from_batches([batch])])
                while len(pending)>=4096:
                    writer.write_table(pending.slice(0,4096).combine_chunks())
                    pending = pending.slice(4096)
        if len(pending):
            writer.write_table(pending.combine_chunks())
    if _NATIVE_PARQUET(str(path)+'.parallel.tmp').metadata.num_rows!=part['rows']:
        raise ValueError('merged metric row count differs')
    Path(str(path)+'.parallel.tmp').replace(path)
    result = {'build_id':m['build_id'],'key':part['task']['key'],'limit':0,
              'input_sha256':part['task']['sha256'],'metrics_sha256':original.sha(path),'counts':dict(counts),
              'seconds':sum(r['seconds'] for _,r in children),'merge_seconds':time.monotonic()-started,
              'completion_method':'bounded worker with equivalent linear tool masks; original-order 4096-row merge',
              'helper_sha256':plan['helper_sha256'],'completion_plan_build_id':plan['build_id'],
              'children':[{k:r[k] for k in ['key','metrics_sha256']} for _,r in children]}
    original.write(root/'stats.json',result)
    return result


def merge(args):
    m,plan = load(args.out)
    for part in plan['parts']:
        r = merge_part(args.out,m,plan,part)
        print(json.dumps({'merged':r['key'],'rows':r['counts'].get('rows',0)}),flush=True)
    original.write(Path(args.out)/'parallel_completion.json',{
        'census_build_id':m['build_id'],'completion_plan_build_id':plan['build_id'],
        'helper_sha256':plan['helper_sha256'],'parts':len(plan['parts']),
        'status':'all bounded parts merged; independent census finalization still required'})


def parity(args):
    """Compare a complete real unit against its original worker output bytes."""
    out = Path(args.out)
    m = original.validate(out)
    task = next(t for t in m['tasks'] if t['key']==args.part)
    baseline_root = out/'census'/task['key']
    baseline = json.loads((baseline_root/'stats.json').read_text())
    if baseline['build_id']!=m['build_id'] or baseline['limit'] or \
       original.sha(baseline_root/'metrics.parquet')!=baseline['metrics_sha256']:
        raise ValueError('real parity baseline is not a verified complete unit')
    pf = _NATIVE_PARQUET(task['path'])
    children = [{'task':dict(task,key=task['key']+f'__parity{i:04d}'),'groups':groups}
                for i,groups in enumerate(group_ranges(pf,args.groups))]
    parity_out = out/'parity'/original.sha(__file__)[:12]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
         initializer=original.init_worker,initargs=(str(parity_out/'parallel_work'),m)) as pool:
        futures = [pool.submit(bounded_process,(c['task'],c['groups'])) for c in children]
        for f in as_completed(futures):
            f.result()
    plan = {'helper_sha256':original.sha(__file__),'build_id':'parity-only'}
    result = merge_part(parity_out,m,plan,dict(task=task,rows=pf.metadata.num_rows,children=children))
    if result['counts']!=baseline['counts'] or result['metrics_sha256']!=baseline['metrics_sha256']:
        raise ValueError('real full-unit parity differs from original worker')
    report = {'status':'passed','task':task['key'],'census_build_id':m['build_id'],
              'rows':pf.metadata.num_rows,'children':len(children),'counts':baseline['counts'],
              'metrics_sha256':baseline['metrics_sha256'],'helper_sha256':plan['helper_sha256'],
              'exact_counts_and_parquet_bytes':True}
    original.write(out/'bounded_parity.json',report)
    print(json.dumps(report),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','merge','parity'])
    p.add_argument('--out',required=True)
    p.add_argument('--groups',type=int,default=16)
    p.add_argument('--workers',type=int,default=64)
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world',type=int,default=1)
    p.add_argument('--part',help='complete canonical unit for real full-unit parity')
    args = p.parse_args()
    {'prepare':prepare,'run':run,'merge':merge,'parity':parity}[args.command](args)


if __name__=='__main__':
    main()
