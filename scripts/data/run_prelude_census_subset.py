#!/usr/bin/env python3
"""Dispatch bounded work units through the unchanged, frozen census adapter."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prelude_midtrain_census as census


def initialize_worker(manifest, output, max_rows, lf_chunks):
    census.init_worker(manifest,output,max_rows)
    if lf_chunks:
        from prelude_lf_counting import LFCountingTokenizer
        census._TOK=LFCountingTokenizer(census._TOK,Path(manifest['tokenizer'])/'tokenizer.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--source',default='stack-v3-train')
    parser.add_argument('--max-task-mib',type=int,default=128)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--rank',type=int,default=int(os.environ.get('SLURM_PROCID','0')))
    parser.add_argument('--world',type=int,default=int(os.environ.get('SLURM_NTASKS','1')))
    parser.add_argument('--max-rows',type=int,default=0)
    parser.add_argument('--limit-tasks',type=int,default=0)
    parser.add_argument('--task-ids',default='')
    parser.add_argument('--lf-chunks',action='store_true')
    args=parser.parse_args()
    out=Path(args.out)
    manifest=json.loads((out/'input_manifest.json').read_text())
    if census.sha256(census.__file__)!=manifest['implementation_sha256']:
        raise ValueError('The census adapter differs from the frozen implementation')
    if args.lf_chunks and args.source!='stack-v3-train':
        raise ValueError('Count-only LF execution is restricted to Stack documents')
    tasks=[task for task in manifest['tasks'] if task['kind']=='parquet'
           and manifest['files'][task['file_id']]['source']==args.source
           and task['weight']<=args.max_task_mib*1024**2]
    if args.task_ids:
        selected_ids={int(value) for value in args.task_ids.split(',')}
        tasks=[task for task in tasks if task['task_id'] in selected_ids]
    if args.limit_tasks:tasks=tasks[:args.limit_tasks]
    groups=[[] for _ in range(args.world)];loads=[0]*args.world
    for task in sorted(tasks,key=lambda task:(-task['weight'],task['task_id'])):
        rank=min(range(args.world),key=lambda rank:loads[rank])
        groups[rank].append(task);loads[rank]+=task['weight']
    print(json.dumps(dict(rank=args.rank,world=args.world,tasks=len(groups[args.rank]),
                          max_task_mib=args.max_task_mib,build_id=manifest['build_id'],
                          lf_chunks=args.lf_chunks,
                          dispatcher_sha256=census.sha256(__file__))),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker,initargs=(manifest,str(out),args.max_rows,args.lf_chunks)) as pool:
        futures={pool.submit(census.process_task,task):task for task in groups[args.rank]}
        for future in as_completed(futures):
            task=futures[future];result=future.result()
            print(json.dumps(dict(rank=args.rank,task_id=task['task_id'],counts=result.get('counts',{}),
                                  seconds=round(result.get('seconds',0),2),resumed=result.get('resumed',False))),flush=True)


if __name__=='__main__':main()
