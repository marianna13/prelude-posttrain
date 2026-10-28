"""Register a completed gate measurement alongside existing evaluation records."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from record_eval import to_yaml


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--id',required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--cluster',choices=['jureca','juwels','mn5'],required=True)
    p.add_argument('--jobs',required=True)
    p.add_argument('--model',default='prelude-9b-anneal300b-ml1-dominant2-iter0989075')
    p.add_argument('--notes',default='Frozen-input initialization/smoke measurement; post-stage matched short-context noninferiority remains separate.')
    a=p.parse_args()
    meta=json.loads((a.results/'metadata.json').read_text())
    scores=json.loads((a.results/'scores.json').read_text())
    if meta.get('status')!='inference-complete':raise ValueError('incomplete inference')
    benchmarks={}
    if 'ruler_mean_pct' in scores:
        if not scores['complete_13_tasks']:raise ValueError('incomplete RULER')
        for name,value,target,n in [('RULER mean',scores['ruler_mean_pct'],85,1300),
                                    ('NIAH repeated passkey',scores['niah_single_1']['score_pct'],95,100)]:
            benchmarks[name]=dict(score_pct=value,target_pct=target,examples=n,
                                  status='pass' if value>=target else 'fail')
    else:
        for name,entry in scores.items():benchmarks[name]=dict(**entry,status='baseline')
    record=dict(schema_version=1,kind='prelude_midtrain_gate_run',id=a.id,recorded=datetime.date.today().isoformat(),
                source='ours',model=a.model,label=a.label,
                config=dict(context=meta['context'],checkpoint=meta['model'],versions=meta['versions'],
                            decoding=meta['decoding'],protocol='rl/evals/prelude_midtrain/gate_policy.json'),
                jobs=dict(cluster=a.cluster,slurm=a.jobs.split(','),results_dir=str(a.results)),
                benchmarks=benchmarks,
                artifacts_sha256={name:hashlib.sha256((a.results/name).read_bytes()).hexdigest()
                                  for name in ('metadata.json','outputs.jsonl','scores.json','per-example-scores.jsonl')},
                notes=a.notes)
    dest=Path(__file__).resolve().parents[1]/'runs'/f'{a.id}.yaml'
    if dest.exists():raise ValueError('record already exists')
    dest.write_text(to_yaml(record)+'\n')
    print(dest)


if __name__=='__main__':main()
