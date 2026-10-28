"""Apply the frozen six-metric noninferiority policy to matched per-example scores.

Input JSON: {"protocol_sha256": "...", "metrics": {"MATH":
[{"id": "...", "score": 0.0}], ...}}. MultiPL-E also requires "language".
The protocol digest must cover prompts/IDs/BOS/budgets/task+backend versions.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def paired_matrix(base, candidate, multilingual=False):
    def index(rows):
        values={}
        for row in rows:
            key=(str(row['id']),row['language'] if multilingual else '')
            if key in values or not 0<=row['score']<=1:
                raise ValueError('duplicate ID or score outside [0,1]')
            values[key]=row['score']
        return values
    left,right=index(base),index(candidate)
    if not left or left.keys()!=right.keys():
        raise ValueError('paired sample/language IDs must match exactly')
    ids=sorted({key[0] for key in left})
    languages=sorted({key[1] for key in left})
    return np.array([[right[(i,lang)]-left[(i,lang)] if (i,lang) in left else np.nan
                      for lang in languages] for i in ids])


def compare(base,candidate,policy):
    if not base.get('protocol_sha256') or base['protocol_sha256']!=candidate.get('protocol_sha256'):
        raise ValueError('protocol fingerprints differ or are missing')
    cfg=policy['short_context_noninferiority']
    rng=np.random.default_rng(cfg['seed'])
    tail=cfg['familywise_alpha']/cfg['comparison_count']
    results={}
    for metric,margin in cfg['maximum_drop_pp'].items():
        matrix=paired_matrix(base['metrics'][metric],candidate['metrics'][metric],metric=='MultiPL-E')
        point=float(np.nanmean(matrix,axis=0).mean()*100)
        draws=[]
        for offset in range(0,cfg['paired_bootstrap_replicates'],128):
            n=min(128,cfg['paired_bootstrap_replicates']-offset)
            indices=rng.integers(0,len(matrix),size=(n,len(matrix)))
            values=np.nanmean(matrix[indices],axis=1).mean(axis=1)*100
            if not np.all(np.isfinite(values)):
                raise ValueError('insufficient multilingual sample coverage for bootstrap')
            draws.extend(values.tolist())
        low,high=np.quantile(draws,[tail,1-tail])
        status='pass' if point>=-margin and low>=-margin else ('fail' if high < -margin else 'inconclusive')
        results[metric]=dict(status=status,delta_pp=point,lower_pp=float(low),upper_pp=float(high),
                             maximum_drop_pp=margin,paired_ids=len(matrix))
    statuses=[x['status'] for x in results.values()]
    return dict(status='fail' if 'fail' in statuses else ('pass' if all(x=='pass' for x in statuses) else 'inconclusive'),
                protocol_sha256=base['protocol_sha256'],metrics=results)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('base',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--policy',type=Path,default=Path(__file__).with_name('gate_policy.json'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.write_text(json.dumps(compare(json.loads(a.base.read_text()),json.loads(a.candidate.read_text()),
                                       json.loads(a.policy.read_text())),indent=2)+'\n')
