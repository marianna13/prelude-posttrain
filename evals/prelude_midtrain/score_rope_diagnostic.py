"""Score the four matched NIAH controls; never label them a full RULER suite."""
import argparse
import json
from pathlib import Path
from lm_eval.tasks.ruler.common_utils import process_results


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    scores={};reference=None
    for name in ('initialized-rope100k','initialized-rope6m','smoke16-rope6m','smoke16-rope100k'):
        folder=a.results/f'diagnostic-{name}'
        meta=json.loads((folder/'metadata.json').read_text())
        rows=[json.loads(l) for l in (folder/'outputs.jsonl').open()]
        if meta.get('status')!='inference-complete' or len(rows)!=100 or len({r['id'] for r in rows})!=100:
            raise ValueError('incomplete diagnostic '+name)
        identity=sorted((r['id'],r['prompt_sha256']) for r in rows)
        if reference is None:reference=identity
        if identity!=reference or any(r['task']!='niah_single_1' or r['context']!=4096 for r in rows):
            raise ValueError('diagnostic prompts differ')
        values=[process_results(r['doc'],[r['raw']])['4096'] for r in rows]
        scores[name]=dict(examples=100,passkey_pct=sum(values),model=meta['model'],versions=meta['versions'])
    a.out.write_text(json.dumps(scores,indent=2)+'\n');print(json.dumps(scores,indent=2))


if __name__=='__main__':main()
