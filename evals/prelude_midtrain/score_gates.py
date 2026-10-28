"""Grade retained outputs using pinned lm-eval scorers and full HumanEval+ tests."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


def final_answer(raw):
    answer=raw.rsplit('</think>',1)[-1]
    if '<think>' in answer:
        return '',True
    return answer.replace('<|im_end|>','').replace('<eos>','').strip(),False


def main():
    import multiprocessing
    multiprocessing.set_start_method('forkserver', force=True)
    multiprocessing.set_forkserver_preload(['numpy'])
    p=argparse.ArgumentParser()
    p.add_argument('--outputs',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(s) for s in a.outputs.read_text().splitlines()]
    groups=defaultdict(list)
    for row in rows:
        groups[(row['suite'],row['task'],row.get('thinking'))].append(row)
    suites={r['suite'] for r in rows}
    if len(suites)!=1 or len({r['context'] for r in rows})!=1:
        raise ValueError('score exactly one suite and context per result file')
    expected_groups=6 if suites=={'chat'} else 13
    if len(groups)!=expected_groups:
        raise ValueError('incomplete suite')
    results={}
    scored=[]
    for (suite,task,thinking),group in groups.items():
        if len(group)!=100 or len({str(r['id']) for r in group})!=100:
            raise ValueError(f'incomplete group {task}: {len(group)}')
        values=[]
        if suite=='ruler':
            from lm_eval.tasks.ruler.common_utils import process_results, process_results_part
            process=process_results_part if task.startswith('ruler_qa_') else process_results
            for row in group:
                value=process(row['doc'],[row['raw']])[str(row['doc']['max_length'])]
                values.append(value)
                scored.append(dict(task=task,id=row['id'],score=value))
        else:
            answers=[final_answer(row['raw']) for row in group]
            if task=='HumanEval+':
                import evaluate
                metric=evaluate.load('code_eval')
                predictions=[]
                for row,(answer,_) in zip(group,answers):
                    blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',answer,re.S)
                    code=blocks[0] if blocks else answer
                    if not re.search(r'\bdef\s+'+re.escape(row['doc']['entry_point'])+r'\s*\(',code):
                        code=row['doc']['prompt']+code
                    predictions.append([code])
                _,detail=metric.compute(references=[r['doc']['test']+'\ncheck('+r['doc']['entry_point']+')' for r in group],
                                        predictions=predictions,k=[1],timeout=20.0,num_workers=4)
                values=[float(detail[i][0][1]['passed']) for i in range(len(group))]
            elif task=='MATH500':
                from lm_eval.tasks.minerva_math.utils import process_results
                values=[process_results(r['doc'],[answer])['math_verify']
                        for r,(answer,_) in zip(group,answers)]
            else:
                from lm_eval.filters.extraction import RegexFilter
                filt=RegexFilter(regex_pattern=r'(-?[$0-9.,]{2,})|(-?[0-9]+)',group_select=-1)
                extracted=filt.apply([[answer] for answer,_ in answers],[r['doc'] for r in group])
                def normalize(s):
                    for pattern in (',',r'\$',r'(?s).*#### ',r'\.$'):
                        s=re.sub(pattern,'',s)
                    return s.strip().lower()
                values=[float(normalize(x[0])==normalize(r['doc']['answer'])) for x,r in zip(extracted,group)]
            for row,value,(answer,opened) in zip(group,values,answers):
                scored.append(dict(task=task,id=row['id'],thinking=thinking,score=value,
                                   open_reasoning=opened,answer=answer))
        key=f'{task}/thinking={thinking}' if suite=='chat' else task
        results[key]=dict(examples=len(group),score_pct=100*sum(values)/len(values))
        if suite=='chat':
            results[key].update(open_reasoning=sum(x[1] for x in answers),
                                reasoning_tag=sum('<think>' in r['raw'] for r in group),
                                generation_limit=sum(r['finish_reason']=='length' for r in group))
    if rows[0]['suite']=='ruler':
        results['ruler_mean_pct']=sum(x['score_pct'] for x in results.values())/13
        results['complete_13_tasks']=len(groups)==13
        results['target_met']=(len(groups)==13 and results['ruler_mean_pct']>=85
                              and results['niah_single_1']['score_pct']>=95)
    a.outputs.with_name('scores.json').write_text(json.dumps(results,indent=2)+'\n')
    a.outputs.with_name('per-example-scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
