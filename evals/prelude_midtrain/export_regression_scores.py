"""Export the existing base_fewshot artifacts for the paired promotion comparator."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def export(root, backend_protocol):
    metrics={name:[] for name in ('MATH','GSM8K','MMLU-Pro','HumanEval+','MBPP+','MultiPL-E')}
    inputs=[];configs={};sources={};model_metadata={}
    def retain(path):sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in ('math','mmlupro','code'):
        results=list((root/directory).glob('*/results_*.json'))
        if len(results)!=1:raise ValueError('need exactly one complete result per suite: '+directory)
        result=results[0];retain(result);meta=json.loads(result.read_text())
        api=meta['config']['model_args']
        model_metadata[directory]={k:meta.get(k) for k in ('model_name','tokenizer_eos_token','eot_token_id')}
        model_metadata[directory]['tokenizer']=api.get('tokenizer')
        configs[directory]=dict(lm_eval=meta['lm_eval_version'],transformers=meta['transformers_version'],
                                 task_versions=meta['versions'],task_hashes=meta['task_hashes'],
                                 n_shot=meta['n-shot'],max_length=meta['max_length'],
                                 tokenizer_bos_token=meta['tokenizer_bos_token'],
                                 api={k:api.get(k) for k in ('tokenized_requests','add_bos_token','max_length','max_gen_toks','num_concurrent','seed')},
                                 seeds={k:v for k,v in meta['config'].items() if k.endswith('_seed')},
                                 gen_kwargs=meta['config']['gen_kwargs'])
        for path in sorted(result.parent.glob('samples_*.jsonl')):
            retain(path)
            task=re.sub(r'_\d{4}-\d{2}-\d{2}T.*\.jsonl$', '', path.name.removeprefix('samples_'))
            if task.startswith('minerva_math_'):metric,key,filter_name='MATH','math_verify','none'
            elif task=='gsm8k_cot':metric,key,filter_name='GSM8K','exact_match','flexible-extract'
            elif task.startswith('mmlu_pro_'):metric,key,filter_name='MMLU-Pro','exact_match','custom-extract'
            elif task=='humaneval_plus':metric,key,filter_name='HumanEval+','pass@1','create_test'
            elif task=='mbpp_plus_base':metric,key,filter_name='MBPP+','pass_at_1','none'
            else:raise ValueError('unexpected task '+task)
            for line in path.open():
                row=json.loads(line)
                if row['filter']!=filter_name:continue
                identity=task+':'+str(row['doc_id'])
                metrics[metric].append(dict(id=identity,score=float(row[key])))
                generation=[v['arg_1'] for v in row['arguments'].values() if isinstance(v,dict) and 'arg_1' in v]
                if len(generation)!=1:raise ValueError('unexpected request schema')
                inputs.append(dict(metric=metric,id=identity,doc_hash=row['doc_hash'],
                                   prompt_hash=row['prompt_hash'],target_hash=row['target_hash'],
                                   generation=generation[0],filter=filter_name))
    summary=root/'multiple/multiple_results.json';retain(summary)
    multiple=json.loads(summary.read_text())
    configs['multiple']={k:v for k,v in multiple['config'].items() if k not in ('model','model_path','base_url','run_tag')}
    for language in sorted(multiple['languages']):
        gen=root/f'multiple/generations_{language}.jsonl'
        exe=root/f'multiple/executions_{language}.jsonl'
        retain(gen);retain(exe)
        generated={row['name']:row for row in map(json.loads,gen.open())}
        executions=list(map(json.loads,exe.open()))
        if len(generated)!=len(executions) or generated.keys()!={r['name'] for r in executions}:
            raise ValueError('missing or duplicate MultiPL-E execution')
        for row in executions:
            original=re.fullmatch(r'HumanEval_(\d+)_.+',row['name'])
            if not original:raise ValueError('unrecognized original problem ID')
            identity='HumanEval/'+original[1]
            metrics['MultiPL-E'].append(dict(id=identity,language=language,score=float(row['status']=='OK')))
            source=generated[row['name']]
            inputs.append(dict(metric='MultiPL-E',id=identity,language=language,
                               prompt_sha256=sha(source['prompt']),tests_sha256=sha(source['tests'])))
    for metric,rows in metrics.items():
        if not rows:raise ValueError('missing metric '+metric)
        identities=[(r['id'],r.get('language')) for r in rows]
        if len(set(identities))!=len(identities):raise ValueError('duplicate IDs: '+metric)
    protocol=dict(backend=backend_protocol,configs=configs,
                  inputs=sorted(inputs,key=lambda x:(x['metric'],x['id'],x.get('language',''))))
    return dict(protocol_sha256=sha(protocol),metrics=metrics,protocol=protocol,source_files_sha256=sources,model_metadata=model_metadata)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--results-dir',type=Path,required=True)
    p.add_argument('--backend-protocol',required=True,
                   help='exact serving version/topology/tokenization contract from run record; keep fixed')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=export(a.results_dir,a.backend_protocol)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(protocol_sha256=result['protocol_sha256'],counts={k:len(v) for k,v in result['metrics'].items()})))
