"""Freeze benchmark inputs before GPU inference; run with the existing lm-eval environment."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import datasets
import lm_eval
from lm_eval.tasks._yaml_loader import load_yaml as load_yaml_config
from transformers import AutoTokenizer


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write(path, rows):
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        temp=Path(f.name)
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    try:
        os.link(temp,path)  # atomic publication; never replace a frozen input
    finally:
        temp.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--suite', choices=['chat', 'ruler'], required=True)
    p.add_argument('--context', type=int, default=4096)
    p.add_argument('--task', default=None)
    p.add_argument('--max-tokens', type=int, default=1024, help='chat generation budget (frozen gate: 4096 / 1024)')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    frozen=a.out/'inputs-sha256.json'
    if frozen.exists():
        checksums=json.loads(frozen.read_text())
        expected=6 if a.suite=='chat' else (1 if a.task else 13)
        if len(checksums)!=expected:
            raise ValueError('frozen directory has a different task set; use a new output path')
        for name,checksum in checksums.items():
            if hashlib.sha256((a.out/name).read_bytes()).hexdigest()!=checksum:
                raise ValueError('frozen benchmark input changed: '+name)
            if a.suite=='ruler' and not name.endswith(f'-{a.context}.jsonl'):
                raise ValueError('frozen context differs; use a new output path')
        print('verified frozen inputs; replay these files without regeneration',flush=True)
        return
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    ruler = Path(lm_eval.__file__).parent / 'tasks/ruler'
    if a.suite == 'ruler':
        tasks = ['niah_single_1','niah_single_2','niah_single_3', 'niah_multikey_1',
                 'niah_multikey_2','niah_multikey_3','niah_multiquery','niah_multivalue',
                 'ruler_vt','ruler_cwe','ruler_fwe','ruler_qa_squad','ruler_qa_hotpot']
        for task in tasks:
            if a.task and task != a.task:
                continue
            dest = a.out / f'{task}-{a.context}.jsonl'
            if dest.exists():
                continue
            filename = task.removeprefix('ruler_')
            config = load_yaml_config(str(ruler / f'{filename}.yaml'))
            budget = config['generation_kwargs']['max_gen_toks']
            generator_window = a.context
            while True:
                ds = config['custom_dataset'](tokenizer=a.tokenizer, max_seq_lengths=[generator_window])['test']
                rows = []
                max_excess = 0
                for i, doc in enumerate(ds.select(range(100))):
                    prompt = doc['input'] + ' ' + doc.get('gen_prefix', '')
                    ids = tok.encode(prompt, add_special_tokens=True)
                    max_excess = max(max_excess, len(ids)+budget-a.context)
                    rows.append(dict(suite='ruler', task=task, id=i, context=a.context,
                                     generator_window=generator_window,
                                     prompt_ids=ids, prompt_sha256=digest(prompt),
                                     prompt_tokens=len(ids), max_tokens=budget, doc=doc))
                if not max_excess:
                    break
                print(f'{task}: upstream prompt overflow {max_excess}; regenerate whole task with headroom',flush=True)
                generator_window -= max_excess+8
                if generator_window < a.context-1024:
                    raise ValueError('unexpectedly large upstream sizing error')
            write(dest, rows)
            print(dest, len(rows), flush=True)
        hashes = {str(f.relative_to(ruler)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted(ruler.rglob('*')) if f.suffix in ('.py','.yaml')}
        (a.out/'ruler-source-hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    else:
        sources = [('GSM8K', 'openai/gsm8k', 'main'),
                   ('MATH500', 'HuggingFaceH4/MATH-500', 'default'),
                   ('HumanEval+', 'evalplus/humanevalplus', None)]
        for name, repo, subset in sources:
            ds = datasets.load_dataset(repo, subset, split='test')
            ranked = []
            for i, doc in enumerate(ds):
                identity = str(doc.get('task_id', doc.get('unique_id', doc.get('question', doc.get('problem')))))
                ranked.append((digest('20261018:'+name+':'+identity), i, identity))
            selected = sorted(ranked)[:100]
            for thinking in (True, False):
                rows=[]
                for _, i, identity in selected:
                    doc=dict(ds[i])
                    if name == 'GSM8K':
                        text=doc['question']+'\nSolve step by step. End with "The answer is NUMBER".'
                    elif name == 'MATH500':
                        text=doc['problem']+'\nShow your solution and put the final answer in \\boxed{}.'
                    else:
                        text='Complete this Python function. Return only the completed Python code.\n```python\n'+doc['prompt']+'\n```'
                    prompt=tok.apply_chat_template([{'role':'user','content':text}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=thinking)
                    ids=tok.encode(prompt, add_special_tokens=False)
                    if len(ids)+a.max_tokens >a.context:
                        raise ValueError(f'oversize {name}/{identity}')
                    rows.append(dict(suite='chat',task=name,id=identity,source_index=i,
                                     thinking=thinking,context=a.context,prompt_ids=ids,
                                     prompt_sha256=digest(prompt),prompt_tokens=len(ids),max_tokens=a.max_tokens,doc=doc))
                dest=a.out/f'chat-{name}-thinking{int(thinking)}.jsonl'
                write(dest,rows)
                print(dest,len(rows),flush=True)
    files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.out.glob('*.jsonl'))}
    (a.out/'inputs-sha256.json').write_text(json.dumps(files,indent=2)+'\n')


if __name__ == '__main__':
    main()
