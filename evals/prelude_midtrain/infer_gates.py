"""Inference only on frozen token IDs. No truncation; retain special tokens/raw outputs."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--context',type=int,required=True)
    p.add_argument('--tp',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    manifest=json.loads((a.inputs/'inputs-sha256.json').read_text())
    grouped=defaultdict(list)
    for name,checksum in sorted(manifest.items()):
        path=a.inputs/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:
            raise ValueError('changed frozen benchmark input: '+name)
        for line in path.open():
            row=json.loads(line)
            if row['context']!=a.context or len(row['prompt_ids'])+row['max_tokens']>a.context:
                raise ValueError('context mismatch or prompt would truncate')
            grouped[row['max_tokens']].append(row)
    a.out.mkdir(parents=True,exist_ok=True)
    metadata=dict(model=a.model,context=a.context,tp=a.tp,input_checksums=manifest,
                  versions={x:importlib.metadata.version(x) for x in ('vllm','transformers','torch')},
                  decoding=dict(temperature=0,seed=20261018,skip_special_tokens=False,
                                stop_token_ids=[2,262146],truncate_prompt_tokens=None))
    def key(row):
        return (row['suite'],row['task'],row['id'],row.get('thinking'),row['context'])
    done=set()
    if a.resume and (a.out/'outputs.jsonl').exists():
        old=json.loads((a.out/'metadata.json').read_text())
        for field in ('model','context','tp','input_checksums','versions','decoding'):
            if old[field]!=metadata[field]:
                raise ValueError('resume protocol changed: '+field)
        raw=(a.out/'outputs.jsonl').read_bytes()
        # A killed job may leave an incomplete final JSON line; retain every
        # complete record and recompute only the interrupted request.
        complete=raw[:raw.rfind(b'\n')+1]
        if complete!=raw:
            (a.out/'outputs.jsonl').write_bytes(complete)
        for line in complete.splitlines():
            identity=key(json.loads(line))
            if identity in done:
                raise ValueError('duplicate result in resume file')
            done.add(identity)
        expected={key(r) for group in grouped.values() for r in group}
        if not done<=expected:
            raise ValueError('existing result is outside frozen input IDs')
        if done==expected:
            print('all frozen inputs already complete')
            return
        grouped={budget:[r for r in rows if key(r) not in done] for budget,rows in grouped.items()}
    (a.out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    from vllm import LLM, SamplingParams
    model=LLM(model=a.model,tokenizer=a.model,tensor_parallel_size=a.tp,
              max_model_len=a.context,gpu_memory_utilization=.90,max_num_seqs=8,
              enforce_eager=True,trust_remote_code=False,seed=20261018)
    start=time.monotonic()
    with (a.out/'outputs.jsonl').open('a' if a.resume else 'x') as f:
        for budget,rows in sorted(grouped.items()):
            params=SamplingParams(temperature=0,max_tokens=budget,seed=20261018,
                                  stop_token_ids=[2,262146],skip_special_tokens=False)
            for offset in range(0,len(rows),32):
                batch=rows[offset:offset+32]
                outputs=model.generate([{'prompt_token_ids':r['prompt_ids']} for r in batch],params)
                for row,result in zip(batch,outputs,strict=True):
                    if list(result.prompt_token_ids)!=row['prompt_ids']:
                        raise ValueError('backend changed prompt token IDs')
                    raw=result.outputs[0]
                    record={k:v for k,v in row.items() if k!='prompt_ids'}
                    record.update(raw=raw.text,generated_ids=list(raw.token_ids),
                                  finish_reason=raw.finish_reason,stop_reason=raw.stop_reason)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n')
                f.flush()
    metadata.update(elapsed_seconds=time.monotonic()-start,
                    examples=len(done)+sum(map(len,grouped.values())),status='inference-complete')
    (a.out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    main()
