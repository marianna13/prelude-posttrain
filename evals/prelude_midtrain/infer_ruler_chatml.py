"""Diagnostic: replay the frozen RULER documents in ChatML (thinking off) instead of plain text.

Each frozen plain-text prompt is <bos> + doc['input'] + ' ' + doc['gen_prefix']. The ChatML variant puts
doc['input'] in a user turn and starts the assistant turn with the empty think block and gen_prefix:
    <bos><|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{gen_prefix}
The wrapper adds ~15 tokens and the 16K/32K inputs have no headroom, so the window is context + 64.
Same documents, references, decoding and scorer (score_gates.py) as the gate; this is not a gate result.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

EXTRA = 64


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--context', type=int, required=True)
    p.add_argument('--tp', type=int, default=4)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    manifest = json.loads((a.inputs / 'inputs-sha256.json').read_text())
    grouped = defaultdict(list)
    plain_mismatch = 0
    for name, checksum in sorted(manifest.items()):
        path = a.inputs / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError('changed frozen benchmark input: ' + name)
        for line in path.open():
            row = json.loads(line)
            doc = row['doc']
            plain = tok(tok.bos_token + doc['input'] + ' ' + doc['gen_prefix'], add_special_tokens=False)['input_ids']
            plain_mismatch += plain != row['prompt_ids']
            text = tok.apply_chat_template([{'role': 'user', 'content': doc['input']}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) + doc['gen_prefix']
            ids = tok(text, add_special_tokens=False)['input_ids']
            if len(ids) + int(row['max_tokens']) > a.context + EXTRA:
                raise ValueError('chat prompt exceeds context + %d: %s %s' % (EXTRA, row['task'], row['id']))
            row = dict(row, prompt_ids=ids, prompt_tokens=len(ids), format='chatml-nothink',
                       prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
            grouped[int(row['max_tokens'])].append(row)
    n = sum(map(len, grouped.values()))
    print(f'{n} prompts; plain-text re-encoding differs from frozen IDs for {plain_mismatch}')
    first = next(iter(grouped.values()))[0]
    print('example prompt tail:', repr(tok.decode(first['prompt_ids'][-40:])))
    a.out.mkdir(parents=True, exist_ok=True)
    metadata = dict(model=a.model, context=a.context, max_model_len=a.context + EXTRA, tp=a.tp,
                    format='chatml-nothink', input_checksums=manifest, plain_reencode_mismatch=plain_mismatch,
                    versions={x: importlib.metadata.version(x) for x in ('vllm', 'transformers', 'torch')},
                    decoding=dict(temperature=0, seed=20261018, skip_special_tokens=False,
                                  stop_token_ids=[2, 262146], truncate_prompt_tokens=None))
    (a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    from vllm import LLM, SamplingParams
    model = LLM(model=a.model, tokenizer=a.model, tensor_parallel_size=a.tp,
                max_model_len=a.context + EXTRA, gpu_memory_utilization=.90, max_num_seqs=8,
                enforce_eager=True, trust_remote_code=False, seed=20261018)
    start = time.monotonic()
    with (a.out / 'outputs.jsonl').open('x') as f:
        for budget, rows in sorted(grouped.items()):
            params = SamplingParams(temperature=0, max_tokens=budget, seed=20261018,
                                    stop_token_ids=[2, 262146], skip_special_tokens=False)
            for offset in range(0, len(rows), 32):
                batch = rows[offset:offset + 32]
                outputs = model.generate([{'prompt_token_ids': r['prompt_ids']} for r in batch], params)
                for row, result in zip(batch, outputs, strict=True):
                    raw = result.outputs[0]
                    record = {k: v for k, v in row.items() if k != 'prompt_ids'}
                    record.update(raw=raw.text, generated_ids=list(raw.token_ids),
                                  finish_reason=raw.finish_reason, stop_reason=raw.stop_reason)
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
    metadata.update(elapsed_seconds=time.monotonic() - start, examples=n, status='inference-complete')
    (a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
