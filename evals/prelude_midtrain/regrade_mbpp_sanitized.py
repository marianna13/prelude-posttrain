"""Diagnostic MBPP+ re-grade with each completion cut back to its longest parseable prefix.

Not part of the frozen gate (the noninferiority verdict uses the registered protocol unchanged). The MBPP+
base protocol stops only at `\\nassert`, `\\nprint(`, `\\nif __name__`, `\\ndef main(` and a closing docstring quote, so a model that
writes a correct function and then runs on (numbered copies, `File: ...` paths) until max_gen_toks is cut
mid-line and fails as a syntax error. This re-grade separates that stopping artifact from coding ability, roughly
like EvalPlus' sanitizer. The raw completions are also re-graded as a control; they must reproduce the recorded
pass@1 exactly.

  python regrade_mbpp_sanitized.py --out OUT.json NAME=SAMPLES_JSONL [NAME=SAMPLES_JSONL ...]
"""
import argparse
import ast
import json
import multiprocessing
import os
from pathlib import Path

import evaluate


def longest_parseable_prefix(code):
    lines = code.split('\n')
    for cut in range(len(lines), 0, -1):
        prefix = '\n'.join(lines[:cut])
        try:
            ast.parse(prefix)
            return prefix
        except SyntaxError:
            continue
    return ''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    p.add_argument('runs', nargs='+')
    a = p.parse_args()
    # as score_gates.py: filelock refuses plain fork from code_eval workers (infrastructure_adjustments.json)
    multiprocessing.set_start_method('forkserver', force=True)
    multiprocessing.set_forkserver_preload(['numpy'])
    code_eval = evaluate.load('code_eval')
    report = {}
    for spec in a.runs:
        name, path = spec.split('=', 1)
        rows = [json.loads(line) for line in open(path)]
        refs = ['\n'.join(r['doc']['test_imports']) + '\n' + r['doc']['test'] for r in rows]
        raw = [r['filtered_resps'][0] for r in rows]
        sanitized = [longest_parseable_prefix(c) for c in raw]
        scores = {}
        for variant, codes in (('raw', raw), ('sanitized', sanitized)):
            _, detail = code_eval.compute(references=refs, predictions=[[c] for c in codes], k=[1],
                                          num_workers=a.workers, timeout=20.0)
            scores[variant] = [int(detail[i][0][1]['passed']) for i in range(len(rows))]
        recorded = sum(r['pass_at_1'] > 0 for r in rows)
        if sum(scores['raw']) != recorded:
            raise ValueError(f'{name}: raw re-grade {sum(scores["raw"])} != recorded {recorded}')
        changed = sum(s != c for s, c in zip(sanitized, raw))
        report[name] = dict(samples=path, problems=len(rows), recorded_pass=recorded,
                            raw_pass=sum(scores['raw']), sanitized_pass=sum(scores['sanitized']),
                            sanitized_pct=100 * sum(scores['sanitized']) / len(rows), completions_cut=changed,
                            per_doc={r['doc_id']: [scores['raw'][i], scores['sanitized'][i]] for i, r in enumerate(rows)})
        print(name, {k: v for k, v in report[name].items() if k not in ('per_doc', 'samples')}, flush=True)
    a.out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
