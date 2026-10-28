"""SWE task overlap for Prelude SFT v2 (issue 0058).

Targets: Scale-SWE-Verified (agentic RL pool) and SWE-bench Verified (agentic eval).
Queries: nebius SWE-rebench-OpenHands trajectories, AgentTrove SWE rows (swesmith / r2egym / SWEGym / Inferred Bugs),
and Scale-SWE itself against SWE-bench Verified.

Two signals:
- id: same (owner/repo, PR number), parsed from instance ids / github_url;
- text: the query issue text shares >= MIN_SHARED word 13-grams with one target issue (after dropping 13-grams
  that occur in > COMMON target issues, i.e. templates).
Writes <out>/swe_overlap.json and per-query match lists.
"""
import argparse
import collections
import glob
import json
import os
import re

import pyarrow.parquet as pq

N, MIN_SHARED, COMMON = 13, 20, 5
WORD = re.compile(r'[a-z0-9]+')


def shingles(text):
    w = WORD.findall((text or '').lower())
    return {hash(' '.join(w[i:i + N])) for i in range(len(w) - N + 1)}


def key_dunder(iid):  # owner__repo-123 -> (owner/repo, 123)
    m = re.match(r'^(.+?)__(.+)-(\d+)$', iid or '')
    return (f'{m[1]}/{m[2]}'.lower(), int(m[3])) if m else None


def read(paths, cols):
    out = []
    for f in paths:
        out += pq.read_table(f, columns=cols).to_pylist()
    return out


class Index:
    def __init__(self, name, items):  # items: (task_id, key, text)
        self.name, self.ids = name, [i for i, _, _ in items]
        self.keys = {k: i for i, k, _ in items if k}
        post = collections.defaultdict(list)
        for n, (_, _, text) in enumerate(items):
            for s in shingles(text):
                post[s].append(n)
        self.post = {s: v for s, v in post.items() if len(v) <= COMMON}

    def match(self, key, text):
        hits = []
        if key and key in self.keys:
            hits.append(('id', self.keys[key], None))
        c = collections.Counter(n for s in shingles(text) for n in self.post.get(s, ()))
        for n, shared in c.most_common(3):
            if shared >= MIN_SHARED:
                hits.append(('text', self.ids[n], shared))
        return hits


def issue_from_openhands(msgs):
    u = next((m.get('content') or '' for m in msgs if m.get('role') == 'user'), '')
    m = re.search(r'<issue_description>(.*?)</issue_description>', u, re.S)
    return m[1] if m else u


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    R = a.root
    swebv = read(glob.glob(f'{R}/princeton-nlp/SWE-bench_Verified/data/*.parquet'),
                 ['instance_id', 'repo', 'problem_statement'])
    scale = read(glob.glob(f'{R}/PrimeIntellect/Scale-SWE-Verified/data/*.parquet'),
                 ['instance_id', 'github_url', 'problem_statement'])

    def scale_key(r):
        m = re.search(r'_pr(\d+)$', r['instance_id'] or '')
        repo = re.sub(r'^https://github\.com/', '', (r['github_url'] or '').rstrip('/')).lower()
        return (repo, int(m[1])) if m and repo else None

    idx_v = Index('swebench_verified', [(r['instance_id'], key_dunder(r['instance_id']), r['problem_statement'])
                                        for r in swebv])
    idx_s = Index('scale_swe', [(r['instance_id'], scale_key(r), r['problem_statement']) for r in scale])
    report = dict(params=dict(ngram=N, min_shared=MIN_SHARED, common_cutoff=COMMON),
                  targets=dict(swebench_verified=len(swebv), scale_swe=len(scale),
                               scale_swe_keys_parsed=len(idx_s.keys), swebench_verified_repos=len({r['repo'] for r in swebv})))
    os.makedirs(a.out, exist_ok=True)

    def run(name, queries):  # queries: (query_id, key, repo, text, extra)
        res = dict(queries=len(queries))
        detail = []
        swebv_repos = {r['repo'].lower() for r in swebv}
        for tgt in (idx_s, idx_v):
            if name == 'scale_swe' and tgt is idx_s:
                continue
            by_id = by_text = either = 0
            for qid, key, repo, text, extra in queries:
                hits = tgt.match(key, text)
                kinds = {h[0] for h in hits}
                by_id += 'id' in kinds; by_text += 'text' in kinds; either += bool(hits)
                if hits:
                    detail.append(dict(query=qid, target=tgt.name, hits=hits, **extra))
            res[tgt.name] = dict(id_match=by_id, text_match=by_text, any=either)
        res['same_repo_as_swebench_verified'] = sum(1 for _, _, repo, _, _ in queries if repo and repo in swebv_repos)
        report[name] = res
        with open(f'{a.out}/swe_overlap_{name}.jsonl', 'w') as f:
            for d in detail:
                f.write(json.dumps(d) + '\n')
        print(name, json.dumps(res), flush=True)

    run('scale_swe', [(r['instance_id'], scale_key(r), (scale_key(r) or ('',))[0], r['problem_statement'], {})
                      for r in scale])
    neb = read(glob.glob(f'{R}/nebius/SWE-rebench-openhands-trajectories/*.parquet'),
               ['instance_id', 'repo', 'trajectory', 'resolved'])
    seen, q = set(), []
    for r in neb:  # one query per instance; record how many resolved trajectories it carries
        if r['instance_id'] in seen:
            continue
        seen.add(r['instance_id'])
        q.append((r['instance_id'], key_dunder(r['instance_id']), (r['repo'] or '').lower(),
                  issue_from_openhands(r['trajectory']), {}))
    resolved_by_inst = collections.Counter(r['instance_id'] for r in neb if r['resolved'])
    run('nebius_instances', q)
    report['nebius_instances']['resolved_trajectories'] = sum(resolved_by_inst.values())
    del neb

    at_files = sorted(glob.glob(f'{R}/open-thoughts/AgentTrove/**/*.parquet', recursive=True))
    swe_src = {'swesmith', 'r2egym', 'SWEGym', 'Inferred Bugs'}
    q, seen = [], set()
    for f in at_files:
        t = pq.read_table(f, columns=['original_source', 'task', 'instruction', 'conversations'])
        for r in t.to_pylist():
            if r['original_source'] not in swe_src:
                continue
            text = r['instruction'] or next((m.get('content') or '' for m in (r['conversations'] or [])
                                             if m.get('role') == 'user'), '')
            task = r['task'] or ''
            if (r['original_source'], task) in seen:
                continue
            seen.add((r['original_source'], task))
            q.append((task, key_dunder(task), '', text, dict(original_source=r['original_source'])))
    run('agenttrove_swe_tasks', q)
    by_src = collections.Counter(x[4]['original_source'] for x in q)
    report['agenttrove_swe_tasks']['tasks_by_source'] = dict(by_src)
    with open(f'{a.out}/swe_overlap.json', 'w') as f:
        json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
