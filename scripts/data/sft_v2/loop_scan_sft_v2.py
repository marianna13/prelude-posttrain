"""Scan SFT v2 normalized rows for degenerate reasoning: repetition loops and unfinished turns.

Per trained assistant turn with reasoning_content:
  tail_z   zlib ratio of the last 4000 chars of reasoning (looping eval outputs were < 0.15)
  rep_line most frequent line (>=20 chars) count among the last 60 non-empty lines
  loop     tail_z < 0.15 (rep_line >= 8 alone flags ordinary lists, so it is reported, not used)
  garbage  CJK junk such as '极' in the last 50 chars (truncated-generation artifact)
  no_answer reasoning present but content empty (turn never answers)
flagged_turns.jsonl gives uuid plus part/row: the rendered id is '<mix_source>-<row>' (1-based row of that part,
prerender_levanter.py), which pack_prelude_stage_megatron.py --exclude-rows matches.
Usage: loop_scan_sft_v2.py NORMALIZED_DIR OUT_DIR [PROCS] [MAX_FILES_PER_SOURCE]
"""
import collections
import glob
import json
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq


def turn_stats(r):
    tail = r[-4000:]
    b = tail.encode()
    z = len(zlib.compress(b)) / max(1, len(b))
    lines = [x.strip() for x in r.splitlines() if x.strip()][-60:]
    c = collections.Counter(x for x in lines if len(x) >= 20)
    rep = max(c.values()) if c else 0
    return z, rep


def scan(path):
    src = path.split('/')[-2]
    part = f"{src}/{os.path.basename(path)[:-len('.parquet')]}"
    out = dict(src=src, rows=0, turns=0, loop=0, rep=0, garbage=0, no_answer=0, loop_rows=0, rlen=[], zs=[], ex=[], flags=[])
    t = pq.read_table(path, columns=['conversations', 'uuid'])
    for n, (conv, uid) in enumerate(zip(t.column('conversations').to_pylist(), t.column('uuid').to_pylist()), 1):
        out['rows'] += 1
        row_loop = False
        for m in conv:
            if m['role'] != 'assistant' or not m['train_loss'] or not m['reasoning_content']:
                continue
            r = m['reasoning_content']
            z, rep = turn_stats(r)
            out['turns'] += 1
            out['rlen'].append(len(r)); out['zs'].append(z)
            loop = z < 0.15
            out['rep'] += rep >= 8
            junk = any('\u4e00' <= ch <= '\u9fff' for ch in r[-50:]) and not any('\u4e00' <= ch <= '\u9fff' for ch in r[:2000])
            out['garbage'] += junk
            if loop or junk:
                out['flags'].append(dict(uuid=uid, part=part, row=n, loop=bool(loop), garbage=bool(junk), tail_z=round(z, 4), chars=len(r)))
            if loop:
                out['loop'] += 1; row_loop = True
                if len(out['ex']) < 3:
                    out['ex'].append(dict(uuid=uid, z=round(z, 3), rep=rep, chars=len(r), tail=r[-300:]))
            if not (m['content'] or '').strip() and not m['tool_calls']:
                out['no_answer'] += 1
        out['loop_rows'] += row_loop
    return out


def main():
    root, outd = sys.argv[1], sys.argv[2]
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    files = []
    for d in sorted(glob.glob(f'{root}/*/')):
        fs = sorted(glob.glob(f'{d}*.parquet'))
        files += fs[:cap] if cap else fs
    os.makedirs(outd, exist_ok=True)
    agg, flags = {}, []
    with Pool(procs) as p:
        for o in p.imap_unordered(scan, files):
            a = agg.setdefault(o['src'], dict(rows=0, turns=0, loop=0, rep=0, garbage=0, no_answer=0, loop_rows=0, rlen=[], zs=[], ex=[]))
            for k in ['rows', 'turns', 'loop', 'rep', 'garbage', 'no_answer', 'loop_rows']:
                a[k] += o[k]
            a['rlen'] += o['rlen']; a['zs'] += o['zs']
            flags += o['flags']
            a['ex'] += o['ex'][:3 - len(a['ex'])] if len(a['ex']) < 3 else []
    summary = {}
    for s, a in sorted(agg.items()):
        rl, zs = np.array(a['rlen'] or [0]), np.array(a['zs'] or [1.0])
        summary[s] = dict(rows=a['rows'], reasoning_turns=a['turns'], loop_turns=a['loop'],
                          loop_turn_pct=round(100 * a['loop'] / max(1, a['turns']), 3),
                          loop_rows=a['loop_rows'], rep_line_turns=a['rep'], garbage_tail_turns=a['garbage'], no_answer_turns=a['no_answer'],
                          reasoning_chars_p50=int(np.percentile(rl, 50)), reasoning_chars_p99=int(np.percentile(rl, 99)),
                          tail_z_p1=round(float(np.percentile(zs, 1)), 3), tail_z_p50=round(float(np.percentile(zs, 50)), 3),
                          examples=a['ex'])
        print(s, {k: v for k, v in summary[s].items() if k != 'examples'}, flush=True)
    with open(f'{outd}/flagged_turns.jsonl', 'w') as f:
        f.writelines(json.dumps(x) + '\n' for x in flags)
    json.dump(summary, open(f'{outd}/summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
