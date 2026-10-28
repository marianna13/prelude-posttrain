#!/usr/bin/env python3
"""Freeze step-4 token exposure weights and audit a deterministic parent selection.

The selection is a reproducible recipe input for step 5, not a token-ID release.
Reserve stage 64K, then 32K, then 16K so shorter stages cannot exhaust scarce long
examples. Never reuse a parent across stages or splits. Fail on exhaustion.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, '/p/project1/laionize/marianna/dc_agent/.prelude_census_deps')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from census_curated_prelude_midtrain import (CASCADE, SPECIALIZED, STACK, TERMINAL, ULTRA, STAGES,
                                            OVERLAP, digest, sha, stage_counts, write)

CATEGORY_WEIGHTS = {'reasoning': .30, 'code_agent': .25, 'chat_if': .10,
                    'long_documents': .125, 'replay': .225}
DOCUMENT_WEIGHT = .65
BUDGETS = {16384: 500_000_000, 32768: 1_000_000_000, 65536: 2_000_000_000}
SEED = 'prelude-midtrain-v1-mixture-20261017'
REPLAY_WEIGHTS = {'dclm_english': .82*33.2/(33.2+24.5+10.9),
                  'nemotron_english': .82*24.5/(33.2+24.5+10.9),
                  'pdf_english': .82*10.9/(33.2+24.5+10.9), 'pdf_multilingual': .18}
SELECTION_SCHEMA = pa.schema([(n,pa.string()) for n in
    ['id','mix_source','pool','category','kind','split','split_group']] +
    [(n,pa.int64()) for n in ['context','original_tokens','seen_tokens','trainable_tokens','chunks']])


def query(con, sql, parameters=None):
    cur = con.execute(sql,parameters or [])
    columns = [x[0] for x in cur.description]
    return [dict(zip(columns,row)) for row in cur.fetchall()]


def availability(con, allow_short_chat):
    """Expose strict preferred buckets and any explicitly authorized short bucket."""
    result = []
    for context, minimum in STAGES:
        preferred = f'stage_{context}_chunks>0'
        effective = f'({preferred} OR category=\'chat_if\')' if allow_short_chat else f'({preferred})'
        result.extend(query(con, f'''SELECT pool,category,kind,split,{context} AS context,
            count(*) AS eligible_rows,sum(tokens) AS original_tokens,sum(trainable_tokens) AS trainable_tokens,
            sum(CASE WHEN {preferred} THEN stage_{context}_seen_tokens ELSE tokens END) AS seen_tokens,
            sum(CASE WHEN {preferred} THEN stage_{context}_chunks ELSE 1 END) AS chunks,
            count(*) FILTER(WHERE NOT ({preferred})) AS short_exception_rows,
            coalesce(sum(tokens) FILTER(WHERE NOT ({preferred})),0) AS short_exception_tokens
            FROM metrics WHERE eligible AND {effective} GROUP BY pool,category,kind,split ORDER BY ALL'''))
    return result


def priors(pools, available):
    """Explicit source priors; two natural sub-mixtures use measured clean counts."""
    prior = {p:0.0 for p in pools}
    fixed = {
        CASCADE+'/math':.13, CASCADE+'/science':.02,
        SPECIALIZED+'/Nemotron-Pretraining-InfiniByte-Reasoning':.06,
        SPECIALIZED+'/Nemotron-Pretraining-RQA':.04,
        SPECIALIZED+'/Nemotron-Pretraining-STEM-SFT':.03,
        SPECIALIZED+'/Nemotron-Pretraining-Math-Textbooks':.02,
        STACK+'/data/code':.04,
        SPECIALIZED+'/Nemotron-Pretraining-Scientific-Coding':.02,
        TERMINAL:.04, CASCADE+'/swe':.04, CASCADE+'/conversational_agent':.02,
        CASCADE+'/chat':.08, CASCADE+'/instruction_following':.02,
        STACK+'/data/long_documents':.0625,
        'FinePDFs-English-long/long_documents':.0625,
        'dclm-baseline-1.0/dclm_english':.225*REPLAY_WEIGHTS['dclm_english'],
        'Nemotron-CC-original-high/nemotron_english':.225*REPLAY_WEIGHTS['nemotron_english'],
        'FinePDFs-English-replay/pdf_english':.225*REPLAY_WEIGHTS['pdf_english'],
    }
    for p,w in fixed.items():
        if p not in pools:
            raise ValueError('missing known source pool: '+p)
        prior[p] = w
    counts = {(r['pool'],r['context']):r['original_tokens'] for r in available if r['split']=='train'}
    for prefix, total in [(ULTRA+'/', .09), ('FinePDFs-EU-replay/', .225*REPLAY_WEIGHTS['pdf_multilingual'])]:
        names = sorted(p for p in pools if p.startswith(prefix))
        capacity = {p:counts.get((p,65536),0) for p in names}
        if not names or sum(capacity.values())==0:
            raise ValueError('empty natural sub-mixture: '+prefix)
        for p in names:
            prior[p] = total*capacity[p]/sum(capacity.values())
    if abs(sum(prior.values())-1)>1e-10:
        raise ValueError('unknown pools or invalid prior sum')
    return prior


def solve_weights(registry, available, prior):
    """Relative squared deviation under category, format and one-pass caps.

    Suffix capacity inequalities are sufficient for disjoint nested length pools.
    Native tokens supply a conservative cap (virtual continuation context can
    only increase seen exposure). Actual whole-parent selection is checked next.
    """
    names = sorted(registry)
    n = len(names)
    counts = {(r['pool'],r['context']):r['original_tokens'] for r in available if r['split']=='train'}
    contexts = [c for c,_ in STAGES]
    caps = {p:min(counts.get((p,c),0)/sum(BUDGETS[x] for x in contexts[i:])
                  for i,c in enumerate(contexts)) for p in names}
    eq, rhs = [], []
    for category,weight in CATEGORY_WEIGHTS.items():
        eq.append([int(registry[p]['category']==category) for p in names]+[0]*n)
        rhs.append(weight)
    eq.append([int(registry[p]['kind']=='document') for p in names]+[0]*n)
    rhs.append(DOCUMENT_WEIGHT)
    # Freeze the replay approximation, including clean within-family language weights.
    for i,p in enumerate(names):
        if registry[p]['category']=='replay':
            eq.append([int(j==i) for j in range(n)]+[0]*n)
            rhs.append(prior[p])
    # Reallocating code must preserve the measured common-stage language mix.
    # Otherwise a degenerate L1 optimum can assign every extra token to one language.
    ultra = [p for p in names if p.startswith(ULTRA+'/')]
    ultra_prior = sum(prior[p] for p in ultra)
    if ultra_prior:
        for p in ultra:
            fraction = prior[p]/ultra_prior
            eq.append([int(q==p)-fraction*int(q in ultra) for q in names]+[0]*n)
            rhs.append(0.0)
    prior_array = np.array([prior[p] for p in names])
    absolute = np.block([[np.eye(n),-np.eye(n)],[-np.eye(n),-np.eye(n)]])
    result = linprog(np.r_[np.zeros(n),np.ones(n)], A_ub=absolute, b_ub=np.r_[prior_array,-prior_array],
                     A_eq=np.array(eq),b_eq=np.array(rhs),bounds=[(0,caps[p]) for p in names]+[(0,None)]*n,
                     method='highs', options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    if not result.success:
        return None, {'status':'infeasible','message':result.message,'one_pass_weight_caps':caps,
                      'category_caps':{cat:sum(caps[p] for p in names if registry[p]['category']==cat)
                                       for cat in CATEGORY_WEIGHTS}}
    # A strictly convex objective removes arbitrary ties between other sources.
    # Reduce dependent equalities (category/replay/language constraints overlap)
    # and fixed-zero variables before SLSQP.
    active = np.array([i for i,p in enumerate(names) if caps[p]>0],dtype=int)
    matrix = np.array(eq)[:,:n][:,active]
    _,triangular,pivots = qr(matrix.T,mode='economic',pivoting=True)
    rank = int(np.sum(np.abs(np.diag(triangular))>1e-10))
    independent = sorted(pivots[:rank])
    matrix,rhs_reduced = matrix[independent],np.array(rhs)[independent]
    desired = prior_array[active]
    scale = np.maximum(desired,1e-6)
    def objective(x):
        delta = x-desired
        return .5*np.dot(delta/scale,delta)
    optimum = minimize(objective,result.x[:n][active],jac=lambda x:(x-desired)/scale,
        method='SLSQP',bounds=Bounds(np.zeros(len(active)),[min(1,caps[names[i]]) for i in active]),
        constraints=LinearConstraint(matrix,rhs_reduced,rhs_reduced),
        options={'ftol':1e-12,'maxiter':1000})
    if not optimum.success:
        raise ValueError('unique balanced mixture optimization failed: '+optimum.message)
    solution = np.zeros(n)
    solution[active] = optimum.x
    weights = {p:round(float(max(0,w)),12) for p,w in zip(names,solution)}
    validate_weights(weights, registry)
    return weights, {'status':'feasible','objective':'minimum sum((weight-prior)^2 / prior), with fixed Ultra language ratios',
                     'total_absolute_adjustment':sum(abs(weights[p]-prior[p]) for p in names),
                     'one_pass_weight_caps':caps,'maximum_parent_repetitions':1,
                     'solver':'HiGHS feasibility, SLSQP strictly convex source balancing',
                     'optimization_objective':float(optimum.fun),
                     'ultra_language_ratios_frozen':bool(ultra_prior)}


def validate_weights(weights,registry):
    if set(weights)!=set(registry) or any(not math.isfinite(w) or w<0 for w in weights.values()):
        raise ValueError('missing/unknown pools, negative or nonfinite weights')
    if abs(sum(weights.values())-1)>1e-8:
        raise ValueError('mixture weights must sum to one')
    for category,target in CATEGORY_WEIGHTS.items():
        if abs(sum(weights[p] for p in weights if registry[p]['category']==category)-target)>1e-8:
            raise ValueError('category target not met: '+category)
    if abs(sum(weights[p] for p in weights if registry[p]['kind']=='document')-DOCUMENT_WEIGHT)>1e-8:
        raise ValueError('format target not met')


def choose_parent(row, context, remaining):
    chunks,seen,trained = stage_counts(row['tokens'],row['trainable_tokens'],context)
    if seen>remaining:
        return None
    return {k:row[k] for k in ['id','mix_source','pool','category','kind','split','split_group']} | {
        'context':context,'original_tokens':row['tokens'],'seen_tokens':seen,'trainable_tokens':trained,'chunks':chunks}


def select(con,out,weights,registry,allow_short_chat):
    """Whole-parent shuffled sampling, token quotas, no replacement across stages."""
    con.execute('DROP TABLE IF EXISTS candidates')
    clause = "OR category='chat_if'" if allow_short_chat else ''
    con.execute(f'''CREATE TABLE candidates AS SELECT *,sha256(?||id) AS sample_order
        FROM metrics WHERE eligible AND split='train' AND (stage_16384_chunks>0 {clause})''',[SEED])
    selected,stage_reports = set(),[]
    for context,minimum in reversed(STAGES):
        budget = BUDGETS[context]
        quotas = {p:int(math.floor(w*budget)) for p,w in weights.items()}
        remaining = dict(quotas)
        counters = {p:Counter() for p in weights}
        output = out/f'selection-{context}.parquet'
        buf = []
        cur = con.execute(f'''SELECT * FROM candidates WHERE stage_{context}_chunks>0 {clause}
                              ORDER BY pool,sample_order,id''')
        columns = [d[0] for d in cur.description]
        with pq.ParquetWriter(str(output)+'.tmp',SELECTION_SCHEMA,compression='zstd') as writer:
            while batch := cur.fetchmany(8192):
                for values in batch:
                    row = dict(zip(columns,values))
                    p = row['pool']
                    if remaining[p]<=0 or row['id'] in selected:
                        continue
                    value = choose_parent(row,context,remaining[p])
                    if value is None:
                        counters[p]['skipped_above_remaining_quota'] += 1
                        continue
                    selected.add(row['id'])
                    remaining[p] -= value['seen_tokens']
                    counters[p]['parents'] += 1
                    for key in ['seen_tokens','trainable_tokens','chunks','original_tokens']:
                        counters[p][key] += value[key]
                    if row['tokens'] < minimum and row['category']!='replay':
                        counters[p]['short_exception_parents'] += 1
                        counters[p]['short_exception_seen_tokens'] += value['seen_tokens']
                    buf.append(value)
                    if len(buf)>=4096:
                        writer.write_table(pa.Table.from_pylist(buf,schema=SELECTION_SCHEMA))
                        buf.clear()
            if buf:
                writer.write_table(pa.Table.from_pylist(buf,schema=SELECTION_SCHEMA))
        Path(str(output)+'.tmp').replace(output)
        actual = sum(c['seen_tokens'] for c in counters.values())
        if actual<=0 or (budget-actual)/budget>.002:
            raise ValueError(f'stage {context} exhausted or budget shortfall exceeds 0.2%: {budget-actual}')
        shares = {p:counters[p]['seen_tokens']/actual for p in weights}
        for category,target in CATEGORY_WEIGHTS.items():
            share = sum(shares[p] for p in weights if registry[p]['category']==category)
            if abs(share-target)>.001:
                raise ValueError('actual sampled category share differs by >0.1 percentage point')
        document_share = sum(shares[p] for p in weights if registry[p]['kind']=='document')
        if abs(document_share-DOCUMENT_WEIGHT)>.001:
            raise ValueError('actual sampled format share differs by >0.1 percentage point')
        if any(abs(shares[p]-weights[p])>.0002 for p in weights):
            raise ValueError('actual per-pool token share differs by >0.02 percentage point')
        report = {'context':context,'budget':budget,'seen_tokens':actual,
                  'trainable_tokens':sum(c['trainable_tokens'] for c in counters.values()),
                  'budget_shortfall':budget-actual,'document_seen_share':document_share,
                  'source_counts':{p:dict(counters[p])|{'requested_weight':weights[p],
                      'actual_seen_share':shares[p], 'remaining_quota':remaining[p]} for p in weights},
                  'selection':output.name,'selection_sha256':sha(output),
                  'category_seen_shares':{cat:sum(shares[p] for p in weights if registry[p]['category']==cat)
                                          for cat in CATEGORY_WEIGHTS},
                  'category_trained_shares':{cat:sum(counters[p]['trainable_tokens'] for p in weights
                       if registry[p]['category']==cat)/sum(c['trainable_tokens'] for c in counters.values())
                       for cat in CATEGORY_WEIGHTS}}
        report['document_trained_share'] = sum(counters[p]['trainable_tokens'] for p in weights
                         if registry[p]['kind']=='document')/report['trainable_tokens']
        stage_reports.append(report)
        print(json.dumps({'context':context,'seen_tokens':actual,'selected_parents':len(selected),
                          'document_seen_share':document_share}),flush=True)
    stage_reports.sort(key=lambda s:s['context'])
    # Independent SQL checks of emitted records, rather than relying on the selection set.
    files = [str(out/s['selection']) for s in stage_reports]
    con.read_parquet(files).create_view('selected',replace=True)
    if con.execute("SELECT count(*)-count(DISTINCT id) FROM selected").fetchone()[0] or \
       con.execute("SELECT count(*) FROM selected WHERE split!='train'").fetchone()[0]:
        raise ValueError('repeated parent or held-out leakage')
    disagreement = con.execute('''SELECT count(*) FROM selected s LEFT JOIN candidates c USING(id)
        WHERE c.id IS NULL OR s.pool!=c.pool OR s.original_tokens!=c.tokens OR s.trainable_tokens!=c.trainable_tokens
        OR s.split_group!=c.split_group OR s.mix_source!=c.mix_source OR s.category!=c.category OR s.kind!=c.kind''').fetchone()[0]
    if disagreement:
        raise ValueError('selected parent differs from census')
    return stage_reports


def verify_census_report(report):
    if report.get('status')!='retained-census-complete':
        raise ValueError('complete retained census required')
    if report.get('build_id')!=digest({k:v for k,v in report.items() if k!='build_id'}):
        raise ValueError('retained census manifest identity changed')


def build(args):
    base,out = Path(args.census),Path(args.out)
    report = json.loads((base/'retained_census.json').read_text())
    verify_census_report(report)
    if sha(base/'census_input_manifest.json')!=report['census_input_manifest_sha256']:
        raise ValueError('census input manifest changed')
    def verify(item):
        key,expected = item
        if sha(base/'census'/key/'metrics.parquet')!=expected:
            raise ValueError('census artifact changed: '+key)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(verify,report['metrics_sha256'].items()))
    out.mkdir(parents=True,exist_ok=True)
    identity = {'retained_census_sha256':sha(base/'retained_census.json'), 'seed':SEED,'budgets':BUDGETS,
                'category_weights':CATEGORY_WEIGHTS,'document_weight':DOCUMENT_WEIGHT,
                'allow_short_chat':args.allow_short_chat,'overlap_tokens':OVERLAP,
                'code_sha256':{n:sha(Path(__file__).with_name(n)) for n in [Path(__file__).name,
                               'census_curated_prelude_midtrain.py','prelude_midtrain.py','prelude_lf_counting.py']},
                'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','scipy','duckdb','pyarrow']}}
    identity['build_id'] = digest(identity)
    ip = out/'mixture_input_manifest.json'
    if ip.exists() and json.loads(ip.read_text())!=json.loads(json.dumps(identity)):
        raise ValueError('different mixture build; choose another output directory')
    write(ip,identity)
    completed = out/'mixture_manifest.json'
    if completed.exists():
        old = json.loads(completed.read_text())
        if old.get('status')!='step4-complete' or old.get('mixture_build_id')!=identity['build_id'] or \
           old.get('build_id')!=digest({k:v for k,v in old.items() if k!='build_id'}):
            raise ValueError('existing mixture completion differs')
        for stage in old['stages']:
            if sha(out/stage['selection'])!=stage['selection_sha256']:
                raise ValueError('completed selection changed')
        print(json.dumps({'status':old['status'],'build_id':old['build_id'],'resumed':True}),flush=True)
        return
    con = duckdb.connect(str(out/'selection_work.duckdb'))
    con.execute(f'SET threads={args.threads}')
    con.execute("SET memory_limit='192GB'")
    con.execute("SET parquet_metadata_cache=true")
    con.read_parquet(str(base/'census/*/metrics.parquet'),hive_partitioning=False).create_view('metrics',replace=True)
    registry = {r['pool']:{k:r[k] for k in ['category','kind']} for r in report['groups']}
    available = availability(con,args.allow_short_chat)
    prior = priors(registry,available)
    weights,feasibility = solve_weights(registry,available,prior)
    write(out/'feasibility.json',{'available':available,'source_priors':prior,'feasibility':feasibility})
    if weights is None:
        raise ValueError('requested mixture is infeasible; see feasibility.json, do not silently change shares or repeat sources')
    stages = select(con,out,weights,registry,args.allow_short_chat)
    base_tokens = {r['pool']:r['trainable_tokens'] for r in query(con,
        "SELECT pool,sum(trainable_tokens) trainable_tokens FROM metrics WHERE eligible AND split='train' GROUP BY pool")}
    exposures = {p:{'weight':weights[p], 'category':registry[p]['category'],'kind':registry[p]['kind'],
                    'trained_tokens':sum(s['source_counts'][p].get('trainable_tokens',0) for s in stages),
                    'seen_tokens':sum(s['source_counts'][p].get('seen_tokens',0) for s in stages),
                    'effective_trained_epochs':sum(s['source_counts'][p].get('trainable_tokens',0) for s in stages)/base_tokens[p]}
                 for p in weights}
    strict_availability = defaultdict(int)
    for group in report['groups']:
        if group['split']=='train' and group['eligible']:
            for context,_ in STAGES:
                strict_availability[(group['pool'],context)] += group[f'stage_{context}_seen_tokens']
    unavailable = [{'pool':p,'context':c,'requested_prior':prior[p],
                    'frozen_weight':weights[p],'short_chat_exception':args.allow_short_chat and registry[p]['category']=='chat_if'}
                   for p in sorted(registry) for c,_ in STAGES if not strict_availability[(p,c)]]
    result = {'schema_version':1,'status':'step4-complete','mixture_build_id':identity['build_id'],
              'census_build_id':report['census_build_id'],'retained_census_sha256':identity['retained_census_sha256'],
              'curation_manifest_sha256':report['curation_manifest_sha256'], 'weights_basis':'seen token exposure, including masked continuation context',
              'category_weights':CATEGORY_WEIGHTS,'format_weights':{'document':.65,'conversation':.35},
              'source_weights':weights,'source_priors':prior,'source_exposures':exposures,
              'stage_budgets':BUDGETS,'stages':stages,'availability':available,'feasibility':feasibility,
              'unavailable_preferred_length_buckets':unavailable,
              'policy':{'seed':SEED,'parent_order':'sha256(seed || canonical id), lexical id tie break',
                        'selection':'whole parents, token quotas, skip parents above remaining quota',
                        'reservation_order':[65536,32768,16384],'training_order':[16384,32768,65536],
                        'replacement':False,'exhaustion':'error; no resampling or weight fallback',
                        'continuation':report['policy']['continuation'],
                        'length_minima':{c:m for c,m in STAGES},'replay_maximum':4096,
                        'short_chat_exception':args.allow_short_chat,
                        'short_chat_reason':'fixed chat/IF allocation may use short whole conversations; counted separately, never long-range evidence' if args.allow_short_chat else None,
                        'category_format_tolerance':.001,'source_share_tolerance':.0002,'budget_shortfall_tolerance':.002},
              'checks':{'parent_ids_unique_across_stages':True,'train_only':True,'census_parent_agreement':True,
                        'actual_seen_and_trained_shares':True,'effective_epochs_recorded':True},
              'token_id_release_built':False,'loader_smoke_completed':False,'training_has_run':False,
              'scope':'frozen maximum 3.5B seen-token data recipe and sampled parent selections; step-5 export must match'}
    result['build_id'] = digest(result)
    write(out/'mixture_manifest.json',result)
    con.close()
    tools = out/'tools'
    tools.mkdir(exist_ok=True)
    for name in [Path(__file__).name,'census_curated_prelude_midtrain.py','prelude_midtrain.py','prelude_lf_counting.py']:
        shutil.copy2(Path(__file__).with_name(name),tools/name)
    print(json.dumps({'status':result['status'],'build_id':result['build_id'],
                      'seen_tokens':sum(s['seen_tokens'] for s in stages)}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--census',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--threads',type=int,default=64)
    p.add_argument('--allow-short-chat',action='store_true')
    build(p.parse_args())


if __name__=='__main__':
    main()
