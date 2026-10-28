#!/usr/bin/env python3
"""Validate the completed census/selection jobs and freeze step-1 decisions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while b:=f.read(16*1024**2):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    p=Path(path);temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');os.replace(temp,p)


def replay_lengths(replay, build_id):
    sys.path.insert(0, '/p/project1/laionize/marianna/dc_agent/.prelude_census_deps')
    import duckdb
    connection=duckdb.connect()
    connection.execute('SET threads=4')
    connection.execute("SET memory_limit='2GB'")
    paths=[str(p) for p in sorted(replay.glob('[0-9][0-9][0-9][0-9].parquet'))]
    connection.read_parquet(paths).create_view('metrics')
    summary=connection.execute('''SELECT family,language,count(*) valid_rows,sum(tokens) tokens,
        sum(tokens-1) trainable_tokens,count(*) FILTER(WHERE eligible) eligible_rows,
        sum(tokens) FILTER(WHERE eligible) eligible_tokens,
        sum(tokens-1) FILTER(WHERE eligible) eligible_trainable_tokens,
        quantile_cont(tokens,[0.5,0.9,0.95,0.99]) length_percentiles,
        quantile_cont(tokens,[0.5,0.9,0.95,0.99]) FILTER(WHERE eligible) eligible_length_percentiles
        FROM metrics GROUP BY family,language ORDER BY family,language''').fetchdf().to_dict('records')
    for row in summary:
        for field in ('length_percentiles','eligible_length_percentiles'):
            if hasattr(row[field],'tolist'):row[field]=row[field].tolist()
    bins=[]
    for low,high,label in [(0,4096,'<4K'),(4096,8192,'4-8K'),(8192,16384,'8-16K'),
                           (16384,32768,'16-32K'),(32768,65537,'32-64K'),(65537,None,'>64K')]:
        for filtered in [False,True]:
            condition=f'tokens>={low}'+(f' AND tokens<{high}' if high else '')+(' AND eligible' if filtered else '')
            rows=connection.execute(f'''SELECT family,language,count(*) n_rows,sum(tokens) tokens,
                sum(tokens-1) trainable_tokens FROM metrics WHERE {condition}
                GROUP BY family,language ORDER BY family,language''').fetchdf().to_dict('records')
            bins.extend(dict(row,length_bin=label,policy='eligible' if filtered else 'raw') for row in rows)
    result=dict(build_id=build_id,summary=summary,length_bins=bins)
    write_json(replay/'length_summary.json',result)
    connection.close()
    return result


def finish(root,license_policy):
    root=Path(root)
    core=root/'census_v2';replay=root/'replay_v1'
    choice=json.loads((root/'license_choice.json').read_text())
    if choice['primary_policy']!=license_policy:
        raise ValueError('Finalization policy differs from the recorded user choice')
    count_execution=dict(native_adapter=str(core/'tools/prelude_midtrain_census.py'))
    proof=None
    if (root/'lf_count_validation.json').exists():
        proof=json.loads((root/'lf_count_validation.json').read_text())
        for name,expected in proof['execution_implementations_sha256'].items():
            if sha256(root/'census_lf_tail/tools'/name)!=expected:
                raise ValueError('Guarded count implementation changed: '+name)
        count_execution['equivalent_lf_execution_validation']=str(root/'lf_count_validation.json')
    report=json.loads((core/'census_summary.json').read_text())
    verified=json.loads((core/'verified_input_manifest.json').read_text())
    if proof and (proof['semantic_change'] or proof['tokenizer_sha256']!=verified['tokenizer_files_sha256']['tokenizer.json']
                  or proof['adapter_implementation_sha256']!=verified['implementation_sha256']):
        raise ValueError('Count execution proof does not match the frozen adapter/tokenizer')
    replay_report=json.loads((replay/'summary.json').read_text())
    replay_verified=json.loads((replay/'verified_input_manifest.json').read_text())
    selected=json.loads((core/'repo_selection'/f'{license_policy}.json').read_text())
    comparison=json.loads((core/'repo_selection'/('all.json' if license_policy=='permissive' else 'permissive.json')).read_text())
    for obj in [report,verified,replay_report,replay_verified]:
        if obj['status']!='complete':raise ValueError('An upstream artifact is incomplete')
    if report['build_id']!=verified['build_id'] or selected['build_id']!=report['build_id']:
        raise ValueError('Census and selection build IDs disagree')
    if comparison['build_id']!=report['build_id'] or replay_report['build_id']!=replay_verified['build_id']:
        raise ValueError('Comparison/replay build IDs disagree')
    if len(verified['files'])!=report['verified_files']:
        raise ValueError('Not every frozen source file was checksum verified')
    for entry in verified['files']+replay_verified['entries']:
        if not entry.get('sha256'):raise ValueError('Missing file checksum')
        st=Path(entry['path']).stat()
        if {'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}!=entry['state']:
            raise ValueError('Input changed after processing: '+entry['path'])
    if sha256(selected['selection_path'])!=selected['selection_sha256']:
        raise ValueError('Repository selection file changed')
    for name,expected in verified['tokenizer_files_sha256'].items():
        if sha256(Path(verified['tokenizer'])/name)!=expected:raise ValueError('Tokenizer changed: '+name)
    if sha256(replay_verified['tokenizer'])!=replay_verified['tokenizer_sha256']:
        raise ValueError('Replay tokenizer changed')
    if replay_verified['tokenizer_sha256']!=verified['tokenizer_files_sha256']['tokenizer.json']:
        raise ValueError('Replay and core tokenizers differ')
    for field, count_field in [('rows','valid_rows'),('tokens','tokens'),
                               ('trainable_tokens','trainable_tokens'),('permissive_tokens','permissive_tokens')]:
        if sum(int(row[field]) for row in report['summary'])!=report['counts'][count_field]:
            raise ValueError('Census metrics and completion counts disagree: '+field)
    if report['counts']['rows_in']!=report['counts']['valid_rows']+report['counts'].get('rejected_rows',0):
        raise ValueError('Census rows are unaccounted for')
    replay_length_report=replay_lengths(replay,replay_report['build_id'])
    for field,count_field in [('valid_rows','valid'),('tokens','tokens'),('eligible_rows','eligible_rows'),
                              ('eligible_tokens','eligible_tokens')]:
        if sum(int(row[field]) for row in replay_length_report['summary'])!=sum(
                row['counts'][count_field] for row in replay_report['results']):
            raise ValueError('Replay metrics and completion counts disagree: '+field)
    included_specialized=['Nemotron-Pretraining-InfiniByte-Reasoning','Nemotron-Pretraining-Math-Textbooks',
                          'Nemotron-Pretraining-RQA','Nemotron-Pretraining-STEM-SFT','Nemotron-Pretraining-Scientific-Coding']
    pool=[]
    for row in report['summary']:
        if row['source']=='Nemotron-Pretraining-Specialized-v1' and row['subset'] not in included_specialized:continue
        if row['source']=='stack-v3-train':continue
        tokens=row['tokens']
        available_rows=int(row['rows'])
        if row['source']=='UltraData-Stack-v3-L2':
            tokens=row['permissive_tokens'] if license_policy=='permissive' else row['all_nonvendor_tokens']
            available_rows=sum(int(value['eligible_rows']) for value in selected['code_policy_summary']
                               if value['source']==row['source'] and value['subset']==row['subset'])
        pool.append(dict(source=row['source'],subset=row['subset'],rows_before_license_vendor_policy=int(row['rows']),
                         available_rows_before_dedup_decontam=available_rows,
                         available_tokens_before_dedup_decontam=int(tokens)))
    policies={license_policy:selected,comparison['license_policy']:comparison}
    license_comparison=dict(
        selected_repository_stats={name:value['stats'] for name,value in policies.items()},
        selected_token_delta_all_minus_permissive=policies['all']['stats']['tokens']-policies['permissive']['stats']['tokens'],
        code_pool_tokens_before_density_selection={name:{source:sum(int(row['eligible_tokens'])
            for row in value['code_policy_summary'] if row['source']==source)
            for source in ['stack-v3-train','UltraData-Stack-v3-L2']} for name,value in policies.items()},
        note='File-density gates are recomputed within each eligible license pool; selected repository sets need not be nested')
    selection=dict(schema_version=1,status='step1-complete',license_policy=license_policy,
        code_policy='Only permissive files; vendor files excluded' if license_policy=='permissive'
                    else 'Permissive and no_license files; vendor files excluded',
        core_build_id=report['build_id'],replay_build_id=replay_report['build_id'],license_comparison=license_comparison,
        count_execution=count_execution,
        sources=dict(cascade='All 9 frozen files: math, science, SWE, conversational agent, chat part 1, IF',
                     specialized=included_specialized,ultra='All 11 frozen languages, subject to the code policy',
                     stack=dict(selection_path=selected['selection_path'],selection_sha256=selected['selection_sha256'],
                                min_tokens=selected['min_tokens'],min_l2_file_density=selected['min_l2_file_density'],
                                min_l2_matched_files=selected['min_l2_matched_files'],stats=selected['stats'],
                                disjoint_categories=selected['by_category']),
                     replay=dict(verified_manifest=str(replay/'verified_input_manifest.json'),
                                 family_weights=replay_report['family_weights'],
                                 eligible_tokens=replay_report['eligible_tokens'],
                                 max_tokens_without_family_repetition=replay_report['max_replay_tokens_without_family_repetition'],
                                 language_weights=replay_report['multilingual_language_weights'],
                                 length_summary=str(replay/'length_summary.json'),
                                 differences_from_native=replay_verified['differences_from_native']),
                     long_pdfs=[r for r in replay_report['results'] if r['family']=='long_documents']),
        excluded=dict(Specialized_Wiki_Rewrite='Downloaded and measured; outside the chosen reasoning/code pool',
                      Cascade_terminal_agent='Duplicate source of Terminal-Corpus',
                      AgentTrove_RTS_TMax='Reserved for downstream agentic SFT',
                      StarCoder_aggregate='Local aggregate has insufficient file-level provenance for this replay cut'),
        available_pools=pool,filter_scope='Nonempty/schema-valid records and license/vendor policy only',
        trainable_token_count_scope='Reference census rendering/mask policy; recompute after production normalization and cleaning',
        deduplication_completed=False,benchmark_decontamination_completed=False,
        final_mixture_weights_frozen=False,training_has_run=False,
        next_stage='Normalize/render sources, preserve tool/environment masks, then deduplicate and decontaminate')
    refs=[core/'verified_input_manifest.json',core/'census_summary.json',core/'repo_selection/permissive.json',
          core/'repo_selection/all.json',replay/'verified_input_manifest.json',replay/'summary.json',
          replay/'length_summary.json',root/'license_choice.json']
    if (root/'lf_count_validation.json').exists():refs.append(root/'lf_count_validation.json')
    selection['artifact_sha256']={str(p.relative_to(root)):sha256(p) for p in refs}
    write_json(root/'source_selection.json',selection)
    print(json.dumps(dict(status=selection['status'],license_policy=license_policy,
                         verified_source_files=len(verified['files'])+len(replay_verified['entries']),
                         selected_repositories=selected['stats'],out=str(root/'source_selection.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--license-policy',choices=['permissive','all'],default='permissive')
    args=parser.parse_args();finish(args.root,args.license_policy)
