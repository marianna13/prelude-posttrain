"""Step 3 identities, exact native 13-grams, global curation and resume fixtures."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

import numpy as np

from prelude_decontam import Scanner,gram_hash
from curate_prelude_midtrain import task_prompt,task_alias,scan_fields,digest,schedule
from prelude_midtrain import TERMINAL

LIBRARY='/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step3/native/ngram_scan.so'


def python_hits(text,index,n=13):
    words=text.lower().split();out={}
    for i in range(len(words)-n+1):
        gram=words[i:i+n];value=' '.join(gram);h=gram_hash(value)
        if h in index:out[h]={'eligible':len(set(gram))>=7,'ngram':value}
    return out,max(0,len(words)-n+1)


class NativeScanTest(unittest.TestCase):
    def test_native_matches_python_including_uncapped_hits(self):
        rng=random.Random(17)
        vocabulary=['α','İ','Straße','é','Z','a','  ','\u2003','\n','\t','long_word','中','λ']
        texts=[' '.join(str(i) for i in range(1700)), ' '.join(['a','b','c','d','e','f']*10),'','A  B\nC']
        texts += [' '.join(rng.choice(vocabulary) for _ in range(150)) for _ in range(60)]
        hashes=set()
        for text in texts:
            w=text.lower().split()
            hashes.update(gram_hash(' '.join(w[i:i+13])) for i in range(len(w)-12))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);np.save(p/'ngrams.npy',np.array(sorted(hashes),dtype=np.uint64));(p/'manifest.json').write_text('{"n":13}')
            scanner=Scanner(p,LIBRARY)
            for text in texts:
                self.assertEqual(scanner.scan(text),python_hits(text,hashes))
            self.assertGreater(len(scanner.scan(texts[0])[0]),1024)
            self.assertTrue(scanner.scan(texts[1])[0])
            self.assertFalse(any(v['eligible'] for v in scanner.scan(texts[1])[0].values()))


class IdentityTest(unittest.TestCase):
    def test_prompt_identity_excludes_tools_and_preserves_code_bytes(self):
        first=[{'role':'system','content':'system differs'}, {'role':'user','content':'  X = "a"\n'},
               {'role':'assistant','content':'reasoning differs'}, {'role':'tool','content':'environment differs'},
               {'role':'user','content':'<tool_response>output</tool_response>'}]
        self.assertEqual(task_prompt(first),digest('  X = "a"\n'))
        for text in ('  x = "a"\n',' X = "a"\n','  X = "A"\n'):
            self.assertNotEqual(task_prompt(first),digest(text))
        entry={'source':TERMINAL}
        a=task_alias({'task':'task_1','source':'dataset_A'},entry)
        b=task_alias({'task':'task_1','source':'dataset_B'},entry)
        self.assertNotEqual(a,b)

    def test_scan_includes_masked_content_reasoning_and_calls(self):
        row={'kind':'conversation','messages':[{'role':'tool','content':'masked environment'},
             {'role':'assistant','content':'answer','reasoning_content':'private reasoning','tool_calls':[{'name':'x'}]}],
             'tools':[{'name':'tool definition'}]}
        values=list(scan_fields(row))
        self.assertIn(('0:tool:content','masked environment'),values)
        self.assertIn(('1:assistant:reasoning_content','private reasoning'),values)
        self.assertTrue(any(k.endswith('tool_calls') for k,v in values))
        self.assertTrue(any(k=='tools' for k,v in values))

    def test_cluster_shards_are_disjoint_and_complete(self):
        tasks=[{'task_key':str(i),'weight':i*17} for i in range(20)]
        shards=[schedule(tasks,i,4) for i in range(4)]
        self.assertEqual(sorted(t['task_key'] for s in shards for t in s),sorted(t['task_key'] for t in tasks))
        self.assertEqual(shards,[schedule(tasks,i,4) for i in range(4)])

    def test_task_components_and_split_hash(self):
        from reduce_prelude_midtrain import union_components,split_for
        c=union_components([('aaa','x'),('bbb','x'),('bbb','y'),('ccc','y')])
        self.assertEqual(c,{'aaa':'aaa','bbb':'aaa','ccc':'aaa'})
        self.assertEqual(split_for('task:aaa'),split_for('task:'+c['ccc']))
        self.assertNotEqual(digest('A\n  b'),digest('a\n b'))

    def test_changed_policy_cannot_reuse_a_frozen_build_identity(self):
        from finalize_prelude_midtrain_step3 import finalize
        real=Path('/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step3/input_manifest.json')
        manifest=json.loads(real.read_text())
        manifest['policy']['validation_seed']='changed-after-scan'
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'input_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError,'input manifest identity changed'):
                finalize(directory)


class GlobalReductionTest(unittest.TestCase):
    def test_actual_duckdb_reservation_code_coverage_aliases_and_boilerplate(self):
        from curate_prelude_midtrain import META,FILES,SAMPLES,REF,STACK,ULTRA,write,sha
        from reduce_prelude_midtrain import summarize,decide,split_for
        real=Path('/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step3/input_manifest.json')
        m=json.loads(real.read_text());m['build_id']='fixture'
        h=gram_hash('one two three four five six seven eight nine ten eleven twelve thirteen')
        text='import alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu'
        boiler=gram_hash(text)
        records=[]
        def row(ident,*,source='fixture',task=None,alias=None,content=None,cid=None,repo=None,hits=None,quality=0):
            r={k:None for k in META.names}
            r.update(id=ident,namespace='core',file_id=1,locator=len(records),source=source,subset='fixture',
                     kind='conversation' if task else 'document',mix_source=source,record_sha256=digest(ident),
                     task_hash=digest(task) if task else None,task_alias=alias,content_sha256=digest(content or ident),
                     repo_key=repo,content_id=cid,quality=float(quality),algo_rel=0.,raw_hits=hits or [],eligible_hits=hits or [])
            records.append(r)
        row('reserved1',task='SFT prompt',alias='same-upstream-task')
        row('reserved2',task='different rendering of SFT task',alias='same-upstream-task')
        row('conv1',task='new task');row('conv2',task='new task')
        row('stack',source=STACK,content='repo wrapper',repo='owner/repo')
        row('covered',source=ULTRA,content='print("a")',cid='same-id',repo='owner/repo')
        row('case-low',source=ULTRA,content='print("A")',cid='same-id',repo='owner/other',quality=1)
        row('case-high',source=ULTRA,content='print("A")',cid='different-id',repo='owner/other',quality=9)
        row('indent',source=ULTRA,content=' print("a")',cid='indent',repo='owner/repo')
        row('string',source=ULTRA,content='print("a ")',cid='string',repo='owner/repo')
        row('whitelisted',content='generic code',hits=[boiler])
        for i in range(25):row('benchmark'+str(i),task='contaminated task '+str(i),hits=[h])
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);scan=out/'scan';scan.mkdir()
            part=scan/'core-1';part.mkdir();ref=scan/'reference-1';ref.mkdir()
            pq.write_table(pa.Table.from_pylist(records,schema=META),part/'records.parquet')
            pq.write_table(pa.Table.from_pylist([{'parent_id':'stack','repo_key':'owner/repo','content_id':'same-id',
                   'content_sha256':digest('print("a")'),'file_path':'x.py'}],schema=FILES),part/'files.parquet')
            pq.write_table(pa.Table.from_pylist([{'h':h,'ngram':'one two three four five six seven eight nine ten eleven twelve thirteen',
                    'eligible':True,'id':'benchmark0','source':'fixture','subset':'fixture','field':'tool'},
                   {'h':boiler,'ngram':text,'eligible':True,'id':'whitelisted','source':'fixture','subset':'fixture','field':'document'}],schema=SAMPLES),part/'samples.parquet')
            pq.write_table(pa.Table.from_pylist([{'task_hash':digest('SFT prompt'),'uuid':'ref','file':'SFT','row':0}],schema=REF),ref/'reference.parquet')
            for root,counts in [(part,{'scanned_rows':len(records),'accepted_rows':len(records)}),(ref,{'scanned_rows':1,'accepted_rows':1})]:
                write(root/'stats.json',{'build_id':'fixture','limit':0,'counts':counts,'artifact_sha256':{p.name:sha(p) for p in root.glob('*.parquet')}})
            m['tasks']=[{'task_key':'core-1'}];m['references']=[{'task_key':'reference-1'}]
            write(out/'input_manifest.json',m)
            args=SimpleNamespace(out=str(out),pilot=False,verify_bytes=True,memory='1GB',threads=2)
            summarize(args)
            exclusions=out/'exclusions.json';write(exclusions,{'scan_build_id':'fixture','status':'reviewed','scope':'full','frequency_cutoff':None,
                       'excluded_hashes':[{'h':boiler,'ngram':text,'reason':'fixture generic import boilerplate'}]})
            args.exclusions=str(exclusions);decide(args)
            decisions=[]
            for path in (out/'reduction/decisions').rglob('*.parquet'):decisions+=pq.ParquetFile(path).read().to_pylist()
            by={r['id']:r for r in decisions}
            self.assertEqual(by['reserved1']['reason'],'reserved_sft_task')
            self.assertEqual(by['reserved2']['reason'],'reserved_sft_task')
            self.assertEqual(by['conv1']['reason'],'keep');self.assertEqual(by['conv2']['duplicate_of'],'conv1')
            self.assertEqual(by['covered']['reason'],'covered_by_stack')
            self.assertEqual(by['case-high']['reason'],'keep');self.assertEqual(by['case-low']['reason'],'exact_duplicate')
            for ident in ['indent','string','stack','whitelisted']:self.assertEqual(by[ident]['reason'],'keep')
            self.assertTrue(all(by['benchmark'+str(i)]['reason']=='benchmark_match' for i in range(25)))
            self.assertEqual(by['stack']['split'],by['indent']['split'])
            for r in decisions:self.assertEqual(r['split'],split_for(r['split_group']))
            self.assertGreater(pq.read_table(out/'reduction/content_id_conflicts.parquet').num_rows,0)


if __name__=='__main__':unittest.main(verbosity=2)
