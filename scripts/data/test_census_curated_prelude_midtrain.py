"""Check exact masks, stage accounting, length cuts and frozen task resumption."""
from collections import Counter
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import census_curated_prelude_midtrain as census
import finalize_prelude_midtrain_census as finalizer
from prelude_midtrain import CASCADE, TERMINAL, dump, normalize_messages, render

STEP2 = Path('/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step2')


def row(source='FinePDFs-English-long/long_documents', kind='document'):
    return dict(id='fixture', kind=kind, mix_source=source, split='train', split_group='fixture',
                provenance_json=dump({'language':'eng_Latn'}), text='test\ntext',
                messages_json=None, tools_json=None)


class CensusTests(unittest.TestCase):
    def test_chunk_counts_match_explicit_windows_and_mask_conservation(self):
        rng = random.Random(183)
        for context, minimum in census.STAGES:
            for length in [1,2,context-1,context,context+1,2*context,10*context+17]:
                loss = [0] + [int(rng.random() > .3) for _ in range(length-1)]
                original = list(range(length))
                windows, masks = [original[:context]], [loss[:context]]
                cursor = min(context,length)
                while cursor < length:
                    stop = min(length, cursor+context-census.OVERLAP-1)
                    windows.append([-1]+original[cursor-census.OVERLAP:cursor]+original[cursor:stop])
                    masks.append([0]*(census.OVERLAP+1)+loss[cursor:stop])
                    cursor = stop
                self.assertTrue(all(len(w)<=context for w in windows))
                self.assertEqual(sum(map(sum,masks)),sum(loss))
                self.assertEqual(census.stage_counts(length,sum(loss),context),
                                 (len(windows),sum(map(len,windows)),sum(loss)))

    def test_replay_and_pdf_eligibility_boundaries_and_stage_exceptions(self):
        r = row('dclm-baseline-1.0/dclm_english')
        self.assertTrue(census.metric(r,4096,4095,1)['eligible'])
        self.assertFalse(census.metric(r,4097,4096,1)['eligible'])
        self.assertEqual(census.metric(r,4096,4095,1)['stage_65536_seen_tokens'],4096)
        self.assertFalse(census.metric(row(),32767,32766,1)['eligible'])
        self.assertTrue(census.metric(row(),32768,32767,1)['eligible'])
        s = row('stack-v3-train/data/long_documents')
        self.assertEqual(census.metric(s,8191,8190,1)['stage_16384_seen_tokens'],0)
        self.assertEqual(census.metric(s,8192,8191,1)['stage_16384_seen_tokens'],8192)

    def test_category_format_registry_disjointness_and_unknowns(self):
        self.assertEqual(census.classify('stack-v3-train/data/code','document')[0],'code_agent')
        self.assertEqual(census.classify('stack-v3-train/data/long_documents','document')[0],'long_documents')
        self.assertEqual(census.classify(TERMINAL+'/dataset_adapters','conversation')[1],TERMINAL)
        eu = census.classify('FinePDFs-EU-replay/pdf_multilingual','document',{'language':'deu_Latn'})
        self.assertTrue(eu[1].endswith('/deu_Latn'))
        for source,kind in [('unknown/source','document'),(CASCADE+'/math','document'),
                            ('UltraData-Stack-v3-L2/data/python','conversation')]:
            with self.assertRaises((KeyError,ValueError)):
                census.classify(source,kind)

    def test_parallel_task_ownership_is_disjoint_complete_and_deterministic(self):
        tasks = [{'key':str(n),'state':{'bytes':(n*17)%37}} for n in range(191)]
        parts = census.partition(tasks,8)
        self.assertEqual(census.partition(list(reversed(tasks)),8),parts)
        self.assertEqual(sorted(t['key'] for p in parts for t in p),sorted(t['key'] for t in tasks))


@unittest.skipUnless((STEP2/'tokenizer').exists(),'Frozen tokenizer required')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(STEP2/'tokenizer',local_files_only=True)
        cls.counter = census.LFCountingTokenizer(cls.tok.backend_tokenizer, STEP2/'tokenizer/tokenizer.json')

    def test_documents_match_native_renderer_including_unicode_and_special_ids(self):
        for text in ['', 'hello\nworld', 'αβ e\u0301 🙂\n  indented\n<eos>\n<|im_end|>',
                     ('def f():\n    return "Case"\n'*10000)]:
            r = row('stack-v3-train/data/code')
            r.update(text=text,messages=None,tools=None)
            expected = render(r,self.tok,max_tokens=1000000)
            n = len(self.counter.encode(text).ids)+2
            self.assertEqual(n,len(expected['ids']))
            self.assertEqual(n-1,sum(expected['loss']))

    def test_conversation_census_matches_renderer_with_reasoning_calls_and_tool_groups(self):
        raw = {'messages':[
            {'role':'system','content':'Keep reasoning.'}, {'role':'user','content':'Solve α.'},
            {'role':'assistant','reasoning_content':'Earlier reasoning.', 'content':'',
             'tool_calls':[{'function':{'name':'python','arguments':{'code':'print(2)'}}}]},
            {'role':'tool','content':'2'}, {'role':'tool','content':'tool continuation'},
            {'role':'user','content':'Why?'},
            {'role':'assistant','reasoning_content':'Later reasoning.', 'content':'The answer is 2.'}]}
        messages,tools = normalize_messages(raw,CASCADE,Counter())
        ids,loss = census.render_conversation(self.tok,messages,tools)
        r = row(CASCADE+'/math','conversation')
        r.update(messages=messages,tools=tools)
        expected = render(r,self.tok,max_tokens=100000)
        self.assertEqual((ids,loss),(expected['ids'],expected['loss']))
        m = census.metric(r,len(ids),sum(loss),0)
        self.assertGreater(m['masked_tokens'],1)
        self.assertEqual(m['trainable_tokens'],sum(expected['loss']))

    def test_worker_coverage_checksum_and_resume_guards(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        r = row('stack-v3-train/data/code')
        r['curation_build_id'] = 'decision-fixture'
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base/'source.parquet'
            pq.write_table(pa.Table.from_pylist([r,r|{'id':'second','text':'different'}]),source)
            task = {'key':'fixture','path':str(source),'sha256':census.sha(source),'state':census.state(source)}
            m = {'build_id':'fixture-build','decision_build_id':'decision-fixture','step2':str(STEP2)}
            census.init_worker(base,m)
            result = census.process((task,0))
            self.assertEqual(result['counts']['rows'],2)
            self.assertTrue(census.process((task,0))['resumed'])
            (base/'curation_manifest.json').write_text('{}')
            m.update(step3=str(base), curation_manifest_sha256=census.sha(base/'curation_manifest.json'),
                     tokenizer_contract_sha256=census.sha(STEP2/'tokenizer_contract.json'),
                     tokenizer_files_sha256={}, policy=census.POLICY, tasks=[task], dependencies={},
                     code_sha256={n:census.sha(Path(census.__file__).with_name(n)) for n in census.CODE},
                     expected_rows=2, expected_counts=[dict(mix_source=r['mix_source'], kind='document',split='train',rows=2)])
            m.pop('build_id')
            m['build_id'] = census.digest(m)
            census.write(base/'census_input_manifest.json',m)
            result['build_id'] = m['build_id']
            census.write(base/'census/fixture/stats.json',result)
            census.summarize(SimpleNamespace(out=str(base),threads=2))
            report = json.loads((base/'retained_census.json').read_text())
            self.assertEqual(report['status'],'retained-census-complete')
            self.assertEqual(report['counts']['rows'],2)
            # Real empty canonical work units still need verified metrics parts.
            empty = base/'census/empty'
            empty.mkdir()
            pq.write_table(pa.Table.from_pylist([],schema=census.METRICS),empty/'metrics.parquet')
            empty_task = dict(task,key='empty')
            census.write(empty/'stats.json',dict(build_id=m['build_id'],limit=0,
                input_sha256=task['sha256'],metrics_sha256=census.sha(empty/'metrics.parquet'),counts={}))
            m['tasks'].append(empty_task)
            m.pop('build_id')
            m['build_id'] = census.digest(m)
            for directory in [base/'census/fixture',empty]:
                stats = json.loads((directory/'stats.json').read_text())
                stats['build_id'] = m['build_id']
                census.write(directory/'stats.json',stats)
            census.write(base/'census_input_manifest.json',m)
            census.summarize(SimpleNamespace(out=str(base),threads=2))
            finalizer.finalize(SimpleNamespace(out=str(base),threads=2))
            final = json.loads((base/'retained_census.json').read_text())
            self.assertEqual(final['counts']['rows'],2)
            self.assertEqual(len(final['metrics_sha256']),2)
            evidence = dict(census_build_id=m['build_id'],helper_sha256='fixture-helper')
            for name in ['parallel_completion.json','parallel_completion_plan.json']:
                census.write(base/name,evidence)
            census.write(base/'bounded_parity.json',evidence|dict(status='passed',exact_counts_and_parquet_bytes=True))
            finalizer.finalize(SimpleNamespace(out=str(base),threads=2))
            checked = json.loads((base/'retained_census.json').read_text())
            self.assertEqual(checked['completion_evidence']['helper_sha256'],'fixture-helper')
            census.write(base/'bounded_parity.json',evidence|dict(status='failed',exact_counts_and_parquet_bytes=False))
            with self.assertRaisesRegex(ValueError,'matching exact parity evidence'):
                finalizer.finalize(SimpleNamespace(out=str(base),threads=2))
            tampered = dict(m,expected_rows=3)
            census.write(base/'census_input_manifest.json',tampered)
            with self.assertRaisesRegex(ValueError,'identity/policy changed'):
                census.validate(base)
            census.write(base/'census_input_manifest.json',m)
            census.init_worker(base,m)
            metrics = base/'census/fixture/metrics.parquet'
            metrics.write_bytes(metrics.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'metrics changed'):
                census.process((task,0))


if __name__ == '__main__':
    unittest.main()
