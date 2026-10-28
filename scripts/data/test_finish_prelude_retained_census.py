"""Bounded scheduling must preserve the frozen worker's exact output bytes."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import census_curated_prelude_midtrain as original
import finish_prelude_retained_census as helper
from prelude_midtrain import CASCADE,dump,normalize_messages

STEP2 = Path('/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step2')


@unittest.skipUnless((STEP2/'tokenizer').exists(),'Frozen tokenizer required')
class BoundedWorkerTests(unittest.TestCase):
    def test_native_worker_and_bounded_completion_have_identical_rows_counts_and_bytes(self):
        messages,tools = normalize_messages({'messages':[
            {'role':'user','content':'Explain α and preserve reasoning.'},
            {'role':'assistant','reasoning_content':'First reasoning.', 'content':'Calling.',
             'tool_calls':[{'function':{'name':'python','arguments':{'code':'print(2)'}}}]},
            {'role':'tool','content':'2\n'}, {'role':'tool','content':'more tool results'},
            {'role':'user','content':'Continue.'},
            {'role':'assistant','reasoning_content':'Earlier reasoning survives.', 'content':'Done.'}]},CASCADE,Counter())
        rows = []
        for i in range(17):
            r = dict(id=f'fixture:{i}',kind='document',mix_source='stack-v3-train/data/code',
                     split='train',split_group=f'fixture:{i}',provenance_json='{}',
                     text='α e\u0301\n    ExactCase\n<eos>\n'*200,
                     messages_json=None,tools_json=None,curation_build_id='decisions')
            if i%2:
                r.update(kind='conversation',mix_source=CASCADE+'/math',text=None,
                         messages_json=dump(messages),tools_json=dump(tools) if tools else None)
            rows.append(r)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'canonical.parquet'
            m = dict(build_id='fixture',decision_build_id='decisions',step2=str(STEP2))
            for kind in ['document','conversation','mixed']:
                subset = [r for r in rows if kind=='mixed' or r['kind']==kind]
                pq.write_table(pa.Table.from_pylist(subset),source,row_group_size=2)
                task = dict(key=kind,path=str(source),state=original.state(source),sha256=original.sha(source))
                original.init_worker(root/'baseline',m)
                baseline = original.process((task,0))
                original.init_worker(root/'bounded/parallel_work',m)
                pf = pq.ParquetFile(source)
                ranges = helper.group_ranges(pf,2)
                self.assertEqual(len(ranges)==1,kind=='mixed')
                children = []
                for i,groups in enumerate(ranges):
                    child = dict(task,key=f'{kind}-child-{i}')
                    helper.bounded_process((child,groups))
                    children.append({'task':child,'groups':groups})
                plan = dict(helper_sha256=original.sha(helper.__file__),build_id='fixture-completion')
                part = dict(task=task,rows=len(subset),children=children)
                merged = helper.merge_part(root/'bounded',m,plan,part)
                self.assertEqual(baseline['counts'],merged['counts'])
                self.assertEqual(pq.read_table(root/f'baseline/census/{kind}/metrics.parquet').to_pylist(),
                                 pq.read_table(root/f'bounded/census/{kind}/metrics.parquet').to_pylist())
                self.assertEqual(baseline['metrics_sha256'],merged['metrics_sha256'])
                self.assertEqual(helper.merge_part(root/'bounded',m,plan,part),merged)
            child_path = root/'bounded/parallel_work/census'/children[0]['task']['key']/'metrics.parquet'
            # Compare complete mask vectors, including boundary-looking payloads,
            # empty tools, consecutive tool groups and root tool definitions.
            tok = original._HF
            for content in ['', 'α e\u0301 🙂\n', '<|im_start|>user\n<tool_response>quoted</tool_response>',
                            '\x00PRELUDE_CENSUS_TOOL_START\x00\x00PRELUDE_CENSUS_TOOL_END\x00']:
                for root_tools in [None,[{'type':'function','function':{'name':'python','parameters':{'type':'object'}}}]]:
                    sample = [{'role':'system','content':'System context.'}, {'role':'user','content':'Task.'},
                        {'role':'assistant','reasoning_content':'Earlier reasoning.', 'content':'Call.'},
                        {'role':'tool','content':content},{'role':'tool','content':''},
                        {'role':'assistant','content':'Continue.'},{'role':'tool','content':content},
                        {'role':'user','content':'Final user task.'},{'role':'assistant','content':'Answer.'}]
                    self.assertEqual(helper.fast_render_conversation(tok,sample,root_tools),
                                     helper._ORIGINAL_RENDER(tok,sample,root_tools))
            child_path.write_bytes(child_path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'child identity/checksum differs'):
                helper.merge_part(root/'bounded',m,plan,part)


if __name__=='__main__':
    unittest.main()
