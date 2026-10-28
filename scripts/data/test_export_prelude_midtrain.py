"""Actual IDs, masks, continuation boundaries, and resumable export integrity."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
import export_prelude_midtrain as e
from prelude_midtrain import render_conversation

STEP2 = Path('/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step2')


class ChunkTests(unittest.TestCase):
    def test_boundaries_overlap_tail_eos_and_target_conservation(self):
        for context in [16384, 32768, 65536]:
            for length in [2, context - 1, context, context + 1,
                           2 * context - 257, 2 * context - 256, 3 * context + 19]:
                ids = np.arange(length, dtype='<i4') % 1000 + 100
                ids[0], ids[-1] = 1, 2
                loss = (np.arange(length) % 7 != 0).astype(np.int8)
                loss[0] = 0
                windows = list(e.token_chunks(ids, loss, context))
                self.assertEqual((len(windows), sum(len(w[-2]) for w in windows),
                                  sum(int(w[-1].sum()) for w in windows)),
                                 e.stage_counts(length, int(loss.sum()), context))
                rebuilt_ids, rebuilt_loss = [], []
                previous = 0
                for i, start, end, a, b, wi, wl in windows:
                    self.assertEqual(start, previous)
                    self.assertLessEqual(len(wi), context)
                    if i:
                        np.testing.assert_array_equal(wi[:257], np.r_[1, ids[a:b]])
                        np.testing.assert_array_equal(wl[:257], np.zeros(257))
                    prefix = 257 if i else 0
                    np.testing.assert_array_equal(wi[prefix:], ids[start:end])
                    np.testing.assert_array_equal(wl[prefix:], loss[start:end])
                    rebuilt_ids.extend(wi[prefix:])
                    rebuilt_loss.extend(wl[prefix:])
                    previous = end
                np.testing.assert_array_equal(rebuilt_ids, ids)
                np.testing.assert_array_equal(rebuilt_loss, loss)
                self.assertEqual(windows[-1][-2][-1], 2)

    def test_fully_masked_continuation_is_preserved(self):
        ids = np.r_[1, np.full(20000, 123), 2].astype('<i4')
        loss = np.zeros(len(ids), dtype=np.int8)
        loss[2] = 1
        windows = list(e.token_chunks(ids, loss, 16384))
        self.assertEqual(len(windows), 2)
        self.assertEqual(int(windows[1][-1].sum()), 0)
        self.assertEqual(sum(len(w[-2]) for w in windows), len(ids) + 257)

    def test_invalid_inputs_fail(self):
        for ids, loss, context in [([2, 2], [0, 1], 16384), ([1, 2], [1, 1], 16384),
                                   ([1, 2], [0], 16384), ([1, 2], [0, 1], 257)]:
            with self.assertRaises(ValueError):
                list(e.token_chunks(np.array(ids), np.array(loss), context))


@unittest.skipUnless((STEP2 / 'tokenizer').exists(), 'Frozen tokenizer required')
class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(STEP2 / 'tokenizer', local_files_only=True)
        cls.tok.model_max_length = 10**12

    def test_document_arrays_match_whole_native_tokenization(self):
        for text in ['', 'def ExactCase():\n\treturn "α e\u0301 🙂"\n',
                     '<|im_start|>assistant\n<eos>\n' * 5000, 'spaces  \t \n' * 10000]:
            ids, loss = e.document_ids(self.tok, text)
            expected = [1] + self.tok.backend_tokenizer.encode(text, add_special_tokens=False).ids + [2]
            np.testing.assert_array_equal(ids, expected)
            np.testing.assert_array_equal(loss, [0] + [1] * (len(ids) - 1))

    def test_export_real_mixed_rows_and_reject_corrupt_resume(self):
        messages = [{'role': 'user', 'content': 'Explain.'},
                    {'role': 'assistant', 'reasoning_content': 'Earlier reasoning.', 'content': 'Call.'},
                    {'role': 'tool', 'content': 'tool payload\n' * 25000},
                    {'role': 'assistant', 'content': 'Done.'}]
        rows = [dict(id='doc', kind='document', text='ExactCase α\n' * 6000,
                     messages_json=None, tools_json=None, provenance_json='{"source":"fixture"}',
                     mix_source='code', split='train', split_group='doc:fixture', curation_build_id='decisions'),
                dict(id='conv', kind='conversation', text=None, messages_json=json.dumps(messages),
                     tools_json=None, provenance_json='{}', mix_source='chat', split='train',
                     split_group='task:fixture', curation_build_id='decisions')]
        rendered = [e.document_ids(self.tok, rows[0]['text']),
                    tuple(np.asarray(v, dtype='<i4' if i == 0 else np.int8)
                          for i, v in enumerate(render_conversation(self.tok, messages)))]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.parquet'
            pq.write_table(pa.Table.from_pylist(rows), source, row_group_size=1)
            selected = []
            for row, (ids, loss) in zip(rows, rendered):
                n, seen, trained = e.stage_counts(len(ids), int(loss.sum()), 16384)
                selected.append({k: row[k] for k in ['id', 'kind', 'mix_source', 'split', 'split_group']} |
                                dict(pool=row['mix_source'], category='fixture', context=16384,
                                     original_tokens=len(ids), seen_tokens=seen, trainable_tokens=trained, chunks=n))
            path = root / 'selected.parquet'
            pq.write_table(pa.Table.from_pylist(selected), path)
            task = dict(key='fixture', source=dict(key='fixture', path=str(source), state=e.state(source)),
                        selection=path.name, groups=[0, 1], parents=len(rows))
            m = dict(build_id='fixture', identity=dict(base=str(STEP2.parent), decision_build_id='decisions'),
                     selection_sha256={path.name: e.sha(path)})
            e.init_worker(root, m)
            first = e.process(task)
            self.assertEqual(e.process(task)['resumed'], True)
            from verify_prelude_midtrain_export import audit_task
            self.assertEqual(audit_task((root, m, task))['status'], 'passed')
            exported = pq.read_table(root / 'release/train/16384/fixture.parquet').to_pylist()
            for source_row, (ids, loss) in zip(rows, rendered):
                parts = [r for r in exported if r['parent_id'] == source_row['id']]
                actual_ids, actual_loss = [], []
                for part in parts:
                    prefix = 257 if part['chunk_index'] else 0
                    actual_ids.extend(part['ids'][prefix:])
                    actual_loss.extend(part['loss'][prefix:])
                np.testing.assert_array_equal(actual_ids, ids)
                np.testing.assert_array_equal(actual_loss, loss)
                self.assertEqual(parts[0]['parent_ids_sha256'], hashlib.sha256(ids.tobytes()).hexdigest())
            self.assertGreater(sum(v['zero_target_chunks'] for v in first['counts'].values()), 0)
            export_path = root / 'release/train/16384/fixture.parquet'
            # Re-sign a deliberately malformed prefix to prove the independent
            # audit checks semantics, not only file checksums.
            bad = [dict(r) for r in exported]
            continuation = next(r for r in bad if r['chunk_index'] == 1)
            continuation['loss'] = list(continuation['loss'])
            continuation['loss'][1] = 1
            pq.write_table(pa.Table.from_pylist(bad, schema=e.SCHEMA), export_path)
            stats = dict(first)
            stats['artifacts_sha256'] = {'train/16384/fixture.parquet': e.sha(export_path)}
            e.write(root / 'work/fixture/stats.json', stats)
            with self.assertRaisesRegex(ValueError, 'continuation prefix'):
                audit_task((root, m, task))
            e.write(root / 'work/fixture/stats.json', first)
            with export_path.open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'completed export changed'):
                e.process(task)

    def test_build_key_covers_policy_and_dependency_changes(self):
        identity = {'policy': e.POLICY, 'code_sha256': {'renderer': 'old'}, 'upstream_sha256': {'tokenizer': 'old'}}
        baseline = e.digest(identity)
        for key in ['policy', 'code_sha256', 'upstream_sha256']:
            changed = dict(identity, **{key: {'changed': True}})
            self.assertNotEqual(e.digest(changed), baseline)


if __name__ == '__main__':
    unittest.main(verbosity=2)
