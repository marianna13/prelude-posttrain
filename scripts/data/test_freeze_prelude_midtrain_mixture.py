"""Constrained weights and real deterministic token-quota selection fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import duckdb
import pyarrow as pa
import freeze_prelude_midtrain_mixture as mixture
from census_curated_prelude_midtrain import METRICS, metric


class MixtureTests(unittest.TestCase):
    def setUp(self):
        self.registry = {
            'reason_doc':{'category':'reasoning','kind':'document'},
            'reason_conv':{'category':'reasoning','kind':'conversation'},
            'code_doc':{'category':'code_agent','kind':'document'},
            'code_conv':{'category':'code_agent','kind':'conversation'},
            'chat':{'category':'chat_if','kind':'conversation'},
            'long':{'category':'long_documents','kind':'document'},
            'replay':{'category':'replay','kind':'document'},
        }
        self.prior = dict(zip(self.registry,[.15,.15,.15,.10,.10,.125,.225]))
        self.available = [dict(pool=p,context=c,split='train',original_tokens=10_000_000_000)
                          for p in self.registry for c,_ in mixture.STAGES]

    def test_joint_weights_feasible_and_priors_preserved(self):
        weights,report = mixture.solve_weights(self.registry,self.available,self.prior)
        mixture.validate_weights(weights,self.registry)
        for p in self.prior:
            self.assertAlmostEqual(weights[p],self.prior[p])
        self.assertEqual(report['maximum_parent_repetitions'],1)

    def test_solver_adjusts_small_source_while_preserving_category_and_format(self):
        available = copy.deepcopy(self.available)
        for r in available:
            if r['pool']=='code_doc':
                r['original_tokens'] = 200_000_000
        weights,report = mixture.solve_weights(self.registry,available,self.prior)
        self.assertIsNotNone(weights)
        self.assertLess(weights['code_doc'],self.prior['code_doc'])
        self.assertGreater(report['total_absolute_adjustment'],0)
        mixture.validate_weights(weights,self.registry)

    def test_reallocation_preserves_measured_code_language_ratios(self):
        registry = dict(self.registry)
        registry.pop('code_doc')
        cpp,python = mixture.ULTRA+'/data/cpp',mixture.ULTRA+'/data/python'
        registry.update({cpp:{'category':'code_agent','kind':'document'},
                         python:{'category':'code_agent','kind':'document'},
                         'unavailable_code':{'category':'code_agent','kind':'document'}})
        prior = {p:w for p,w in self.prior.items() if p!='code_doc'}
        prior.update({cpp:.08,python:.04,'unavailable_code':.03})
        available = [dict(pool=p,context=c,split='train',original_tokens=10_000_000_000)
                     for p in registry for c,_ in mixture.STAGES if p!='unavailable_code']
        weights,report = mixture.solve_weights(registry,available,prior)
        self.assertEqual(weights['unavailable_code'],0)
        self.assertAlmostEqual(weights[cpp]/weights[python],2,places=8)
        self.assertTrue(report['ultra_language_ratios_frozen'])
        again,_ = mixture.solve_weights(dict(reversed(list(registry.items()))),available,prior)
        self.assertEqual(weights,again)

    def test_missing_long_chat_bucket_is_infeasible_without_silent_repetition(self):
        available = [r for r in self.available if not (r['pool']=='chat' and r['context']==65536)]
        weights,report = mixture.solve_weights(self.registry,available,self.prior)
        self.assertIsNone(weights)
        self.assertEqual(report['status'],'infeasible')

    def test_invalid_weight_registries_are_rejected(self):
        invalid = [dict(self.prior,unknown=0),dict(self.prior,chat=-.1),dict(self.prior,chat=float('nan')),
                   dict(self.prior,chat=0),{k:v for k,v in self.prior.items() if k!='chat'}]
        for weights in invalid:
            with self.assertRaises(ValueError):
                mixture.validate_weights(weights,self.registry)

    def test_changed_census_manifest_is_rejected(self):
        report = {'status':'retained-census-complete','counts':{'tokens':123}}
        report['build_id'] = mixture.digest(report)
        mixture.verify_census_report(report)
        report['counts']['tokens'] += 1
        with self.assertRaisesRegex(ValueError,'manifest identity changed'):
            mixture.verify_census_report(report)

    def test_whole_parent_quota_never_silently_slices(self):
        row = dict(id='x',mix_source='x',pool='long',category='long_documents',kind='document',split='train',
                   split_group='x',tokens=100000,trainable_tokens=99999)
        self.assertIsNone(mixture.choose_parent(row,16384,99999))
        result = mixture.choose_parent(row,16384,200000)
        self.assertGreater(result['seen_tokens'],row['tokens'])
        self.assertEqual(result['trainable_tokens'],99999)
        self.assertEqual(result['original_tokens'],100000)

    def test_real_duckdb_sampler_is_disjoint_deterministic_and_counts_masks(self):
        # Sufficient parents keep the production share tolerances meaningful.
        old_budgets = mixture.BUDGETS
        mixture.BUDGETS = {16384:200_000_000,32768:400_000_000,65536:800_000_000}
        try:
            records = []
            for p,attributes in self.registry.items():
                for i in range(80000 if p=='replay' else 7000):
                    r = dict(id=f'{p}:{i}',mix_source=p,pool=p,split='train',split_group=f'{p}:{i}',
                             category=attributes['category'],kind=attributes['kind'],language='',eligible=True,
                             eligibility_reason='',tokens=4000 if p=='replay' else 33000,
                             trainable_tokens=3999 if p=='replay' else (25000 if attributes['kind']=='conversation' else 32999),chars=1)
                    r['masked_tokens'] = r['tokens']-r['trainable_tokens']
                    for c,_ in mixture.STAGES:
                        chunks,seen,trained = mixture.stage_counts(r['tokens'],r['trainable_tokens'],c)
                        r.update({f'stage_{c}_chunks':chunks,f'stage_{c}_seen_tokens':seen,
                                  f'stage_{c}_trainable_tokens':trained})
                    records.append(r)
            con = duckdb.connect()
            con.register('fixture',pa.Table.from_pylist(records,schema=METRICS))
            con.execute('CREATE VIEW metrics AS SELECT * FROM fixture')
            with tempfile.TemporaryDirectory() as tmp:
                first = mixture.select(con,Path(tmp),self.prior,self.registry,False)
                counts = [(s['context'],s['seen_tokens'],s['trainable_tokens']) for s in first]
                # Drop the view before replacing its backing files.
                con.execute('DROP VIEW selected')
                second = mixture.select(con,Path(tmp),self.prior,self.registry,False)
                self.assertEqual(counts,[(s['context'],s['seen_tokens'],s['trainable_tokens']) for s in second])
                self.assertTrue(all(s['trainable_tokens']<s['seen_tokens'] for s in second))
                for s in second:
                    self.assertAlmostEqual(s['document_seen_share'],.65,delta=.001)
            con.close()
        finally:
            mixture.BUDGETS = old_budgets


if __name__=='__main__':
    unittest.main()
