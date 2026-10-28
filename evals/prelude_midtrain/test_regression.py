"""Controls for promotion decisions and paired multilingual uncertainty."""
import copy
import json
from pathlib import Path
import unittest
from compare_regression import compare


class RegressionTest(unittest.TestCase):
    def setUp(self):
        self.policy=json.loads(Path(__file__).with_name('gate_policy.json').read_text())
        self.policy['short_context_noninferiority']['paired_bootstrap_replicates']=2000
        self.base={'protocol_sha256':'control','metrics':{}}
        for task in self.policy['short_context_noninferiority']['maximum_drop_pp']:
            self.base['metrics'][task]=[
                dict(id=str(i),score=float(i%2),**({'language':lang} if task=='MultiPL-E' else {}))
                for i in range(200) for lang in (['py','cpp'] if task=='MultiPL-E' else [''])]

    def test_identical_and_catastrophic_controls(self):
        self.assertEqual(compare(self.base,self.base,self.policy)['status'],'pass')
        candidate=copy.deepcopy(self.base)
        for rows in candidate['metrics'].values():
            for row in rows:row['score']=0.
        self.assertEqual(compare(self.base,candidate,self.policy)['status'],'fail')

    def test_shared_language_ids_are_resampled_jointly(self):
        candidate=copy.deepcopy(self.base)
        for original,changed in zip(self.base['metrics']['MultiPL-E'],candidate['metrics']['MultiPL-E']):
            first=int(original['id'])%2
            original['score']=float(first if original['language']=='py' else 1-first)
            changed['score']=1-original['score']
        result=compare(self.base,candidate,self.policy)['metrics']['MultiPL-E']
        self.assertEqual(result['lower_pp'],0.)
        self.assertEqual(result['upper_pp'],0.)

    def test_mismatches_never_compare(self):
        candidate=copy.deepcopy(self.base)
        candidate['protocol_sha256']='changed'
        with self.assertRaises(ValueError):compare(self.base,candidate,self.policy)
        candidate=copy.deepcopy(self.base)
        candidate['metrics']['MATH'].pop()
        with self.assertRaises(ValueError):compare(self.base,candidate,self.policy)


if __name__=='__main__':unittest.main()
