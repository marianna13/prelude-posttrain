"""Small binary fixtures for the same bounded-memory initializer used on 9B."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from prepare_midtrain_tokenizer import encode_mean, matrix_mean, header, sha


class InitializationTest(unittest.TestCase):
    def test_initialized_9b_rows_against_current_policy(self):
        step2=Path('/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step2')
        checkpoint=step2/'base_hf'
        if not (checkpoint/'midtrain_initialization.json').is_file():
            self.skipTest('initialize the base HF checkpoint first')
        contract=json.loads((step2/'tokenizer_contract.json').read_text())
        audit=json.loads((checkpoint/'midtrain_initialization.json').read_text())
        self.assertEqual(audit['contract_sha256'],sha(step2/'tokenizer_contract.json'))
        self.assertTrue(audit['verified_all_other_model_bytes_unchanged'])
        for name,m in contract['matrices'].items():
            mean=matrix_mean(Path(contract['base_model'])/m['file'],{**m,'name':name})
            offset,_=header(checkpoint/m['file'])
            with open(checkpoint/m['file'],'rb') as f:
                for index in contract['initialization']['ids']:
                    f.seek(offset+m['data_offsets'][0]+index*len(mean))
                    self.assertEqual(f.read(len(mean)),mean)
        for name,expected in audit['shard_sha256'].items():
            self.assertEqual(sha(checkpoint/name),expected)
        self.assertEqual(json.loads((checkpoint/'config.json').read_text())['vocab_size'],262272)

    def test_matrix_means_and_bfloat_rounding(self):
        original = (np.arange(32).reshape(8,4)-16).astype(np.float32)/8
        expected = original.mean(axis=0,dtype=np.float64)
        for dtype in ("BF16","F32","F16"):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as d:
                if dtype=="BF16":
                    raw = (original.view("<u4") >> 16).astype("<u2").tobytes()
                else: raw = original.astype("<f4" if dtype=="F32" else "<f2").tobytes()
                meta={"dtype":dtype,"shape":[8,4],"data_offsets":[0,len(raw)]}
                h=json.dumps({"w":meta}).encode(); p=Path(d)/"model.safetensors"
                p.write_bytes(struct.pack("<Q",len(h))+h+raw)
                mean=matrix_mean(p,{**meta,"name":"w"},native_end=8)
                if dtype=="BF16":
                    actual=(np.frombuffer(mean,dtype="<u2").astype("<u4")<<16).view("<f4")
                else: actual=np.frombuffer(mean,dtype="<f4" if dtype=="F32" else "<f2")
                np.testing.assert_array_equal(actual,expected)
        # Halfway cases must round to the even bfloat mantissa.
        floats=np.array([0x3f808000,0x3f818000],dtype="<u4").view("<f4")
        np.testing.assert_array_equal(np.frombuffer(encode_mean(floats,"BF16"),dtype="<u2"),[0x3f80,0x3f82])


if __name__=="__main__": unittest.main(verbosity=2)
