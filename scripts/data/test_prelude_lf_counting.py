"""Verify exact native IDs across the guarded document counting boundaries."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('lf_counting',Path(__file__).with_name('prelude_lf_counting.py'))
counting=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(counting)
TOKENIZER=Path('/p/data1/mmlaion/marianna/sft_sources/prelude_chat_tokenizer/tokenizer.json')


@unittest.skipUnless(TOKENIZER.exists(),'Frozen Prelude tokenizer is required')
class LFCountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer=Tokenizer.from_file(str(TOKENIZER))

    def test_exact_ids_and_lengths_with_unicode_markup_and_empty_inputs(self):
        rng=random.Random(913)
        alphabet='abCD []{}▁αé中🙂\r\t\n '
        examples=['','\n','\n\n','hello\nworld','a'*1000+'\n'+'b'*1000,
                  '<think>\nreason\n</think>\n<|im_end|>','e\u0301\n\u0301e','x\r\ny\t\nz']
        examples+=[''.join(rng.choices(alphabet,k=rng.randrange(1,10000))) for _ in range(100)]
        counter=counting.LFCountingTokenizer(self.tokenizer,TOKENIZER)
        expected=[]
        for text in examples:
            whole=self.tokenizer.encode(text,add_special_tokens=False).ids
            split=[token for piece in counting.chunks(text,target=31)
                   for token in self.tokenizer.encode(piece,add_special_tokens=False).ids]
            self.assertEqual(whole,split)
            expected.append(len(whole))
        self.assertEqual([len(e.ids) for e in counter.encode_batch(examples)],expected)

    def test_boundary_changing_tokenizer_is_rejected_and_offsets_are_unavailable(self):
        config=json.loads(TOKENIZER.read_text())
        config['pre_tokenizer']['prepend_scheme']='always'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tokenizer.json';path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):counting.LFCountingTokenizer(self.tokenizer,path)
        with self.assertRaises(RuntimeError):_ = counting.CountEncoding(3).offsets


if __name__=='__main__':unittest.main()
