"""Run inside the pinned Levanter image; real processor/cache/packing/causal loader.

No mocks and no model weights. Source adapter fixtures run separately under the
census environment; this verifies the actual training boundary in the image.
"""
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from haliax import Axis
from levanter.data.text.datasets import ChatDataset
from levanter.store.cache import SerialCacheWriter
from levanter.tokenizers import load_tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts/data"))
sys.path.insert(0, str(ROOT/"sft/levanter"))
from prelude_midtrain import CASCADE, normalize_messages, render, render_conversation
from test_prelude_midtrain import fixture
from midtrain_format import PreludeMidtrainFormat

STEP2 = Path("/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step2")


class LoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(STEP2/"tokenizer", local_files_only=True)
        cls.marin = load_tokenizer(str(STEP2/"tokenizer"))

    def load(self, rows, length, max_segments=64):
        fmt = PreludeMidtrainFormat(max_tokens=length)
        proc = fmt.build_preprocessor(self.marin)
        processed = proc(rows)
        temp = tempfile.TemporaryDirectory(dir=str(STEP2))
        self.addCleanup(temp.cleanup)
        with SerialCacheWriter(temp.name+"/cache", proc.output_exemplar) as w:
            w.write_batch(processed)
        ds = ChatDataset(w.result(), Axis("position", length), max_segments_per_example=max_segments,
                         slice_strategy=fmt.slice_strategy, mask_user_turns=fmt.mask_user_turns,
                         block_cross_document_attention=True)
        return ds

    def test_exact_packed_ids_shift_padding_and_attention(self):
        m, tools = normalize_messages(fixture(), CASCADE, Counter())
        ids, loss = render_conversation(self.tok, m, tools)
        doc = render({"id":"doc", "kind":"document", "text":"\tdef x():\n    return 1\n", "mix_source":"code"}, self.tok)
        rows = [{"id":"chat", "ids":ids, "loss":loss}, doc]
        length = len(ids)+len(doc["ids"])+7
        ds = self.load(rows, length)
        e = asyncio.run(ds.getitem_async(0))
        expected_ids = np.array(ids+doc["ids"]+[0]*7, np.int32)
        expected_targets = np.array(loss+doc["loss"]+[0]*7, np.float32)
        expected_weights = np.roll(expected_targets, -1); expected_weights[-1] = 0
        np.testing.assert_array_equal(np.asarray(e.tokens), expected_ids)
        np.testing.assert_array_equal(np.asarray(e.loss_weight), expected_weights)
        seg, _ = e.attn_mask.segment_ids
        np.testing.assert_array_equal(np.asarray(seg), [0]*len(ids)+[1]*len(doc["ids"])+[-1]*7)
        charged = np.nonzero(np.asarray(e.loss_weight))[0]
        self.assertTrue(np.all(np.asarray(seg)[charged] == np.asarray(seg)[charged+1]))
        self.assertEqual(float(e.loss_weight[len(ids)-1]), 0.)  # next document BOS
        # Mask materialization verifies cross-document attention is actually blocked.
        from levanter.grug.attention import reference_attention
        import jax.numpy as jnp
        q = jnp.ones((1,length,1,2)); k = jnp.zeros_like(q).at[0,0,0,0].set(100.)
        v = jnp.zeros_like(q).at[0,0,0,1].set(300.)
        result = np.asarray(reference_attention(q,k,v,e.attn_mask,logits_dtype=jnp.float32))[0,:,0,1]
        np.testing.assert_allclose(result[:len(ids)], 300., atol=.01)
        np.testing.assert_array_equal(result[len(ids):], 0.)

    def test_real_64k_boundary_and_overlength_rejection(self):
        ids = [1]+[123]*65534+[2]
        loss = [0]+[1]*65535
        ds = self.load([{"id":"64k", "ids":ids,"loss":loss}],65536)
        e = asyncio.run(ds.getitem_async(0))
        np.testing.assert_array_equal(np.asarray(e.tokens),ids)
        np.testing.assert_array_equal(np.asarray(e.loss_weight),[1]*65535+[0])
        with self.assertRaisesRegex(ValueError, "rows must be"):
            self.load([{"id":"over", "ids":ids+[2], "loss":loss+[1]}],65536)

    def test_empty_content_and_unpacked_masks(self):
        messages = [{"role":"system","content":""},{"role":"user","content":""},
                    {"role":"assistant","content":""},{"role":"tool","content":""},
                    {"role":"assistant","content":""}]
        ids, loss = render_conversation(self.tok,messages)
        ds = self.load([{"id":"empty","ids":ids,"loss":loss}]*2,len(ids)+4,max_segments=1)
        for index in (0,1):
            e=asyncio.run(ds.getitem_async(index))
            np.testing.assert_array_equal(np.asarray(e.tokens),ids+[0]*4)
            np.testing.assert_array_equal(np.asarray(e.loss_weight),loss[1:]+[0]*5)
        with self.assertRaisesRegex(ValueError,"mask_user_turns"):
            PreludeMidtrainFormat(mask_user_turns=False).build_preprocessor(self.marin)
        with self.assertRaisesRegex(ValueError,"implicit truncation"):
            PreludeMidtrainFormat(slice_strategy="left").build_preprocessor(self.marin)

    def test_frozen_bundle_and_export(self):
        from prepare_midtrain_tokenizer import export_tokenizer, sha
        c=json.loads((STEP2/"tokenizer_contract.json").read_text())
        self.assertEqual(self.tok.eos_token_id,262146)
        self.assertEqual(self.marin.eos_token_id,262146)
        for name,expected in c["tokenizer_files_sha256"].items():
            self.assertEqual(sha(STEP2/"tokenizer"/name),expected)
        with tempfile.TemporaryDirectory(dir=STEP2) as d:
            p=Path(d);(p/"config.json").write_text(json.dumps({"vocab_size":262272}))
            export_tokenizer(STEP2/"tokenizer_contract.json",p)
            exported=AutoTokenizer.from_pretrained(p,local_files_only=True)
            self.assertEqual(exported.eos_token_id,262146)
            self.assertEqual(exported("<think>",add_special_tokens=False)["input_ids"],[262147])
            self.assertEqual(json.loads((p/"generation_config.json").read_text())["eos_token_id"],[262146,2])

    def test_trainer_format_keeps_sft_default(self):
        from scripts_grug.sft_main import _format
        kwargs={"messages_field":"messages","chat_template":None,"seq_len":65536,"pack":True}
        sft=_format("prerendered",**kwargs)
        mid=_format("midtrain",**kwargs)
        self.assertEqual(sft.slice_strategy,"left")
        self.assertEqual(mid.slice_strategy,"raise")
        self.assertTrue(mid.mask_user_turns)

    def test_midtrain_keeps_fully_masked_tool_continuations(self):
        from scripts_grug.grug_datakit_chat import SnowballPrerenderedChatFormat
        rows = [{'id': 'masked', 'ids': [1, 42, 43], 'loss': [0, 0, 0]},
                {'id': 'target', 'ids': [1, 123, 2], 'loss': [0, 1, 1]}]
        ds = self.load(rows, 8)
        example = asyncio.run(ds.getitem_async(0))
        np.testing.assert_array_equal(np.asarray(example.tokens), [1, 42, 43, 1, 123, 2, 0, 0])
        np.testing.assert_array_equal(np.asarray(example.loss_weight), [0, 0, 0, 1, 1, 0, 0, 0])
        # Existing SFT formats retain their previous validation and metadata.
        sft = SnowballPrerenderedChatFormat().build_preprocessor(self.marin)
        self.assertNotIn('allow_zero_loss', sft.metadata)
        with self.assertRaisesRegex(ValueError, 'trains no token'):
            sft(rows[:1])

    def test_resume_exemplar_uses_padded_vocabulary(self):
        from prepare_midtrain_tokenizer import install_checkpoint_hooks
        from levanter.testing.helpers import use_test_mesh
        import levanter.main.train_lm as train_lm
        install_checkpoint_hooks(STEP2 / 'tokenizer_contract.json')
        with use_test_mesh():
            actual = train_lm.round_axis_for_partitioning(Axis('vocab', len(self.tok)), {})
            self.assertEqual(actual.size, 262272)
            self.assertEqual(len(self.tok), 262151)
            self.assertEqual(train_lm.round_axis_for_partitioning(Axis('embed', 16), {}).size, 16)
            with self.assertRaisesRegex(ValueError, 'unexpected Prelude training vocabulary'):
                train_lm.round_axis_for_partitioning(Axis('vocab', 123), {})

    def test_tiny_qwen_hf_import_and_checkpoint_export(self):
        # Exercise the real converter with Prelude's full vocabulary, tiny hidden
        # layers, and the export hook. No 9B allocation and no training step.
        from transformers import Qwen3Config as HfConfig, Qwen3ForCausalLM
        from levanter.models.qwen import Qwen3Config
        from prepare_midtrain_tokenizer import install_checkpoint_hooks, export_tokenizer, sha
        from levanter.testing.helpers import use_test_mesh
        import jax.numpy as jnp
        c=json.loads((STEP2/"tokenizer_contract.json").read_text())
        install_checkpoint_hooks(STEP2/"tokenizer_contract.json")
        with tempfile.TemporaryDirectory(dir=STEP2) as d:
            source,dest=Path(d)/"source",Path(d)/"export"
            config=HfConfig(vocab_size=262272,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                            num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=16,
                            tie_word_embeddings=False)
            model=Qwen3ForCausalLM(config)
            model.save_pretrained(source)
            export_tokenizer(STEP2/"tokenizer_contract.json",source)
            del model
            converter=Qwen3Config().hf_checkpoint_converter(str(source))
            with use_test_mesh():
                converted=converter.load_pretrained(Qwen3Config().model_type,dtype=jnp.float32)
                self.assertEqual(converted.Vocab.size,262272)
                converter.save_pretrained(converted,str(dest),save_reference_code=False)
            self.assertEqual(json.loads((dest/"config.json").read_text())["vocab_size"],262272)
            self.assertEqual(AutoTokenizer.from_pretrained(dest,local_files_only=True).eos_token_id,262146)
            for name,expected in c["tokenizer_files_sha256"].items(): self.assertEqual(sha(dest/name),expected)


if __name__ == "__main__": unittest.main(verbosity=2)
