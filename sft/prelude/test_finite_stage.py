"""Actual pinned-runtime finite tail and global-step resume contract."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from haliax import Axis
from levanter.data.loader import DataLoader
from levanter.data.text.datasets import LmDataConfig
from levanter.testing.helpers import use_test_mesh
from finite_stage import finite_stage, local_batch, safe_empty_loss
from prepare_midtrain_levanter import make_cache

TOKENIZER = Path('/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step2/tokenizer')


class FiniteStageTest(unittest.TestCase):
    def test_zero_target_microbatch_has_zero_finite_gradient(self):
        import haliax as hax
        import haliax.nn.loss as loss
        import jax
        import jax.numpy as jnp
        pos=Axis('position',4)
        original=loss.maybe_reduce_loss
        def objective(x, weights):
            return loss.maybe_reduce_loss(hax.named(x*x,pos),hax.mean,None,None,
                                         hax.named(weights,pos)).scalar()
        with safe_empty_loss():
            x=jnp.arange(4,dtype=jnp.float32)
            empty=jnp.zeros(4)
            self.assertEqual(float(objective(x,empty)),0.)
            np.testing.assert_array_equal(jax.grad(objective)(x,empty),0.)
            weights=jnp.array([0.,1.,1.,0.])
            np.testing.assert_allclose(jax.grad(objective)(x,weights),[0,1,2,0])
        self.assertIs(loss.maybe_reduce_loss,original)

    def test_tail_and_resume_cover_real_examples_once(self):
        # Five full packs in batches of four exercises a real partial tail.
        rows = [dict(id=str(i), ids=[1,100+i,101+i,2], loss=[0,1,1,1],
                     context=4, split='train') for i in range(5)]
        config = LmDataConfig(tokenizer=str(TOKENIZER))
        original_train, original_iter = type(config).train_set, DataLoader.iter_from_step
        with tempfile.TemporaryDirectory() as tmp, use_test_mesh() as mesh:
            cache = Path(tmp)/'cache'
            make_cache(rows, cache, TOKENIZER, 4)
            with finite_stage(config, cache, TOKENIZER, 4, 4, 607, 609):
                ds = config.train_set(Axis('position',4))
                loader = DataLoader(ds, batch_size=4, mesh=mesh,
                                    axis_resources={'batch':'data'}, max_buffered_batches=-1,
                                    fetch_batch_size=1)
                all_batches = list(loader.iter_from_step(607))
                self.assertEqual(len(all_batches),2)
                tokens=np.concatenate([np.asarray(b.tokens.array) for b in all_batches])
                weights=np.concatenate([np.asarray(b.loss_weight.array) for b in all_batches])
                real=np.any(weights,axis=1)
                self.assertEqual(sorted(tokens[real,1].tolist()),list(range(100,105)))
                self.assertEqual(int(weights.sum()),15)
                self.assertEqual(int((~real).sum()),3)
                np.testing.assert_array_equal(tokens[~real],0)
                resumed=list(loader.iter_from_step(608))
                self.assertEqual(len(resumed),1)
                np.testing.assert_array_equal(resumed[0].tokens.array,all_batches[1].tokens.array)
                self.assertEqual(len(list(loader.iter_from_step(609))),0)
                with self.assertRaisesRegex(ValueError,'outside finite stage'):
                    loader.iter_from_step(606)
            self.assertIs(type(config).train_set,original_train)
            self.assertIs(DataLoader.iter_from_step,original_iter)

    def test_invalid_resume_rejected(self):
        self.assertEqual(local_batch(707,607,1214),100)
        for step in (606,1822):
            with self.assertRaises(ValueError):
                local_batch(step,607,1214)


if __name__ == '__main__':
    unittest.main()
