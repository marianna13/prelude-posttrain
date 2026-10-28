"""Use the frozen finite cache and translate global checkpoint steps to stage batches."""
from contextlib import contextmanager
import math


@contextmanager
def safe_empty_loss():
    """Avoid a masked 0/0 in the pinned Haliax weighted mean's backward pass.

    Its where(denom != 0, numerator/denom, 0) has zero forward loss but NaN
    gradients for an all-padding microbatch. Guard the denominator itself.
    """
    import haliax as hax
    import haliax.nn.loss as loss
    from jax import numpy as jnp
    original = loss.maybe_reduce_loss

    def safe_reduce(arr, reduction, reduction_axis, where, weight):
        if (weight is not None or where is not None) and reduction_axis != () and (
            reduction is hax.mean or reduction is jnp.mean or reduction is loss.UNSPECIFIED
        ):
            effective = loss._resolve_effective_weight(arr, weight, where)
            numerator = hax.sum(arr * effective, axis=reduction_axis)
            denominator = hax.sum(effective, axis=reduction_axis)
            return numerator / hax.where(denominator != 0, denominator, hax.ones_like(denominator))
        return original(arr, reduction, reduction_axis, where, weight)

    loss.maybe_reduce_loss = safe_reduce
    try:
        yield
    finally:
        loss.maybe_reduce_loss = original


def local_batch(global_step, stage_start_step, stage_batches):
    step = int(global_step or 0) - stage_start_step
    if not 0 <= step <= stage_batches:
        raise ValueError(f'checkpoint step {global_step} outside finite stage '
                         f'[{stage_start_step}, {stage_start_step + stage_batches}]')
    return step


@contextmanager
def finite_stage(data_config, cache_path, tokenizer_path, context, batch_size, stage_start_step, end_step):
    """Scope compatibility hooks to this dataset; restore the generic SFT methods.

    Levanter's loader zero-pads finite tails, including loss weights. Its trainer
    resumes loaders using the global optimizer step, whereas a fresh stage cache
    starts at local batch zero. Both cross-stage and same-stage resumes need the
    same subtraction, independent of whether an output checkpoint already exists.
    """
    from levanter.data.loader import DataLoader
    from levanter.data.text.datasets import NamedLmDataset
    from levanter.utils.thread_utils import blocking_wait
    from prepare_midtrain_levanter import stage_dataset

    cls = type(data_config)
    original_train_set, original_iter = cls.train_set, DataLoader.iter_from_step
    named = None
    batches = None
    info = {'stage_start_step': stage_start_step}

    def train_set(self, Pos, *args, **kwargs):
        nonlocal named, batches
        if self is not data_config:
            return original_train_set(self, Pos, *args, **kwargs)
        # train_lm initializes JAX distributed before calling train_set. Creating
        # the seeded shuffle earlier would initialize a GPU backend too soon.
        dataset = stage_dataset(cache_path, tokenizer_path, context)
        count = blocking_wait(dataset.async_len())
        batches = math.ceil(count / batch_size)
        if not stage_start_step < end_step <= stage_start_step + batches:
            raise ValueError('requested steps exceed finite stage or precede its start')
        info.update(packed_sequences=count, batches=batches)
        named = NamedLmDataset(dataset, Pos)
        return named

    def iter_from_step(self, start_from_batch=None):
        if named is not None and self.data_store is named:
            return original_iter(self, local_batch(start_from_batch, stage_start_step, batches))
        return original_iter(self, start_from_batch)

    cls.train_set, DataLoader.iter_from_step = train_set, iter_from_step
    try:
        with safe_empty_loss():
            yield info
    finally:
        cls.train_set, DataLoader.iter_from_step = original_train_set, original_iter
