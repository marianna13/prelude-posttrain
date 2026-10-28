#!/usr/bin/env python3
"""CPU qualification: tiny real Qwen3, full Prelude vocabulary, native Levanter.

Mixed document/conversation data -> actual cache/packing -> two Adam updates,
native checkpoint restore -> identical second update -> HF export/reload.
This is a correctness check, not a 9B or long-context GPU performance canary.
"""
import argparse
import asyncio
import dataclasses
import json
from pathlib import Path
import sys

import equinox as eqx
from haliax import Axis
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from transformers import AutoTokenizer, Qwen3Config as HfConfig, Qwen3ForCausalLM
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.data.text.datasets import NamedLmDataset
from levanter.models.qwen import Qwen3Config
from levanter.testing.helpers import use_test_mesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts/data'))
from prelude_midtrain import render_conversation
from prepare_midtrain_tokenizer import export_tokenizer, install_checkpoint_hooks, sha
from prepare_midtrain_levanter import make_cache, stage_dataset, write


def same_tree(a, b):
    aa, ta = jax.tree.flatten(eqx.filter(a, eqx.is_array))
    bb, tb = jax.tree.flatten(eqx.filter(b, eqx.is_array))
    if ta != tb:
        raise ValueError('restored structure differs')
    for x, y in zip(aa, bb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def run(args):
    out, step2 = args.out, args.step2
    if out.exists():
        raise ValueError('use a fresh smoke output directory')
    out.mkdir(parents=True)
    tokenizer = step2 / 'tokenizer'
    contract = step2 / 'tokenizer_contract.json'
    tok = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    messages = [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'U'},
                {'role': 'assistant', 'reasoning_content': 'R', 'content': 'Call.',
                 'tool_calls': [{'type': 'function', 'function': {'name': 'x', 'arguments': '{}'}}]},
                {'role': 'tool', 'content': 'T'}, {'role': 'assistant', 'content': 'A'}]
    ids, loss = render_conversation(tok, messages)
    context = max(128, 1 << (len(ids) + 16 - 1).bit_length())
    doc_ids = [1] + tok('ExactCase\n  code α\n', add_special_tokens=False)['input_ids'] + [2]
    rows = [dict(id='document', ids=doc_ids, loss=[0] + [1] * (len(doc_ids) - 1), context=context, split='train'),
            dict(id='conversation', ids=ids, loss=loss, context=context, split='train')]
    if not set(range(262145, 262151)) <= set(ids):
        raise ValueError('fixture must exercise all six added tokens')
    counts = make_cache(rows, out / 'cache', tokenizer, context)
    raw = stage_dataset(out / 'cache', tokenizer, context, shuffle_seed=None)
    dataset = NamedLmDataset(raw, Axis('position', context))
    if asyncio.run(dataset.async_len()) != 1:
        raise ValueError('mixed fixture should occupy a single pack')
    example = asyncio.run(dataset.getitem_async(0))
    source = out / 'source_hf'
    cfg = HfConfig(vocab_size=262272, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                   num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=context,
                   tie_word_embeddings=False)
    torch.manual_seed(20261017)
    hf = Qwen3ForCausalLM(cfg)
    with torch.no_grad():
        for weight in [hf.model.embed_tokens.weight, hf.lm_head.weight]:
            weight[262145:262151] = weight[:262144].double().mean(0).to(weight.dtype)
    hf.save_pretrained(source)
    export_tokenizer(contract, source)
    del hf
    install_checkpoint_hooks(contract)
    converter = Qwen3Config().hf_checkpoint_converter(str(source))
    optimizer = optax.adam(1e-4)
    with use_test_mesh():
        model = converter.load_pretrained(Qwen3Config().model_type, dtype=jnp.float32)
        params = eqx.filter(model, eqx.is_inexact_array)
        opt_state = optimizer.init(params)

        @eqx.filter_jit
        def update(m, state, sample):
            value, grad = eqx.filter_value_and_grad(lambda mm: mm.compute_next_token_loss(sample).array)(m)
            updates, state = optimizer.update(grad, state, eqx.filter(m, eqx.is_inexact_array))
            return eqx.apply_updates(m, updates), state, value, grad

        first_model, first_opt, loss1, grad = update(model, opt_state, example)
        norms = {}
        for name, weight in [('embedding', grad.embeddings.token_embeddings.weight), ('head', grad.get_lm_head())]:
            axis = [a.name for a in weight.axes].index('vocab')
            added = np.take(np.asarray(weight.array), np.arange(262145, 262151), axis=axis)
            added = np.moveaxis(added, axis, 0).reshape(6, -1)
            values = np.linalg.norm(added, axis=1)
            if not np.isfinite(values).all() or not np.all(values > 0):
                raise ValueError('missing added-token gradient: ' + name)
            norms[name] = values.tolist()
        checkpoint = out / 'checkpoint-step-1'
        state1 = {'model': first_model, 'optimizer': first_opt, 'step': jnp.array(1)}
        save_checkpoint(state1, 1, checkpoint, is_temporary=False)
        restored = load_checkpoint(state1, checkpoint)
        same_tree(state1, restored)
        uninterrupted, opt2, loss2, _ = update(first_model, first_opt, example)
        resumed, resumed_opt, resumed_loss, _ = update(restored['model'], restored['optimizer'], example)
        same_tree((uninterrupted, opt2), (resumed, resumed_opt))
        np.testing.assert_array_equal(np.asarray(loss2), np.asarray(resumed_loss))
        if not np.isfinite([float(loss1), float(loss2)]).all():
            raise ValueError('nonfinite training loss')
        # An entirely masked continuation must contribute finite zero loss.
        zero = dataclasses.replace(example, loss_weight=example.loss_weight * 0)
        zero_loss = float(model.compute_next_token_loss(zero).array)
        if zero_loss != 0:
            raise ValueError('fully masked example does not have finite zero loss')
        export = out / 'hf_export'
        converter.save_pretrained(resumed, str(export), save_reference_code=False)
        reloaded_converter = Qwen3Config().hf_checkpoint_converter(str(export))
        reloaded = reloaded_converter.load_pretrained(Qwen3Config().model_type, dtype=jnp.float32)
        same_tree(resumed, reloaded)
        exported_tok = AutoTokenizer.from_pretrained(export, local_files_only=True)
        for name, expected in json.loads(contract.read_text())['tokenizer_files_sha256'].items():
            if sha(export / name) != expected:
                raise ValueError('exported tokenizer changed')
        if exported_tok.eos_token_id != 262146 or reloaded.Vocab.size != 262272:
            raise ValueError('exported vocabulary changed')
    result = dict(status='passed', scope='tiny Qwen3 CPU correctness only; target 9B GPU canary remains on MN5',
        context=context, counts=counts, loss=[float(loss1), float(loss2)], zero_mask_loss=zero_loss,
        added_token_gradient_norms=norms, checks=['mixed formats through real cache and packer',
            'finite Adam updates', 'all six added embedding/head rows have gradients',
            'native model+optimizer checkpoint reload', 'bit-identical resumed second update',
            'HF export/reimport exact model leaves', 'frozen tokenizer and padded vocabulary retained'],
        script_sha256=sha(__file__))
    write(out / 'smoke.json', result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step2', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
