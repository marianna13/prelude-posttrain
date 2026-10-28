"""Qwen3 attention adapter for the Grug FA4/CuTe GPU kernel."""

from __future__ import annotations

import math
from typing import Any

import haliax as hax
import jax.numpy as jnp
from levanter.grug.attention import gpu_fa4_cute_attention
from levanter.grug.attention._core import AttentionMask as Fa4AttentionMask
from levanter.layers import attention as levanter_attention
from levanter.layers.attention_mask import AttentionMask as LevanterAttentionMask


_ORIGINAL_DOT_PRODUCT_ATTENTION = None
_FA4_ENABLED = False


def _qwen3_named_inputs(query: Any, key: Any, value: Any):
    try:
        query = query.rearrange(("batch", "position", "kv_head", "q_heads_per_group", "head_size"))
        key = key.rearrange(("batch", "key_position", "kv_head", "head_size"))
        value = value.rearrange(("batch", "key_position", "kv_head", "head_size"))
    except (AttributeError, ValueError):
        return None

    try:
        query_heads = query.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
    except (AttributeError, ValueError):
        return None

    return query, query_heads, key, value


def _as_fa4_attention_mask(mask: Any):
    if mask is None:
        return Fa4AttentionMask.causal()
    if isinstance(mask, Fa4AttentionMask):
        return mask
    if not isinstance(mask, LevanterAttentionMask):
        return None
    if (
        not mask.is_causal
        or mask.causal_offset is not None
        or mask.explicit_mask is not None
        or mask.bidirectional_window is not None
    ):
        return None

    segment_ids = mask.segment_ids
    if segment_ids is None:
        raw_segment_ids = None
    elif isinstance(segment_ids, tuple):
        if len(segment_ids) != 2:
            return None
        raw_segment_ids = tuple(
            segment.array if isinstance(segment, hax.NamedArray) else segment
            for segment in segment_ids
        )
    else:
        raw_segment_ids = segment_ids.array if isinstance(segment_ids, hax.NamedArray) else segment_ids
        raw_segment_ids = (raw_segment_ids, raw_segment_ids)
    return Fa4AttentionMask(
        is_causal=True,
        segment_ids=raw_segment_ids,
        sliding_window=mask.sliding_window,
        fa4_bounds=None,
    )


def _fa4_dot_product_attention(
    query_position,
    key_position,
    key_dimension,
    query,
    key,
    value,
    mask=None,
    bias=None,
    attention_dtype=None,
    precision=None,
    use_flash=None,
    attn_backend=None,
    flash_block_size=None,
    dropout=0.0,
    *,
    logits_soft_cap=None,
    scaling_factor=None,
    inference=True,
    prng=None,
    attn_sink=None,
):
    if not _FA4_ENABLED:
        return _ORIGINAL_DOT_PRODUCT_ATTENTION(
            query_position,
            key_position,
            key_dimension,
            query,
            key,
            value,
            mask,
            bias=bias,
            attention_dtype=attention_dtype,
            precision=precision,
            use_flash=use_flash,
            attn_backend=attn_backend,
            flash_block_size=flash_block_size,
            dropout=dropout,
            logits_soft_cap=logits_soft_cap,
            scaling_factor=scaling_factor,
            inference=inference,
            prng=prng,
            attn_sink=attn_sink,
        )

    unsupported = (
        query_position != "position"
        or key_position != "key_position"
        or key_dimension != "head_size"
        or bias is not None
        or dropout != 0.0
        or logits_soft_cap is not None
        or attn_sink is not None
    )
    named_inputs = None if unsupported else _qwen3_named_inputs(query, key, value)
    if named_inputs is None:
        return _ORIGINAL_DOT_PRODUCT_ATTENTION(
            query_position,
            key_position,
            key_dimension,
            query,
            key,
            value,
            mask,
            bias=bias,
            attention_dtype=attention_dtype,
            precision=precision,
            use_flash=use_flash,
            attn_backend=attn_backend,
            flash_block_size=flash_block_size,
            dropout=dropout,
            logits_soft_cap=logits_soft_cap,
            scaling_factor=scaling_factor,
            inference=inference,
            prng=prng,
            attn_sink=attn_sink,
        )

    query_named, query_heads_named, key_named, value_named = named_inputs
    query_array = query_heads_named.array
    key_array = key_named.array
    value_array = value_named.array
    if scaling_factor is not None:
        default_scaling_factor = 1.0 / math.sqrt(query_named.resolve_axis("head_size").size)
        query_array = query_array * (scaling_factor / default_scaling_factor)

    attention_mask = _as_fa4_attention_mask(mask)
    if attention_mask is None:
        return _ORIGINAL_DOT_PRODUCT_ATTENTION(
            query_position,
            key_position,
            key_dimension,
            query,
            key,
            value,
            mask,
            bias=bias,
            attention_dtype=attention_dtype,
            precision=precision,
            use_flash=use_flash,
            attn_backend=attn_backend,
            flash_block_size=flash_block_size,
            dropout=dropout,
            logits_soft_cap=logits_soft_cap,
            scaling_factor=scaling_factor,
            inference=inference,
            prng=prng,
            attn_sink=attn_sink,
        )
    output_array = gpu_fa4_cute_attention(
        query_array,
        key_array,
        value_array,
        attention_mask,
    )
    batch_axis = query_named.resolve_axis("batch")
    position_axis = query_named.resolve_axis("position")
    kv_head_axis = query_named.resolve_axis("kv_head")
    query_heads_per_group_axis = query_named.resolve_axis("q_heads_per_group")
    head_size_axis = query_named.resolve_axis("head_size")
    output_array = output_array.reshape(
        (
            batch_axis.size,
            position_axis.size,
            kv_head_axis.size,
            query_heads_per_group_axis.size,
            head_size_axis.size,
        )
    ).transpose(0, 2, 3, 1, 4)
    return hax.NamedArray(
        output_array,
        axis_names=(
            batch_axis,
            kv_head_axis,
            query_heads_per_group_axis,
            position_axis,
            head_size_axis,
        ),
    )


def install_qwen3_fa4_attention() -> None:
    """Route supported Qwen3 attention calls through FA4/CuTe on GPU."""
    global _FA4_ENABLED, _ORIGINAL_DOT_PRODUCT_ATTENTION
    if _ORIGINAL_DOT_PRODUCT_ATTENTION is None:
        _ORIGINAL_DOT_PRODUCT_ATTENTION = levanter_attention.dot_product_attention
    levanter_attention.dot_product_attention = _fa4_dot_product_attention
    _FA4_ENABLED = True
