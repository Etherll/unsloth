# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DeepSeek Ulysses CP attention scores each query chunk only against the keys it can attend to:
V4 sink attention the causal sliding window plus the compressed blocks, V3.2 sparse MLA the causal
prefix. Outputs and gradients must match the full-width masked computation, with fewer FLOPs."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from unsloth.utils import ulysses_deepseek as ud


def _dense_sink(q, kv, block_bias, sinks, window, scaling):
    # Full-width reference: every query row against every token key, masked afterwards.
    batch, _, rows, _ = q.shape
    full_seq = rows
    dtype = q.dtype
    idx = torch.arange(full_seq)
    diff = idx[:, None] - idx[None, :]
    keep = diff >= 0
    if window is not None:
        keep = keep & (diff < window)
    mask = torch.zeros((1, 1, rows, full_seq), dtype = dtype)
    mask.masked_fill_(~keep.view(1, 1, rows, full_seq), torch.finfo(dtype).min)
    extra = kv.shape[2] - full_seq
    if extra > 0:
        if block_bias is not None:
            mask = torch.cat([mask.expand(batch, -1, -1, -1), block_bias.to(dtype)], dim = -1)
        else:
            mask = F.pad(mask, (0, extra), value = 0.0)
    logits = torch.matmul(q, kv.transpose(2, 3)) * scaling + mask
    sink = sinks.reshape(1, -1, 1, 1).expand(batch, -1, rows, -1)
    combined = torch.cat([logits, sink], dim = -1)
    combined = combined - combined.max(dim = -1, keepdim = True).values
    probs = F.softmax(combined, dim = -1, dtype = combined.dtype)[..., :-1].to(kv.dtype)
    return torch.matmul(probs, kv).transpose(1, 2).contiguous()


def _dense_mla(q, k, v, topk, scaling):
    batch, _, rows, _ = q.shape
    full_seq = k.shape[2]
    idx = torch.arange(full_seq)
    causal = idx[:rows, None] >= idx[None, :]
    selected = torch.zeros((batch, rows, full_seq), dtype = torch.bool)
    selected.scatter_(-1, topk.long(), True)
    keep = (selected & causal.unsqueeze(0)).unsqueeze(1)
    logits = torch.matmul(q, k.transpose(2, 3)) * scaling
    logits = logits.masked_fill(~keep, torch.finfo(logits.dtype).min)
    probs = F.softmax(logits, dim = -1, dtype = logits.dtype)
    return torch.matmul(probs, v).transpose(1, 2).contiguous()


def _block_bias(batch, seq, n_blocks, ratio, dtype):
    # Compressed block b covers tokens [b * ratio, (b + 1) * ratio): visible once complete.
    q_idx = torch.arange(seq)[:, None]
    done = (torch.arange(n_blocks)[None, :] + 1) * ratio <= q_idx + 1
    bias = torch.zeros((batch, 1, seq, n_blocks), dtype = dtype)
    return bias.masked_fill_(~done.view(1, 1, seq, n_blocks), float("-inf"))


def _close(actual, expected, dtype):
    # expected is the float64 full-width reference on the same (rounded) inputs.
    if dtype == torch.float32:
        torch.testing.assert_close(actual.double(), expected, rtol = 1e-5, atol = 1e-6)
    else:  # bf16 rounding of the outputs and the probabilities: relative to the tensor's scale
        err = (actual.double() - expected).abs().max()
        assert err <= 3e-2 * expected.abs().max(), (err, expected.abs().max())


# Lengths around the window (8) and chunk (16 / 7) boundaries, odd lengths included.
SINK_CASES = [
    (seq, chunk, window, n_blocks)
    for seq in (7, 8, 9, 16, 17, 33, 64)
    for chunk in (16, 7)
    for window in (8, None, 1)
    for n_blocks in (0, 3)
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq,chunk,window,n_blocks", SINK_CASES)
def test_sink_attention_matches_full_width(seq, chunk, window, n_blocks, dtype, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CP_ATTN_CHUNK", str(chunk))
    g = torch.Generator().manual_seed(seq * 31 + chunk)
    batch, heads, dim = 2, 3, 8
    q = torch.randn(batch, heads, seq, dim, generator = g).to(dtype)
    kv = torch.randn(batch, 1, seq + n_blocks, dim, generator = g).to(dtype)
    sinks = torch.randn(heads, generator = g).to(dtype)
    bias = _block_bias(batch, seq, n_blocks, 4, dtype) if n_blocks else None
    ins = [q, kv, sinks]

    leaves = [x.clone().requires_grad_() for x in ins]
    out = ud._sink_attention(leaves[0], leaves[1], bias, leaves[2], window, 0.3, 0.0, True)
    grad_out = torch.randn(out.shape, generator = g).to(dtype)
    out.backward(grad_out)
    ref_leaves = [x.double().requires_grad_() for x in ins]
    ref_bias = None if bias is None else bias.double()
    ref = _dense_sink(ref_leaves[0], ref_leaves[1], ref_bias, ref_leaves[2], window, 0.3)
    ref.backward(grad_out.double())

    _close(out, ref, dtype)
    for got, want in zip(leaves, ref_leaves):
        _close(got.grad, want.grad, dtype)
    with torch.no_grad():  # the no-grad route (no recompute) gives the same values
        _close(ud._sink_attention(q, kv, bias, sinks, window, 0.3, 0.0, False), ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq", [5, 8, 9, 16, 17, 33])
@pytest.mark.parametrize("chunk", [16, 7])
def test_sparse_mla_matches_full_width(seq, chunk, dtype, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CP_ATTN_CHUNK", str(chunk))
    g = torch.Generator().manual_seed(seq * 17 + chunk)
    batch, heads = 2, 3
    q = torch.randn(batch, heads, seq, 8, generator = g).to(dtype)
    k = torch.randn(batch, heads, seq, 8, generator = g).to(dtype)
    v = torch.randn(batch, heads, seq, 6, generator = g).to(dtype)
    # Like the indexer: early rows carry future (masked) indices, up to seq - 1.
    topk = torch.randint(0, seq, (batch, seq, 4), generator = g)
    topk[..., 0] = torch.arange(seq)
    ins = [q, k, v]

    leaves = [x.clone().requires_grad_() for x in ins]
    out = ud._sparse_mla_attention(*leaves, topk, 0.3, 0.0, True)
    grad_out = torch.randn(out.shape, generator = g).to(dtype)
    out.backward(grad_out)
    ref_leaves = [x.double().requires_grad_() for x in ins]
    ref = _dense_mla(*ref_leaves, topk, 0.3)
    ref.backward(grad_out.double())

    _close(out, ref, dtype)
    for got, want in zip(leaves, ref_leaves):
        _close(got.grad, want.grad, dtype)


def _flops(fn, *args):
    with FlopCounterMode(display = False) as counter:
        fn(*args)
    return counter.get_total_flops()


def test_chunked_attention_scores_only_reachable_keys(monkeypatch):
    # 256 tokens in 64-row chunks: the 8-token window needs at most 71 keys per chunk, causal
    # sparse MLA on average 160 of 256. Full-width scoring would cost 256 keys per chunk.
    monkeypatch.setenv("UNSLOTH_CP_ATTN_CHUNK", "64")
    seq, heads, dim = 256, 2, 16
    q = torch.randn(1, heads, seq, dim)
    kv = torch.randn(1, 1, seq, dim)
    sinks = torch.randn(heads)
    dense = 2 * (2 * heads * seq * seq * dim)  # q @ kv^T and probs @ kv
    with torch.no_grad():
        sink = _flops(ud._sink_attention, q, kv, None, sinks, 8, 0.3, 0.0, False)
        assert sink <= 0.3 * dense, (sink, dense)

        k, v = torch.randn(1, heads, seq, dim), torch.randn(1, heads, seq, dim)
        topk = torch.arange(seq).view(1, seq, 1)
        mla = _flops(ud._sparse_mla_attention, q, k, v, topk, 0.3, 0.0, False)
        assert mla <= 0.7 * dense, (mla, dense)
