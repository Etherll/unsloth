# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DeepSeek-V4 Ulysses CP on CPU gloo ranks: the compressor layers read the step's full position_ids
from the CP manager with no collective (forward and gradient checkpoint recompute alike), so ranks
cannot disagree on issuing one, and loss / grads are bitwise those of gathering in every layer."""

import datetime
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from tests.test_ulysses_deepseek import KINDS, VOCAB, _loss, _manager, _model, _spawn
from unsloth.utils import attention_dispatch as ad
from unsloth.utils import ulysses_deepseek as ud

pytestmark = pytest.mark.skipif("v4" not in KINDS, reason = "transformers without DeepSeek-V4")


def _gather_every_layer(position_ids, group):
    pos_local = position_ids.contiguous()
    pos_list = [torch.empty_like(pos_local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(pos_list, pos_local, group = group)
    return torch.cat(pos_list, dim = -1)


def _step(model, manager, ids, count, gathers, calls):
    gathers.clear()
    calls.clear()
    model.zero_grad(set_to_none = True)
    inputs = {"input_ids": ids.clone(), "labels": ids.clone()}
    manager._prepare_inputs(inputs)
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"], position_ids = inputs["position_ids"], use_cache = False
        )
        loss = _loss(out.logits, inputs["shift_labels"], count)
        loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads, len(gathers), len(calls)


def _worker(rank, world, use_reentrant):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _model("v4")
    compressors = sum(
        getattr(m, "compressor", None) is not None
        for m in model.modules()
        if type(m).__name__ == "DeepseekV4Attention"
    )
    assert compressors == 2, compressors
    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs = {"use_reentrant": use_reentrant}
    )

    gathers, calls = [], []
    all_gather = dist.all_gather

    def counted_gather(tensor_list, tensor, *args, **kwargs):
        if tensor.dtype == torch.long:  # positions; sequence gathers are float
            gathers.append(1)
        return all_gather(tensor_list, tensor, *args, **kwargs)

    real_forward = ud.ulysses_deepseek_v4_forward

    def counted_forward(*args, **kwargs):
        calls.append(1)
        return real_forward(*args, **kwargs)

    dist.all_gather = counted_gather
    ud.ulysses_deepseek_v4_forward = counted_forward
    try:
        ids = torch.randint(0, VOCAB, (2, 16), generator = torch.Generator().manual_seed(1))
        count = F.pad(ids, (0, 1), value = -100)[:, 1:].ne(-100).sum()
        layers = model.config.num_hidden_layers
        # Two steps: the second step's new position tensor must gather again.
        for _ in range(2):
            loss, grads, n_gathers, n_calls = _step(model, manager, ids, count, gathers, calls)
            # Forward plus checkpoint recompute of every layer: the Ulysses path provably ran.
            assert n_calls == 2 * layers, n_calls
            assert n_gathers == 0, n_gathers

        full_position_ids = ud._full_position_ids
        ud._full_position_ids = _gather_every_layer
        try:
            ref_loss, ref_grads, ref_gathers, _ = _step(model, manager, ids, count, gathers, calls)
        finally:
            ud._full_position_ids = full_position_ids
        assert ref_gathers == 2 * compressors, ref_gathers
        assert torch.equal(loss, ref_loss), (loss, ref_loss)
        assert grads.keys() == ref_grads.keys()
        for name, grad in grads.items():
            assert torch.equal(grad, ref_grads[name]), name
    finally:
        dist.all_gather = all_gather
        ud.ulysses_deepseek_v4_forward = real_forward


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_deepseek_v4_positions_need_no_collective(use_reentrant):
    _spawn(_worker, 2, use_reentrant)


def _divergence_worker(rank, world):
    # Rank 1 writes its position shard in place between two compressor layers (any data-dependent
    # in-place edit), rank 0 does not. A per-forward cache keyed on the tensor's version would hit on
    # rank 0 and miss on rank 1: rank 1 alone all-gathers and the ranks deadlock or mismatch.
    manager = _manager(world, rank)
    gathers = []
    all_gather = dist.all_gather

    def counted_gather(*args, **kwargs):
        gathers.append(1)
        return all_gather(*args, **kwargs)

    dist.all_gather = counted_gather
    try:
        inputs = {"input_ids": torch.zeros((2, 8), dtype = torch.long)}
        manager._prepare_inputs(inputs)
        with manager.apply(inputs):
            pos = inputs["position_ids"]
            first = ud._full_position_ids(pos, None)
            if rank == 1:
                pos.add_(0)
            second = ud._full_position_ids(pos.detach(), None)
            third = ud._full_position_ids(pos, None)
        expected = torch.arange(8).expand(2, -1)
        for full in (first, second, third):
            assert torch.equal(full, expected)
    finally:
        dist.all_gather = all_gather
    counts = [None] * world
    dist.all_gather_object(counts, len(gathers))
    assert counts == [0] * world, counts


def _spawn_with_timeout(fn, world, seconds):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs the fork start method")
    with tempfile.TemporaryDirectory() as d:
        mp.start_processes(
            _timeout_entry,
            args = (world, os.path.join(d, "init"), fn, seconds),
            nprocs = world,
            join = True,
            start_method = "fork",
        )


def _timeout_entry(rank, world, init_file, fn, seconds):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method = f"file://{init_file}",
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = seconds),
    )
    try:
        fn(rank, world)
    finally:
        dist.destroy_process_group()


def test_rank_local_position_writes_issue_no_divergent_collective():
    _spawn_with_timeout(_divergence_worker, 2, 20)


def _fallback_worker(rank, world):
    manager = _manager(world, rank)
    gathers = []
    all_gather = dist.all_gather

    def counted_gather(*args, **kwargs):
        gathers.append(1)
        return all_gather(*args, **kwargs)

    dist.all_gather = counted_gather
    try:
        local = torch.arange(rank * 4, rank * 4 + 4).unsqueeze(0)
        # Outside a step there is no manager: gather on every call, on every rank.
        for _ in range(2):
            assert torch.equal(ud._full_position_ids(local, None), torch.arange(8).unsqueeze(0))
        assert len(gathers) == 2
        inputs = {"input_ids": torch.zeros((1, 8), dtype = torch.long)}
        manager._prepare_inputs(inputs)
        with manager.apply(inputs):
            pos = inputs["position_ids"]
            assert torch.equal(ud._full_position_ids(pos, None), torch.arange(8).unsqueeze(0))
            assert len(gathers) == 2
            # A shape the step's positions do not cover (a sub-batch): gather, on every rank alike.
            sub = torch.cat([pos, pos + 100])
            want = torch.cat([torch.arange(8), torch.arange(8) + 100]).view(2, 8)
            assert torch.equal(ud._full_position_ids(sub, None), want)
            assert len(gathers) == 3
            with torch.inference_mode():
                assert torch.equal(ud._full_position_ids(pos, None), torch.arange(8).unsqueeze(0))
            assert len(gathers) == 3
        assert manager.full_position_ids is None
    finally:
        dist.all_gather = all_gather


def test_full_position_ids_fallback_gathers_on_every_rank():
    _spawn(_fallback_worker, 2)
