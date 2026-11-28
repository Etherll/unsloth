# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two batching guards: a torchao fp8 / int8 linear with per-tensor activation scaling couples batched rows, and a
tensor-parallel load gets its own batching verdict key."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_batch_race as race  # noqa: E402
from core.inference import diffusion_cfg_batch as cfg_batch  # noqa: E402


def _torchao():
    quant = pytest.importorskip("torchao.quantization")
    for name in ("Float8Tensor", "Int8Tensor", "PerRow", "PerTensor"):
        if not hasattr(quant, name):
            pytest.skip(f"this torchao has no {name}")
    return quant


def _fp8_linear(act_granularity):
    q = _torchao()
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
        QuantizeTensorToFloat8Kwargs,
    )

    lin = torch.nn.Linear(32, 32, bias = False).to(torch.bfloat16)
    weight = q.Float8Tensor.from_hp(
        lin.weight.detach(),
        granularity = q.PerRow(),
        act_quant_kwargs = QuantizeTensorToFloat8Kwargs(granularity = act_granularity),
    )
    lin.weight = torch.nn.Parameter(weight, requires_grad = False)
    return lin


def _int8_linear(act_granularity):
    q = _torchao()
    lin = torch.nn.Sequential(torch.nn.Linear(32, 32, bias = False).to(torch.bfloat16))
    q.quantize_(lin, q.Int8DynamicActivationInt8WeightConfig(granularity = act_granularity))
    assert type(lin[0].weight).__name__ == "Int8Tensor"
    return lin[0]


def _denoiser(layer):
    net = torch.nn.Module()
    net.add_module("blocks", torch.nn.Sequential(torch.nn.Identity(), layer))
    return net


@pytest.mark.parametrize("build", [_fp8_linear, _int8_linear], ids = ["fp8", "int8"])
def test_per_tensor_activation_scaling_is_per_call(build):
    q = _torchao()
    found = cfg_batch.per_call_scaled(_denoiser(build(q.PerTensor())))
    assert found is not None and "PerTensor" in found and found.startswith("blocks.1.weight")


@pytest.mark.parametrize("build", [_fp8_linear, _int8_linear], ids = ["fp8", "int8"])
def test_per_row_activation_scaling_is_not(build):
    q = _torchao()
    assert cfg_batch.per_call_scaled(_denoiser(build(q.PerRow()))) is None


def test_a_plain_denoiser_is_not():
    assert cfg_batch.per_call_scaled(_denoiser(torch.nn.Linear(4, 4))) is None


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


def _state(tp):
    transformer = torch.nn.Linear(2, 2)
    if tp is not None:
        transformer._unsloth_tp2 = tp
    return types.SimpleNamespace(pipe = _Pipe(transformer))


def test_a_tensor_parallel_load_has_its_own_topology():
    one = race.topology(_state(None))
    tp = race.topology(_state({"devices": ["cuda:0", "cuda:1"], "comm": "nccl"}))
    assert one != tp
    assert "tp2:cuda:0,cuda:1" in tp
    assert race.topology(_state(None)) == one


def test_the_record_on_the_pipe_counts_too():
    state = _state(None)
    state.pipe._unsloth_tp2 = {"devices": ["cuda:1", "cuda:0"]}
    assert "tp2:cuda:1,cuda:0" in race.topology(state)
