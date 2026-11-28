# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The default process group Unsloth creates itself (parallel_env's layout mesh, the UNSLOTH_EP_SIZE mesh, native
FSDP2's trainer setup) is bound to this rank's GPU on torch 2.8 to 2.14 (init_process_group(device_id=...)), so
torch splits sub-groups from it: DeviceMesh dims, and torch 2.14's per-direction pipeline P2P groups, which raise
without a bound device. CPU only: init_process_group is replaced by a recorder."""

import os
import sys

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))

from unsloth_models_loader import load_models_module  # noqa: E402


class _Stop(Exception):
    pass


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def init(*args, **kwargs):
        calls.append((args, kwargs))
        raise _Stop

    monkeypatch.delenv("UNSLOTH_PG_DEVICE_ID", raising = False)
    monkeypatch.setattr(dist, "init_process_group", init)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch, "__version__", "2.14.1+cu130")
    monkeypatch.setenv("LOCAL_RANK", "1")
    return calls


def _nccl_env():
    load_models_module("fsdp2_native")  # puts the unsloth stub package in place
    import unsloth.nccl_env as nccl_env
    return nccl_env


@pytest.mark.parametrize(
    "version,bound",
    [
        ("2.6.0", False),
        ("2.7.1+cu126", False),
        ("2.8.0", True),
        ("2.14.1+cu130", True),
        ("2.15.0", False),
    ],
)
def test_nccl_default_group_is_bound_on_torch_2_8_to_2_14(recorded, monkeypatch, version, bound):
    nccl_env = _nccl_env()
    monkeypatch.setattr(torch, "__version__", version)
    with pytest.raises(_Stop):
        nccl_env.init_default_process_group("nccl")
    args, kwargs = recorded[-1]
    assert kwargs["backend"] == "nccl"
    assert kwargs.get("device_id") == (torch.device("cuda", 1) if bound else None), kwargs


def test_gloo_explicit_device_and_kill_switch(recorded, monkeypatch):
    nccl_env = _nccl_env()
    with pytest.raises(_Stop):
        nccl_env.init_default_process_group("gloo", timeout = 5)
    assert recorded[-1][1] == {"backend": "gloo", "timeout": 5}
    with pytest.raises(_Stop):
        nccl_env.init_default_process_group("nccl", device = torch.device("cuda", 3))
    assert recorded[-1][1]["device_id"] == torch.device("cuda", 3)
    with pytest.raises(_Stop):
        nccl_env.init_default_process_group("nccl", device_id = torch.device("cuda", 0))
    assert recorded[-1][1]["device_id"] == torch.device("cuda", 0)
    monkeypatch.setenv("UNSLOTH_PG_DEVICE_ID", "0")
    with pytest.raises(_Stop):
        nccl_env.init_default_process_group("nccl")
    assert "device_id" not in recorded[-1][1]


def test_every_unsloth_created_group_is_bound(recorded, monkeypatch):
    # parallel_env's layout mesh (UNSLOTH_PP_SIZE / UNSLOTH_TP_SIZE)
    _nccl_env()
    import unsloth.parallel_env as parallel_env

    layout = parallel_env.ParallelLayout(tp = 1, dp = 1, world = 2, fsdp2 = False, pp = 2)
    with pytest.raises(_Stop):
        layout.mesh()
    assert recorded[-1][1] == {"backend": "nccl", "device_id": torch.device("cuda", 1)}
    # UNSLOTH_EP_SIZE
    ep_native = load_models_module("ep_native")
    monkeypatch.setenv("UNSLOTH_EP_SIZE", "2")
    monkeypatch.setattr(ep_native, "_ENV_MESH", [])
    with pytest.raises(_Stop):
        ep_native.ep_env_mesh()
    assert recorded[-1][1] == {"backend": "nccl", "device_id": torch.device("cuda", 1)}
    # native FSDP2's trainer setup: its device (cuda:LOCAL_RANK), the trainer's timeout kept
    fsdp2_native = load_models_module("fsdp2_native")
    with pytest.raises(_Stop):
        fsdp2_native._ensure_process_group(torch.device("cuda", 1), timeout = 9)
    assert recorded[-1][1] == {
        "backend": "nccl",
        "timeout": 9,
        "device_id": torch.device("cuda", 1),
    }
