# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import sys

__all__ = ["apply_nccl_default_env", "init_default_process_group"]

# Communicators torch splits from the default one (the dims of a multi-dimensional DeviceMesh, as
# in dp x tp, pp x dp or ep_rep x ep layouts) reuse its channels and buffers instead of allocating
# their own: two 2-D meshes on 2 T4s drop from 1196 to 444 MiB per rank on torch 2.15. A 1-D mesh
# over the world reuses the default group, so nothing changes there. NCCL reads it at communicator
# creation, so it has to be in the environment before init_process_group. An explicit value,
# including NCCL_COMM_SPLIT_SHARE_RESOURCES=0, is kept.
_NCCL_DEFAULT_ENV = (("NCCL_COMM_SPLIT_SHARE_RESOURCES", "1"),)


def apply_nccl_default_env(environ = None):
    """Set Unsloth's NCCL defaults the user has not set. Returns the names that were set.

    Nothing is changed once the default process group exists: NCCL already read its
    configuration, and ProcessGroupNCCL threads read the environment concurrently.
    """
    environ = os.environ if environ is None else environ
    dist = sys.modules.get("torch.distributed")
    try:
        if dist is not None and dist.is_available() and dist.is_initialized():
            return []
    except Exception:
        return []
    changed = []
    for name, value in _NCCL_DEFAULT_ENV:
        if name not in environ:
            environ[name] = value
            changed.append(name)
    return changed


# torch 2.8 to 2.14 split a sub-group from the default communicator (the dims of a multi-dimensional DeviceMesh,
# torch 2.14's per-direction pipeline P2P groups) only when the default group is bound to a device. Unbound, each
# mesh dim gets a fresh communicator (NCCL_COMM_SPLIT_SHARE_RESOURCES above does nothing) and
# torch.distributed.split_group raises "No device associated with the default pg". torch 2.15 binds the device
# itself. Binding also makes NCCL initialize inside init_process_group instead of at the first collective.
# torch 2.6 / 2.7 are left as they were. UNSLOTH_PG_DEVICE_ID=0 turns it off.
_DEVICE_ID_TORCH = ((2, 8), (2, 15))


def _bind_device_id(torch):
    if os.environ.get("UNSLOTH_PG_DEVICE_ID", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except ValueError:
        return False
    return _DEVICE_ID_TORCH[0] <= version < _DEVICE_ID_TORCH[1]


def init_default_process_group(
    backend,
    device = None,
    **kwargs,
):
    """dist.init_process_group(backend, **kwargs), with NCCL's default group bound to this rank's GPU (`device`,
    else the current CUDA device) on the torch versions above."""
    import torch
    import torch.distributed as dist

    if backend == "nccl" and "device_id" not in kwargs and _bind_device_id(torch):
        if device is None:
            try:
                device = torch.device("cuda", torch.cuda.current_device())
            except Exception:  # no usable CUDA device: leave the group unbound
                device = None
        if device is not None and device.type == "cuda":
            kwargs["device_id"] = device
    dist.init_process_group(backend = backend, **kwargs)
