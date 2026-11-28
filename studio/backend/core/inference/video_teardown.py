# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Teardown of a replaced video pipeline that frees its memory even when the pipeline object itself survives.

A load that replaces the resident model tears the old state down inside the new load. Measured on MiniMax-H3 (L4,
52 GB host): the old pipeline object was still referenced after that teardown, so its pinned denoiser slabs and host
text-encoder copy (about 35 GB) stayed allocated while the new load sized itself. The new load then saw 0 MiB of
pinnable host RAM and kept every denoiser block on pageable host copies.

The pipeline object, and the components manager it owns, hold the only outside references to its components (the
rest are cycles among the components and their hooks), so dropping those lets the next collection free them. The
modules themselves are left alone: some submodule instances are shared with modules a later load builds."""

from __future__ import annotations

from typing import Any


def _is_components_manager(obj: Any) -> bool:
    return type(obj).__name__ == "ComponentsManager"


def _holds_module(value: Any) -> bool:
    import torch

    if isinstance(value, torch.nn.Module) or _is_components_manager(value):
        return True
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple, set)):
        return any(isinstance(v, torch.nn.Module) or _is_components_manager(v) for v in value)
    return False


def _drop_module_attributes(obj: Any) -> int:
    """Replace each attribute of ``obj`` that holds a module: containers by a fresh empty one (a shared container is
    never mutated), anything else by None. The attribute names stay, so a stray ``getattr`` still answers."""
    try:
        attributes = vars(obj)
    except TypeError:
        return 0
    dropped = 0
    for key, value in list(attributes.items()):
        try:
            if not _holds_module(value):
                continue
            if _is_components_manager(value):
                _drop_module_attributes(value)
            replacement = None
            if isinstance(value, (dict, list, set, tuple)):
                try:
                    replacement = type(value)()
                except Exception:  # noqa: BLE001 - a container type that needs arguments
                    replacement = None
            attributes[key] = replacement
            dropped += 1
        except Exception:  # noqa: BLE001 - teardown is best effort
            continue
    return dropped


def release_pipe_references(pipe: Any) -> int:
    """Drop every component reference a torn-down ``pipe`` holds, and those of a components manager it owns.
    Returns how many pipe attributes were dropped; never raises. Only for a pipeline nothing will run again."""
    if pipe is None:
        return 0
    return _drop_module_attributes(pipe)
