# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A video teardown frees the old pipeline's components even when the pipeline object outlives it (a load replacing
the resident model kept ~35 GB of pinned / host copies of MiniMax-H3 held through its own sizing on a 52 GB host)."""

from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")


class _Pipe:
    pass


class ComponentsManager:
    """Named like diffusers' manager: a registry holding the components by name."""

    def __init__(self, **components):
        self.components = dict(components)


def _hooked(module):
    # An offload hook's host copy, reachable only through the module, plus a hook cycle back to the module.
    module._offload_copy = torch.zeros(4096)
    module._hook = lambda: module
    return module


def _pipe():
    pipe = _Pipe()
    pipe.transformer = _hooked(
        torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 64))
    )
    pipe.vae = torch.nn.Linear(8, 8)
    pipe.text_encoder = _hooked(torch.nn.Linear(16, 16))
    pipe._components_manager = ComponentsManager(
        transformer = pipe.transformer, text_encoder = pipe.text_encoder
    )
    held = {
        "transformer": weakref.ref(pipe.transformer[0].weight),
        "offload_copy": weakref.ref(pipe.transformer._offload_copy),
        "vae": weakref.ref(pipe.vae.weight),
        "text_encoder": weakref.ref(pipe.text_encoder.weight),
    }
    return pipe, held


def _state(video_module, pipe):
    from core.inference.video_families import detect_video_family
    fam = detect_video_family("", override = "minimax-h3")
    return video_module._VideoLoadState(
        pipe = pipe,
        family = fam,
        repo_id = "MiniMaxAI/MiniMax-H3",
        base_repo = fam.base_repo,
        device = "cpu",
        dtype = "bfloat16",
        kind = "pipeline",
    )


def test_unload_frees_components_while_a_pipe_reference_survives():
    import core.inference.video as video_module

    pipe, held = _pipe()
    survivor = [pipe]
    backend = video_module.VideoBackend()
    backend._state = _state(video_module, pipe)
    del pipe
    backend.unload()
    gc.collect()
    assert survivor[0] is not None
    assert {name: ref() is None for name, ref in held.items()} == dict.fromkeys(held, True)


def test_a_module_shared_with_a_live_owner_keeps_its_state():
    from core.inference.video_teardown import release_pipe_references

    shared = torch.nn.GELU()
    live = torch.nn.Sequential(torch.nn.Linear(4, 4), shared)
    pipe = _Pipe()
    pipe.text_encoder = torch.nn.Sequential(torch.nn.Linear(4, 4), shared)
    assert release_pipe_references(pipe) == 1
    gc.collect()
    assert live[1] is shared and "_modules" in vars(shared)
    assert live(torch.ones(1, 4)).shape == (1, 4)


def test_release_drops_component_references_of_the_pipe_and_its_manager():
    from core.inference.video_teardown import release_pipe_references

    pipe, _ = _pipe()
    pipe.blocks = [torch.nn.Linear(2, 2)]
    pipe.name = "kept"
    manager = pipe._components_manager
    registry = manager.components
    assert release_pipe_references(pipe) == 5
    assert (pipe.transformer, pipe.vae, pipe.text_encoder, pipe._components_manager) == (None,) * 4
    assert pipe.blocks == [] and pipe.name == "kept"
    assert manager.components == {}
    # The manager's old registry object is replaced, never emptied in place.
    assert len(registry) == 2


def test_release_is_a_no_op_for_no_pipe_or_an_object_without_modules():
    from core.inference.video_teardown import release_pipe_references

    assert release_pipe_references(None) == 0
    assert release_pipe_references(object()) == 0
    slot = _Pipe()
    slot.server_slot = "stopped"
    assert release_pipe_references(slot) == 0
    assert vars(slot) == {"server_slot": "stopped"}


def test_teardown_releases_after_every_step_that_reads_the_pipe():
    import inspect
    import textwrap

    import core.inference.video as video_module

    src = textwrap.dedent(inspect.getsource(video_module.VideoBackend._teardown_state_locked))
    release = src.index("release_pipe_references(")
    for reader in (
        "stop_background_pins(",
        "_finish_cast_writes(",
        "uninstall_cfg_parallel(",
        "uninstall_static_step_skip(",
        "diffusion_prompt_cache.release(",
    ):
        assert src.index(reader) < release, reader
    assert release < src.index("del state")
