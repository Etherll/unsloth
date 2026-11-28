# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Idle media headroom already cached by the allocator is in NVML's used figure: don't subtract it twice."""

import os
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from core.inference import gpu_arbiter as arb
from core.inference import gpu_residency as gr

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_gpu_residency_coexistence import (  # noqa: E402
    _ArbiterResetMixin,
    _cuda,
    _need,
    _parent_ids,
    _util,
)

MIB = 1024 * 1024
TOTAL_GB = 24.0
WEIGHTS_GB = 12.0
HEADROOM = 6000  # the resident's priced generation headroom, MiB


def _allocator(
    reserved_mib,
    allocated_mib,
    graph_mib = 0,
    initialised = True,
):
    stack = ExitStack()
    stack.enter_context(patch.object(torch.cuda, "is_initialized", return_value = initialised))
    stack.enter_context(
        patch.object(torch.cuda, "memory_reserved", return_value = reserved_mib * MIB)
    )
    stack.enter_context(
        patch.object(torch.cuda, "memory_allocated", return_value = allocated_mib * MIB)
    )
    stack.enter_context(
        patch(
            "core.inference.diffusion_memory._graph_held_cache_bytes", return_value = graph_mib * MIB
        )
    )
    return stack


class TestIdleCacheCredit(_ArbiterResetMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._resident(
            arb.DIFFUSION,
            {0},
            _need({0: 12 * 1024 + HEADROOM}, weights = 12 * 1024, headroom = HEADROOM, kind = "image"),
        )
        self.alloc = WEIGHTS_GB * 1024

    def _free(
        self,
        cached_mib,
        graph_mib = 0,
        initialised = True,
    ):
        """free_mib for a 6 GiB chat load beside the idle image resident, NVML used = weights + cached pool."""
        used_gb = WEIGHTS_GB + cached_mib / 1024
        with (
            _cuda(),
            _parent_ids((0,)),
            _util((0, TOTAL_GB, used_gb)),
            _allocator(int(self.alloc + cached_mib), int(self.alloc), graph_mib, initialised),
        ):
            _, info = gr.can_load_alongside(arb.CHAT, gr.ChatNeed({0: 6144}))
            unallocated = gr.media_unallocated_mib()
        return info["free_mib"][0], unallocated[0]

    def test_first_render_cache_is_not_counted_twice(self):
        before = self._free(cached_mib = 0)
        after = self._free(cached_mib = HEADROOM)
        self.assertEqual(before[0], int(TOTAL_GB * 1024 - WEIGHTS_GB * 1024) - HEADROOM)
        self.assertEqual(after, (before[0], 0))
        self.assertEqual(before[1], HEADROOM)

    def test_credit_is_capped_at_the_idle_headroom(self):
        free, unallocated = self._free(cached_mib = HEADROOM + 3000)
        self.assertEqual(free, int(TOTAL_GB * 1024 - WEIGHTS_GB * 1024) - HEADROOM - 3000)
        self.assertEqual(unallocated, 0)

    def test_partial_cache_credits_only_what_is_cached(self):
        free, unallocated = self._free(cached_mib = 2000)
        self.assertEqual(free, int(TOTAL_GB * 1024 - WEIGHTS_GB * 1024) - 2000 - (HEADROOM - 2000))
        self.assertEqual(unallocated, HEADROOM - 2000)

    def test_cuda_graph_pool_is_not_credited(self):
        free, unallocated = self._free(cached_mib = HEADROOM, graph_mib = HEADROOM)
        self.assertEqual(free, int(TOTAL_GB * 1024 - WEIGHTS_GB * 1024) - 2 * HEADROOM)
        self.assertEqual(unallocated, HEADROOM)

    def test_uninitialised_cuda_credits_nothing(self):
        free, unallocated = self._free(cached_mib = HEADROOM, initialised = False)
        self.assertEqual(free, int(TOTAL_GB * 1024 - WEIGHTS_GB * 1024) - 2 * HEADROOM)
        self.assertEqual(unallocated, HEADROOM)

    def test_a_loading_resident_still_counts_in_full(self):
        with patch.object(gr, "_media_busy", lambda owner: (owner == arb.DIFFUSION, False)):
            free, unallocated = self._free(cached_mib = HEADROOM)
        need = 12 * 1024 + HEADROOM
        self.assertEqual(free, max(int(TOTAL_GB * 1024 - WEIGHTS_GB * 1024) - HEADROOM - need, 0))
        self.assertEqual(unallocated, need)


if __name__ == "__main__":
    unittest.main()
