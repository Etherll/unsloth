# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests reset diffusion_int8_gemm's per-device ``_CUT_*`` state from the module, not from a list of names.

#20201 added five of those dicts. Every test file that cleared a hand-written list missed them, so a load in one
file left ``_CUT_MIX`` set for the next, and ``_run`` then counted a step call on another file's stand-in
operands (the TMA tests failed whenever they ran after test_diffusion_int8_cutlass_defer.py on one worker)."""

from __future__ import annotations

import ast
from pathlib import Path

_TESTS = Path(__file__).resolve().parent


def test_no_test_resets_a_hand_written_list_of_cut_dicts():
    stale = []
    for path in sorted(_TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                continue
            named = [
                e.attr
                for e in node.elts
                if isinstance(e, ast.Attribute)
                and isinstance(e.value, ast.Name)
                and e.value.id == "g8"
                and e.attr.startswith("_CUT_")
            ]
            if len(named) >= 2:
                stale.append(f"{path.name}:{node.lineno} {named}")
    assert not stale, (
        "reset every _CUT_* dict, e.g. [v for k, v in vars(g8).items() if k.startswith('_CUT_') and "
        f"isinstance(v, dict)], so state added later is cleared too: {stale}"
    )
