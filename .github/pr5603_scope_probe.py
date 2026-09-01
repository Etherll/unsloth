# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studio" / "backend"))

from core.inference.tools import _check_code_safety


CASES = (
    (
        "global_sensitive_path",
        "path='README.md'\ndef arm():\n global path\n path='/etc/shadow'\narm()\nopen(path)",
        True,
    ),
    (
        "nonlocal_sensitive_path",
        "def outer():\n path='README.md'\n def arm():\n  nonlocal path\n  path='/etc/shadow'\n arm()\n open(path)\nouter()",
        True,
    ),
    (
        "nested_global_visible_at_module",
        "path='README.md'\ndef outer():\n def arm():\n  global path\n  path='/etc/shadow'\n arm()\nouter()\nopen(path)",
        True,
    ),
    (
        "eval_alias_shadowed",
        "import builtins\nf=builtins.exec\ndef use(f, payload):\n f(payload)",
        False,
    ),
    (
        "requests_alias_shadowed",
        "import requests as r\ndef use(r):\n r.get('https://private.example')",
        False,
    ),
    (
        "shutil_alias_shadowed",
        "import shutil as sh\ndef use(sh):\n sh.copy('/etc/shadow','x')",
        False,
    ),
    (
        "os_alias_shadowed",
        "import os as o\ndef use(o):\n o.system('cat /etc/shadow')",
        False,
    ),
    (
        "pathlib_alias_shadowed",
        "import pathlib as pl\ndef use(pl):\n pl.Path('/etc/shadow').read_text()",
        False,
    ),
    (
        "async_alias_shadowed",
        "import requests as r\nasync def use(r):\n r.get('https://private.example')",
        False,
    ),
    (
        "lambda_alias_shadowed",
        "import requests as r\nuse=lambda r:r.get('https://private.example')",
        False,
    ),
    (
        "nested_global_preserves_outer_local",
        "path='README.md'\ndef outer(path):\n def arm():\n  global path\n  path='/etc/shadow'\n arm()\n open(path)\nouter('README.md')",
        False,
    ),
    ("real_eval_alias", "import builtins\nf=builtins.exec\nf(\"open('/etc/shadow')\")", True),
    ("real_requests_alias", "import requests as r\nr.get('https://private.example')", True),
    ("real_os_alias", "import os as o\no.system('cat /etc/shadow')", True),
)


def verify() -> None:
    for name, code, expected_blocked in CASES:
        blocked = _check_code_safety(code) is not None
        if blocked != expected_blocked:
            raise AssertionError(f"{name}: expected blocked={expected_blocked}, got {blocked}")


def main() -> int:
    verify()
    iterations = 250
    started = time.perf_counter()
    for _ in range(iterations):
        verify()
    elapsed = time.perf_counter() - started
    calls = iterations * len(CASES)
    print(f"PASS scope_matrix={len(CASES)} benchmark_calls={calls}")
    print(f"BENCH elapsed_seconds={elapsed:.6f} calls_per_second={calls / elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
