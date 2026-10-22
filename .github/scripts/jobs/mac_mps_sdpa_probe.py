"""Real-MPS check for Studio's SDPA capability probe (unslothai/unsloth#14488).

    python jobs/mac_mps_sdpa_probe.py --base-sha SHA [--torch 2.12.1,2.13.0,2.14.1] [--out probe.json]

Per torch version (own venv, fresh process per cell):
  * measured: torch.mps.driver_allocated_memory() growth of ONE SDPA call (B=1 H=8 N=4096 D=128 bf16,
    no mask); quadratic = growth >= half the bf16 score matrix. Also on the newest torch with
    PYTORCH_MPS_DISABLE_PREFILL_ATTENTION = "1" / "" / "0", and under sdpa_kernel(FLASH) only.
  * claimed: available_sdpa_kernels / sdpa_math_only / sdpa_subquadratic_confirmed from
    studio/backend/core/inference/diffusion_attention.py at HEAD (this checkout) and at --base-sha
    (only that file swapped in a copy of studio/backend).
A cell agrees when sdpa_math_only == measured quadratic. Exit 0 every head cell agrees, 1 otherwise,
3 MPS unavailable (not a pass). Prints `JOB_RESULT {json}` last.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ENV = "PYTORCH_MPS_DISABLE_PREFILL_ATTENTION"
B, H, N, D = 1, 8, 4096, 128
SCORE_MIB = B * H * N * N * 2 / 2**20

MEASURE = r"""
import json, sys, torch
out = {"torch": torch.__version__, "mps": bool(torch.backends.mps.is_available())}
if out["mps"]:
    import torch.nn.functional as F
    q, k, v = (torch.randn(%d, %d, %d, %d, device="mps", dtype=torch.bfloat16) for _ in range(3))
    torch.mps.synchronize()
    ctx = None
    if sys.argv[1] == "flash_only":
        from torch.nn.attention import sdpa_kernel, SDPBackend
        ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    before = torch.mps.driver_allocated_memory()
    if ctx is not None:
        with ctx:
            o = F.scaled_dot_product_attention(q, k, v)
    else:
        o = F.scaled_dot_product_attention(q, k, v)
    torch.mps.synchronize()
    out["growth_mib"] = (torch.mps.driver_allocated_memory() - before) / 2**20
print("MEASURE " + json.dumps(out))
""" % (B, H, N, D)

PROBE = r"""
import json, sys, types, torch
sys.path.insert(0, sys.argv[1])
from core.inference import diffusion_attention as att
t = types.SimpleNamespace(device="mps", torch_device="mps", dtype=torch.bfloat16)
print("PROBE " + json.dumps({
    "kernels": list(att.available_sdpa_kernels(t)),
    "math_only": att.sdpa_math_only(t),
    "subquadratic": att.sdpa_subquadratic_confirmed(t),
}))
"""


def _run(
    py,
    code,
    *args,
    env_extra = None,
):
    env = {k: v for k, v in os.environ.items() if k != ENV}
    env.update(env_extra or {})
    r = subprocess.run(
        [py, "-c", code, *args], capture_output = True, text = True, env = env, timeout = 900
    )
    for line in r.stdout.splitlines():
        if line.startswith(("MEASURE ", "PROBE ")):
            return json.loads(line.split(" ", 1)[1])
    return {"error": (r.stderr or r.stdout)[-1500:], "rc": r.returncode}


def _venv(root, version):
    d = root / f"venv_torch_{version}"
    py = d / "bin" / "python"
    if not py.exists():
        subprocess.run([sys.executable, "-m", "venv", str(d)], check = True)
        subprocess.run([str(py), "-m", "pip", "install", "-q", f"torch=={version}"], check = True)
    return str(py)


def _base_tree(root, sha):
    src = Path("studio/backend")
    dst = root / "base_backend"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore = shutil.ignore_patterns("tests", "__pycache__", "node_modules"))
    url = f"https://raw.githubusercontent.com/unslothai/unsloth/{sha}/studio/backend/core/inference/diffusion_attention.py"
    with urllib.request.urlopen(url, timeout = 60) as r:
        (dst / "core/inference/diffusion_attention.py").write_bytes(r.read())
    return str(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-sha", required = True)
    ap.add_argument("--torch", default = "2.12.1,2.13.0,2.14.1")
    ap.add_argument("--out", default = "mac_mps_sdpa_probe.json")
    ap.add_argument("--tiny", action = "store_true", help = "ignored (staging_ci)")
    a = ap.parse_args()
    root = Path(os.environ.get("RUNNER_TEMP") or "temp_mps_probe").resolve() / "mps_probe"
    root.mkdir(parents = True, exist_ok = True)
    trees = {"head": str(Path("studio/backend").resolve()), "base": _base_tree(root, a.base_sha)}
    versions = [v.strip() for v in a.torch.split(",") if v.strip()]
    res = {
        "job": "mac_mps_sdpa_probe",
        "machine": platform.machine(),
        "mac": platform.mac_ver()[0],
        "base_sha": a.base_sha,
        "score_matrix_mib": SCORE_MIB,
        "cells": [],
    }
    for ver in versions:
        py = _venv(root, ver)
        envs = [None]
        if ver == versions[-1]:
            envs += ["1", "", "0"]
        for env in envs:
            extra = None if env is None else {ENV: env}
            m = _run(py, MEASURE, "plain", env_extra = extra)
            if not m.get("mps", True) is True or "error" in m:
                res["cells"].append({"torch": ver, "env": env, "measure": m})
                if m.get("mps") is False:
                    res["skipped"] = "torch.backends.mps.is_available() is False on this runner"
                continue
            cell = {
                "torch": ver,
                "env": env,
                "measure": m,
                "quadratic": m["growth_mib"] >= SCORE_MIB / 2,
            }
            for arm, tree in trees.items():
                p = _run(py, PROBE, tree, env_extra = extra)
                cell[arm] = p
                cell[f"{arm}_agrees"] = ("math_only" in p) and p["math_only"] == cell["quadratic"]
            res["cells"].append(cell)
            print("CELL " + json.dumps(cell), flush = True)
        if ver == versions[-1]:
            f = _run(py, MEASURE, "flash_only")
            res["flash_only_on_newest"] = f
    head_ok = [c.get("head_agrees") for c in res["cells"] if "head_agrees" in c]
    res["passed"] = bool(head_ok) and all(head_ok)
    Path(a.out).write_text(json.dumps(res, indent = 1))
    print("JOB_RESULT " + json.dumps(res), flush = True)
    if res.get("skipped") and not head_ok:
        sys.exit(3)
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
