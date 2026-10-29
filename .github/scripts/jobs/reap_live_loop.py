"""Loop test_reap_recorded_pid_spares_live_server N times per arm, base vs head, psutil present vs absent.

Arms: {base tree (pre #15921), head tree (staged)} x {psutil installed, psutil uninstalled}, plus a
"product" variant that does NOT patch _pid_is_llama_server (what a real install would do).
Run from the staged repo root (staging --job step). Prints a counts table and writes --out JSON.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--tiny", action = "store_true")
p.add_argument("--out", default = "reap_live_loop.json")
p.add_argument("--n", type = int, default = 200)
p.add_argument(
    "--no-pip", action = "store_true", help = "local smoke: never touch the installed psutil"
)
p.add_argument("--base-sha", default = "6165c444eda9ec7a8a5b2b2b346998360b5c1ab2")
a = p.parse_args()

LOOP = """
import importlib.util, os, sys, subprocess
from pathlib import Path
from unittest.mock import patch
import pytest

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_reap_target", _here / "test_llama_cpp_wait_for_vram_settle.py")
m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m)
LB = m.LlamaCppBackend
_src = Path(sys.modules[LB.__module__].__file__).resolve()
assert str(_src).startswith(str(_here.parents[2])), ("arm imports a foreign tree", _src, _here)
print("ARM_SRC", _src, "psutil" in sys.modules or importlib.util.find_spec("psutil") is not None)

N = int(os.environ["REAP_LOOP_N"])

@pytest.mark.parametrize("i", range(N))
def test_loop_patched(tmp_path, i):
    m.test_reap_recorded_pid_spares_live_server(tmp_path)

@pytest.mark.parametrize("i", range(N))
def test_loop_product(tmp_path, i):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pidfile = tmp_path / "llama-server.pid"
    pidfile.write_text(str(proc.pid))
    try:
        with patch.object(LB, "_server_pidfile_path", staticmethod(lambda: pidfile)):
            n = LB._reap_recorded_pid()
        assert n == 0 and proc.poll() is None, "real (unpatched identity) reaper killed a live child"
    finally:
        proc.kill(); proc.wait(timeout=5)
"""

_FAILED = re.compile(r"^FAILED \S+::(test_loop_\w+)\[", re.M)


def run_arm(root):
    tdir = root / "studio" / "backend" / "tests"
    (tdir / "test_r10_reap_loop.py").write_text(LOOP, encoding = "utf-8")
    env = dict(os.environ, REAP_LOOP_N = str(a.n), PYTHONPATH = str(root / "studio" / "backend"))
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "-s",
            "--tb=line",
            str(tdir / "test_r10_reap_loop.py"),
        ],
        cwd = str(root),
        env = env,
        capture_output = True,
        text = True,
    )
    out = (r.stdout or "") + (r.stderr or "")
    src = re.findall(r"^ARM_SRC .*$", out, re.M)
    fails = {"test_loop_patched": 0, "test_loop_product": 0}
    for name in _FAILED.findall(out):
        fails[name] += 1
    summ = re.findall(
        r"^=*\s*((?:\d+ (?:passed|failed|skipped|errors?)(?:, )?)+) in [\d.]+s",
        (r.stdout or ""),
        re.M,
    )
    tail = summ[-1] if summ else "NO SUMMARY"
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed)", tail)}
    return {
        "rc": r.returncode,
        "src": src[:1],
        "fail_patched": fails["test_loop_patched"],
        "fail_product": fails["test_loop_product"],
        "n": a.n,
        "tail": tail,
        "ran": counts.get("passed", 0) + counts.get("failed", 0),
        "excerpt": [l for l in out.splitlines() if "Killed orphaned" in l or l.startswith("E ")][
            :4
        ],
    }


head = Path.cwd().resolve()
base = Path(tempfile.mkdtemp(prefix = "reap_base_")) / "base"
subprocess.run(["git", "init", "-q", str(base)], check = True)
subprocess.run(
    [
        "git",
        "-C",
        str(base),
        "fetch",
        "-q",
        "--depth=1",
        "https://github.com/unslothai/unsloth",
        a.base_sha,
    ],
    check = True,
)
subprocess.run(["git", "-C", str(base), "checkout", "-q", "FETCH_HEAD"], check = True)

pip = [sys.executable, "-m", "pip", "-q"]
res = {}
if not a.no_pip:
    subprocess.run(pip + ["install", "psutil"], check = True)
for arm, root in (("head", head), ("base", base)):
    res[f"{arm}_psutil"] = run_arm(root)
if not a.no_pip:
    subprocess.run(pip + ["uninstall", "-y", "psutil"], check = True)
    for arm, root in (("head", head), ("base", base)):
        res[f"{arm}_nopsutil"] = run_arm(root)
    subprocess.run(pip + ["install", "psutil"], check = True)

print(f"{'arm':16} {'n':>4} {'fail_patched':>12} {'fail_product':>12}  src")
for k, v in res.items():
    print(
        f"{k:16} {v['n']:>4} {v['fail_patched']:>12} {v['fail_product']:>12}  {v['src']}  | {v['tail']}"
    )
    for e in v["excerpt"]:
        print("    ", e)
Path(a.out).write_text(json.dumps(res, indent = 2), encoding = "utf-8")
# Every arm must have actually run all 2N tests (passed + failed == 2N).
ran_ok = all(v["ran"] == 2 * a.n for v in res.values())
sys.exit(0 if ran_ok else 1)
