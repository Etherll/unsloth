#!/usr/bin/env python3
"""Small hosted-runner benchmark for PR 7101 sandbox overhead."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "studio" / "backend"))

from core.inference import sandbox, tools


def distribution_ms(call, iterations: int) -> dict[str, float]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000)
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[round(0.95 * (len(ordered) - 1))],
    }


def main() -> None:
    if not sandbox.sandbox_available():
        raise SystemExit("bubblewrap sandbox unavailable; refusing vacuous benchmark")

    with tempfile.TemporaryDirectory() as workdir:
        tools._workdirs["_pr7101_bench"] = workdir

        def run_bash(*, bypass: bool) -> None:
            output = tools._bash_exec(
                "true",
                session_id = "_pr7101_bench",
                timeout = 30,
                disable_sandbox = bypass,
            )
            if "Exit code" in output or "Error" in output:
                raise RuntimeError(output)

        run_bash(bypass = True)
        run_bash(bypass = False)
        bypass = distribution_ms(lambda: run_bash(bypass = True), 20)
        sandboxed = distribution_ms(lambda: run_bash(bypass = False), 20)
        read_paths = distribution_ms(sandbox._python_read_paths, 500)
        safe_env = distribution_ms(lambda: tools._build_safe_env(workdir), 2_000)
        external_read_paths = sandbox._python_read_paths()
        external_runtime_scan = distribution_ms(
            lambda: sandbox._assert_external_read_paths_have_no_special_nodes(
                workdir, external_read_paths
            ),
            20,
        )
        seccomp_filter = distribution_ms(
            lambda: os.close(sandbox._linux_socket_seccomp_fd()),
            100,
        )

        scan_root = Path(workdir) / "boundary-scan"
        for directory_index in range(10):
            directory = scan_root / f"d{directory_index}"
            directory.mkdir(parents = True)
            for file_index in range(50):
                (directory / f"f{file_index}.txt").write_text("sandbox benchmark\n")
        sandbox._assert_no_external_hardlinks(str(scan_root))
        boundary_scan = distribution_ms(
            lambda: sandbox._assert_no_external_hardlinks(str(scan_root)),
            30,
        )

    results = {
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "bash_bypass": bypass,
        "bash_sandboxed": sandboxed,
        "bash_sandbox_incremental_median_ms": sandboxed["median_ms"] - bypass["median_ms"],
        "bash_sandbox_ratio": sandboxed["median_ms"] / max(bypass["median_ms"], 0.001),
        "python_read_paths": read_paths,
        "build_safe_env": safe_env,
        "external_runtime_special_scan": external_runtime_scan,
        "seccomp_filter_build": seccomp_filter,
        "boundary_scan_500_files": boundary_scan,
    }
    print("BENCHMARK_JSON=" + json.dumps(results, sort_keys = True))
    for name in (
        "bash_bypass",
        "bash_sandboxed",
        "python_read_paths",
        "build_safe_env",
        "external_runtime_special_scan",
        "seccomp_filter_build",
        "boundary_scan_500_files",
    ):
        values = results[name]
        print(
            f"BENCH {name} mean_ms={values['mean_ms']:.4f} "
            f"median_ms={values['median_ms']:.4f} p95_ms={values['p95_ms']:.4f}"
        )
    print(
        "BENCH bash_sandbox_incremental "
        f"median_ms={results['bash_sandbox_incremental_median_ms']:.4f} "
        f"ratio={results['bash_sandbox_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
