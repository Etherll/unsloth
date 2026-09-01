from __future__ import annotations

import argparse
import platform
import statistics
import time

from core.inference.tools import _check_code_safety


ATTACK_PROBES = {
    "fresh-client": (
        "import requests\n"
        "requests.Session().get('http://169.254.169.254/latest')"
    ),
    "inline-request": (
        "import urllib.request\n"
        "urllib.request.urlopen(urllib.request.Request("
        "'http://169.254.169.254/latest'))"
    ),
    "reader-alias": (
        "from pandas import read_csv\n"
        "read_csv('/home/u/.aws/credentials')"
    ),
}

BENIGN_PROBES = {
    "fresh-client-trusted": (
        "import requests\n"
        "requests.Session().get('https://google.com/')"
    ),
    "inline-request-trusted": (
        "import urllib.request\n"
        "urllib.request.urlopen(urllib.request.Request('https://google.com/'))"
    ),
    "reader-alias-project-file": (
        "from pandas import read_csv\n"
        "read_csv('data.csv')"
    ),
}


def is_blocked(code: str) -> bool:
    return _check_code_safety(code) is not None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-attacks", choices=("allowed", "blocked"), required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    expected_attack_blocked = args.expect_attacks == "blocked"
    failures: list[str] = []

    print(
        f"ENV label={args.label} os={platform.platform()} "
        f"python={platform.python_version()}"
    )
    for name, code in ATTACK_PROBES.items():
        observed = is_blocked(code)
        passed = observed == expected_attack_blocked
        print(
            f"REPRO attack={name} expected={args.expect_attacks} "
            f"observed={'blocked' if observed else 'allowed'} "
            f"{'PASS' if passed else 'FAIL'}"
        )
        if not passed:
            failures.append(name)

    for name, code in BENIGN_PROBES.items():
        observed = is_blocked(code)
        passed = not observed
        print(
            f"REPRO benign={name} expected=allowed "
            f"observed={'blocked' if observed else 'allowed'} "
            f"{'PASS' if passed else 'FAIL'}"
        )
        if not passed:
            failures.append(name)

    corpus = [*ATTACK_PROBES.values(), *BENIGN_PROBES.values()]
    for code in corpus:
        is_blocked(code)

    samples_us: list[float] = []
    started = time.perf_counter()
    for _ in range(args.iterations):
        for code in corpus:
            sample_started = time.perf_counter()
            is_blocked(code)
            samples_us.append((time.perf_counter() - sample_started) * 1_000_000)
    elapsed_ms = (time.perf_counter() - started) * 1_000

    print(
        f"BENCH label={args.label} cases={len(corpus)} "
        f"iterations={args.iterations} calls={len(samples_us)} "
        f"total_ms={elapsed_ms:.3f} mean_us={statistics.fmean(samples_us):.3f} "
        f"p50_us={statistics.median(samples_us):.3f} "
        f"p95_us={percentile(samples_us, 0.95):.3f}"
    )

    if failures:
        print(f"FAIL mismatches={','.join(failures)}")
        return 1
    print(f"PASS label={args.label} assertions={len(ATTACK_PROBES) + len(BENIGN_PROBES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
