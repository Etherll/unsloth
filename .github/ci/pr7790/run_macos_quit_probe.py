#!/usr/bin/env python3
"""Run PR 7790's instrumented native macOS application lifecycle matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def wait_for(path: Path, needle: str, timeout: float = 20) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and needle in path.read_text(errors="replace"):
            return True
        time.sleep(0.1)
    return False


def stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGKILL)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("base", "target"), required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    artifacts = Path(args.artifacts).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    cases = [
        {"name": "baseline-training-applescript", "state": "training", "trigger": "applescript", "exit": True},
    ] if args.mode == "base" else [
        {"name": "inactive-applescript", "state": "inactive", "trigger": "applescript", "exit": True,
         "has": ["applicationShouldTerminate NOW"]},
        {"name": "training-cancel", "state": "training", "trigger": "applescript", "response": "cancel", "exit": False,
         "has": ["applicationShouldTerminate LATER", "prompt training", "replyToApplicationShouldTerminate false"]},
        {"name": "training-confirm", "state": "training", "trigger": "applescript", "response": "confirm", "exit": True,
         "has": ["applicationShouldTerminate LATER", "prompt training", "replyToApplicationShouldTerminate true"]},
        {"name": "install-cancel", "state": "install", "trigger": "applescript", "response": "cancel", "exit": False,
         "has": ["applicationShouldTerminate LATER", "prompt install", "replyToApplicationShouldTerminate false"]},
        {"name": "install-confirm", "state": "install", "trigger": "applescript", "response": "confirm", "exit": True,
         "has": ["applicationShouldTerminate LATER", "prompt install", "replyToApplicationShouldTerminate true"]},
        {"name": "update-cancel", "state": "update", "trigger": "applescript", "response": "cancel", "exit": False,
         "has": ["applicationShouldTerminate LATER", "prompt update", "replyToApplicationShouldTerminate false"]},
        {"name": "update-confirm", "state": "update", "trigger": "applescript", "response": "confirm", "exit": True,
         "has": ["applicationShouldTerminate LATER", "prompt update", "replyToApplicationShouldTerminate true"]},
        {"name": "shell-update-cancel", "state": "shell-update", "trigger": "applescript", "response": "cancel", "exit": False,
         "has": ["applicationShouldTerminate LATER", "prompt shell-update", "replyToApplicationShouldTerminate false"]},
        {"name": "shell-update-confirm", "state": "shell-update", "trigger": "applescript", "response": "confirm", "exit": True,
         "has": ["applicationShouldTerminate LATER", "prompt shell-update", "replyToApplicationShouldTerminate true"]},
        {"name": "downloads-cancel", "state": "downloads", "trigger": "applescript", "response": "cancel", "exit": False,
         "has": ["applicationShouldTerminate LATER", "prompt downloads", "replyToApplicationShouldTerminate false"]},
        {"name": "downloads-confirm", "state": "downloads", "trigger": "applescript", "response": "confirm", "exit": True,
         "has": ["applicationShouldTerminate LATER", "prompt downloads", "replyToApplicationShouldTerminate true"]},
        {"name": "both-install-cancel", "state": "install,training", "trigger": "applescript", "response": "install=cancel,training=confirm", "exit": False,
         "has": ["prompt install", "replyToApplicationShouldTerminate false"], "lacks": ["prompt training"]},
        {"name": "both-training-cancel", "state": "install,training", "trigger": "applescript", "response": "install=confirm,training=cancel", "exit": False,
         "has": ["prompt install", "prompt training", "replyToApplicationShouldTerminate false"]},
        {"name": "both-confirm", "state": "install,training", "trigger": "applescript", "response": "confirm", "exit": True,
         "has": ["prompt install", "prompt training", "replyToApplicationShouldTerminate true"]},
        {"name": "all-states-download-cancel", "state": "install,update,shell-update,training,downloads", "trigger": "applescript",
         "response": "install=confirm,update=confirm,shell-update=confirm,training=confirm,downloads=cancel", "exit": False,
         "has": ["prompt install", "prompt update", "prompt shell-update", "prompt training", "prompt downloads",
                 "replyToApplicationShouldTerminate false"]},
        {"name": "all-states-confirm", "state": "install,update,shell-update,training,downloads", "trigger": "applescript",
         "response": "confirm", "exit": True,
         "has": ["prompt install", "prompt update", "prompt shell-update", "prompt training", "prompt downloads",
                 "replyToApplicationShouldTerminate true"]},
        {"name": "duplicate-applescript-serialized", "state": "training", "trigger": "applescript-double", "response": "cancel", "delay": "1200", "exit": False,
         "has": ["applicationShouldTerminate LATER", "replyToApplicationShouldTerminate false"],
         "lacks": ["applicationShouldTerminate CANCEL duplicate"]},
        {"name": "duplicate-direct-guard", "state": "training", "trigger": "delegate-double", "response": "cancel", "delay": "1200", "exit": False,
         "has": ["applicationShouldTerminate LATER", "applicationShouldTerminate CANCEL duplicate", "direct duplicate results 2 0"]},
        {"name": "native-menu-cancel", "state": "training", "trigger": "native-menu", "response": "cancel", "exit": False,
         "has": ["native menu item performClick", "menu confirmation path", "prompt training"]},
        {"name": "native-menu-confirm", "state": "training", "trigger": "native-menu", "response": "confirm", "exit": True,
         "has": ["native menu item performClick", "menu confirmation path", "prompt training"]},
        {"name": "programmatic-exit", "state": "install,update,shell-update,training,downloads", "trigger": "programmatic", "exit": True,
         "lacks": ["prompt install", "prompt update", "prompt shell-update", "prompt training", "prompt downloads",
                   "applicationShouldTerminate entered"]},
        {"name": "real-dialog-training", "state": "training", "trigger": "applescript", "exit": False,
         "has": ["applicationShouldTerminate LATER", "prompt training"], "screenshot": True},
    ]

    for case in cases:
        name = str(case["name"])
        log = artifacts / f"{name}.events.log"
        stdout = artifacts / f"{name}.process.log"
        env = os.environ.copy()
        env.update({
            "UNSLOTH_QUIT_CI_LOG": str(log),
            "UNSLOTH_QUIT_CI_STATE": str(case["state"]),
            "RUST_BACKTRACE": "1",
        })
        trigger = str(case["trigger"])
        if not trigger.startswith("applescript"):
            env["UNSLOTH_QUIT_CI_TRIGGER"] = trigger
        if "response" in case:
            env["UNSLOTH_QUIT_CI_RESPONSE"] = str(case["response"])
        if "delay" in case:
            env["UNSLOTH_QUIT_CI_DELAY_MS"] = str(case["delay"])

        with stdout.open("wb") as output:
            proc = subprocess.Popen([str(binary)], env=env, stdout=output, stderr=subprocess.STDOUT)
        ready = wait_for(log, "ready state=", 30)
        if ready and trigger.startswith("applescript"):
            command = [
                "osascript",
                "-e",
                'tell application id "ai.unsloth.studio" to quit',
            ]
            apple_log = artifacts / f"{name}.applescript.log"
            with apple_log.open("wb") as output:
                requests = [subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)]
                if trigger == "applescript-double":
                    time.sleep(0.1)
                    requests.append(subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT))
                for request in requests:
                    try:
                        request.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        request.kill()
                        request.wait(timeout=5)
        time.sleep(3 if case.get("screenshot") else 2)
        running = proc.poll() is None
        events = log.read_text(errors="replace") if log.exists() else ""
        expected_exit = bool(case["exit"])
        ok = ready and (running != expected_exit)
        for needle in case.get("has", []):
            ok = ok and needle in events
        for needle in case.get("lacks", []):
            ok = ok and needle not in events
        if case.get("screenshot"):
            shot = artifacts / f"{name}.png"
            capture = subprocess.run(["screencapture", "-x", str(shot)], capture_output=True, text=True)
            ok = ok and capture.returncode == 0 and shot.exists()
        results.append({
            "name": name,
            "passed": ok,
            "ready": ready,
            "expected": "exit" if expected_exit else "remain running",
            "observed": "running" if running else f"exited ({proc.returncode})",
            "events": events.splitlines(),
        })
        stop(proc)
        print(("PASS" if ok else "FAIL"), name, results[-1]["observed"])

    summary = {"mode": args.mode, "binary": str(binary), "results": results}
    (artifacts / f"{args.mode}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if all(bool(result["passed"]) for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
