# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The same smoke test on three operating systems has to be the same test.

The multi-turn chat check ran inline in studio-inference-smoke.yml,
studio-mac-inference-smoke.yml and studio-windows-inference-smoke.yml as three copies of
one script. On 2026-05-22 an unrelated event-loop fix (#5669) weakened only the Linux
copy. Nothing compared them, so for three months the leg that runs on every pull request
was checking a different contract from the two legs that run less often.

So the copies are gone, and these tests are about keeping them gone.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / ".github" / "scripts" / "studio_smoke" / "multi_turn_chat.py"
LEGS = (
    "studio-inference-smoke.yml",
    "studio-mac-ui-smoke.yml",
    "studio-windows-inference-smoke.yml",
)


def _workflow(name: str) -> str:
    return (REPO / ".github" / "workflows" / name).read_text(encoding = "utf-8")


@pytest.fixture(scope = "module")
def script():
    """The shared script, imported. It reads no environment and imports no SDK at module
    level precisely so this is possible."""
    assert SCRIPT.is_file(), f"{SCRIPT} is gone; the three legs have nothing to share"
    spec = importlib.util.spec_from_file_location("multi_turn_chat", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["multi_turn_chat"] = module
    spec.loader.exec_module(module)
    return module


def test_every_leg_runs_the_shared_script():
    missing = [name for name in LEGS if "studio_smoke/multi_turn_chat.py" not in _workflow(name)]
    assert not missing, (
        f"{missing} no longer run the shared multi-turn check. Three copies of it is how "
        f"one of them drifted without anyone noticing."
    )


def test_no_leg_has_grown_its_own_copy_back():
    """Reverting one leg to an inline block is the regression, and it looks additive."""
    offenders = [name for name in LEGS if "def run_anthropic" in _workflow(name)]
    assert not offenders, (
        f"{offenders} carry an inline copy of the multi-turn check again. Change "
        f"{SCRIPT.relative_to(REPO)} instead, so the other legs get the change too."
    )


def _turns(script, texts = ("a", "b", "c", "d"), tokens = (10, 20, 30, 40)):
    return [
        script.TurnResult(text, prompt_tokens, "stop")
        for text, prompt_tokens in zip(texts, tokens)
    ]


def test_model_wording_and_empty_eos_may_vary(script):
    """Tiny-model prose is not a server contract; completion metadata is."""
    first = _turns(script, ("2", "", "Paris", "Paris"))
    second = _turns(script, ("Two", "You asked 1+1", "", "The city was Paris."))
    script.check_multi_turn_contract("valid variation", first, second)


def test_every_run_must_return_all_four_turns(script):
    clean = _turns(script)
    with pytest.raises(AssertionError, match = "3/4 turns"):
        script.check_multi_turn_contract("truncated", clean[:-1], clean)


def test_each_turn_must_have_a_stop_reason(script):
    clean = _turns(script)
    broken = list(clean)
    broken[2] = script.TurnResult("c", 30, None)
    with pytest.raises(AssertionError, match = "no completion stop reason"):
        script.check_multi_turn_contract("unfinished", clean, broken)


def test_prompt_usage_must_grow_with_history_in_both_runs(script):
    clean = _turns(script)
    with pytest.raises(AssertionError, match = "did not grow with history"):
        script.check_multi_turn_contract(
            "dropped history", clean, _turns(script, tokens = (10, 20, 20, 40))
        )


def test_identical_fresh_turns_have_identical_prompt_usage(script):
    clean = _turns(script)
    with pytest.raises(AssertionError, match = "fresh first turns"):
        script.check_multi_turn_contract(
            "unstable accounting", clean, _turns(script, tokens = (11, 21, 31, 41))
        )


def test_the_script_needs_no_environment_to_import(script):
    """What lets every test above exist.

    Reading BASE_URL at module level, or importing the SDKs there, would make the
    checking half unreachable from a test and put it back where it was: only ever
    exercised by a full smoke run on three operating systems.
    """
    source = SCRIPT.read_text(encoding = "utf-8")
    head = source.split("def _server", 1)[0]
    for forbidden in ("os.environ[", "from openai", "from anthropic"):
        assert forbidden not in head, (
            f"{forbidden} moved to module level in {SCRIPT.name}, so importing it now "
            f"needs a running server or the SDKs installed, and these tests cannot run"
        )
