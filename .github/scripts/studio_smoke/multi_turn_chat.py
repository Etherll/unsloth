# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Four turns through both SDKs, twice, against a running Unsloth server.

The smoke checks transport and history plumbing without treating a tiny model's prose as
an API contract. Prompt-token usage must grow on every turn as the earlier user and
assistant messages are carried forward. Each request must also finish normally, and the
first-turn accounting must agree across two identical fresh conversations. Generated text
may differ or be empty when the model immediately chooses EOS; both are valid inference
results and do not mean that the server dropped history.

This lived inline in three workflows, and being three copies is what let one of them
stop checking. On 2026-05-22 an unrelated event-loop fix (#5669) relaxed the Linux copy
to print a warning instead of failing; the macOS and Windows copies, which are otherwise
byte-identical in logic, kept the assertion. Linux is the leg that runs on every pull
request, so the check was effectively off where it mattered most, for three months, with
nothing to notice it. One file cannot drift from itself.

Reads BASE_URL and TOKEN from the environment, which is the only thing that differs
between the three callers: each boots its server on its own port.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

SEED = 3407
MAX_TOKENS = 80

PROMPTS = [
    "What is 1+1?",
    "What did I ask before?",
    "What is the capital of France?",
    "Repeat the city name.",
]


@dataclass(frozen = True)
class TurnResult:
    text: str
    prompt_tokens: int
    stop_reason: str | None


def _server() -> tuple[str, str]:
    """Where to talk and what to send. The only thing that differs per caller: each
    workflow boots its server on its own port. Read here rather than at import, so the
    checking half of this file can be exercised without a server or the SDKs."""
    return os.environ["BASE_URL"], os.environ["TOKEN"]  # a JWT is accepted as Bearer


def run_openai() -> list[TurnResult]:
    from openai import OpenAI

    BASE, KEY = _server()
    client = OpenAI(base_url = f"{BASE}/v1", api_key = KEY)
    history: list[dict] = []
    replies = []
    for prompt in PROMPTS:
        history.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model = "default",
            messages = history,
            temperature = 0.0,
            max_tokens = MAX_TOKENS,
            seed = SEED,
            extra_body = {"enable_thinking": False},
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        prompt_tokens = int(resp.usage.prompt_tokens) if resp.usage is not None else 0
        replies.append(TurnResult(text, prompt_tokens, choice.finish_reason))
        history.append({"role": "assistant", "content": text})
    return replies


def run_anthropic() -> list[TurnResult]:
    from anthropic import Anthropic

    BASE, KEY = _server()
    # Two SDK quirks against Unsloth:
    #   1. base_url must NOT include /v1 -- the SDK appends /v1/messages itself, and a
    #      base_url that already has it hits /v1/v1/messages and 405s.
    #   2. The SDK sends x-api-key by default, but Unsloth's auth layer is HTTPBearer
    #      only, so Authorization has to be set through default_headers instead.
    client = Anthropic(
        base_url = BASE,
        api_key = "unused",
        default_headers = {"Authorization": f"Bearer {KEY}"},
    )
    history: list[dict] = []
    replies = []
    for prompt in PROMPTS:
        history.append({"role": "user", "content": prompt})
        msg = client.messages.create(
            model = "default",
            max_tokens = MAX_TOKENS,
            messages = history,
            temperature = 0.0,
            extra_body = {"seed": SEED, "enable_thinking": False},
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        replies.append(TurnResult(text, int(msg.usage.input_tokens), msg.stop_reason))
        history.append({"role": "assistant", "content": text})
    return replies


def check_multi_turn_contract(
    label: str, first: list[TurnResult], second: list[TurnResult]
) -> None:
    expected = len(PROMPTS)
    assert len(first) == expected, f"{label}: first run returned {len(first)}/{expected} turns"
    assert len(second) == expected, f"{label}: second run returned {len(second)}/{expected} turns"

    for run_name, replies in (("run1", first), ("run2", second)):
        previous_prompt_tokens = 0
        for i, reply in enumerate(replies, start = 1):
            print(
                f"[{label} {run_name} turn {i}] text={reply.text!r} "
                f"prompt_tokens={reply.prompt_tokens} stop={reply.stop_reason!r}"
            )
            assert isinstance(reply.text, str), f"{label}: turn {i} returned non-text content"
            assert reply.stop_reason, f"{label}: turn {i} has no completion stop reason"
            assert reply.prompt_tokens > previous_prompt_tokens, (
                f"{label}: {run_name} turn {i} prompt usage did not grow with history: "
                f"{reply.prompt_tokens} <= {previous_prompt_tokens}"
            )
            previous_prompt_tokens = reply.prompt_tokens

    assert first[0].prompt_tokens == second[0].prompt_tokens, (
        f"{label}: identical fresh first turns reported different prompt usage: "
        f"{first[0].prompt_tokens} != {second[0].prompt_tokens}"
    )
    print(f"[{label}] OK -- 4 completed turns, cumulative history accounted for twice")


def main() -> int:
    for label, runner in (("openai", run_openai), ("anthropic", run_anthropic)):
        check_multi_turn_contract(label, runner(), runner())
    return 0


if __name__ == "__main__":
    sys.exit(main())
