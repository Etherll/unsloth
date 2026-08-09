# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from pydantic import ValidationError

from core.training.chat_template import (
    add_mlx_chat_template_config,
    apply_chat_template_override,
)
from core.training.training import _build_training_worker_config
from models.training import TrainingStartRequest
from picker.schemas import MAX_CHAT_TEMPLATE_BYTES


TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


def request(**overrides):
    values = {
        "model_name": "Qwen/Qwen3.5-0.8B-Base",
        "training_type": "LoRA/QLoRA",
        "format_type": "chatml",
    }
    values.update(overrides)
    return TrainingStartRequest(**values)


def test_request_preserves_valid_template_and_normalizes_blank():
    assert request(chat_template = TEMPLATE).chat_template == TEMPLATE
    assert request(chat_template = "  \n").chat_template is None


def test_request_rejects_invalid_or_oversized_template():
    with pytest.raises(ValidationError, match = "chat_template is invalid"):
        request(chat_template = "{% for message in messages %}")
    with pytest.raises(ValidationError, match = "byte limit"):
        request(chat_template = "x" * (MAX_CHAT_TEMPLATE_BYTES + 1))


def test_worker_config_only_forwards_template_for_conversational_sft():
    base = request(chat_template = TEMPLATE).model_dump()
    assert _build_training_worker_config(base)["chat_template"] == TEMPLATE

    for overrides in (
        {"training_type": "Continued Pretraining"},
        {"format_type": "raw"},
        {"is_embedding": True},
    ):
        values = {**base, **overrides}
        assert _build_training_worker_config(values)["chat_template"] is None


def test_tokenizer_override_and_mlx_forwarding_are_explicit():
    class Tokenizer:
        chat_template = None

    tokenizer = Tokenizer()
    apply_chat_template_override(tokenizer, TEMPLATE)
    assert tokenizer.chat_template == TEMPLATE

    mlx_config = {}
    add_mlx_chat_template_config(mlx_config, {"chat_template": object()}, TEMPLATE)
    assert mlx_config["chat_template"] == TEMPLATE

    with pytest.raises(RuntimeError, match = "does not support"):
        add_mlx_chat_template_config({}, {}, TEMPLATE)
