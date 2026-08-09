# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Any, Mapping, MutableMapping, Optional


def apply_chat_template_override(tokenizer: Any, chat_template: Optional[str]) -> None:
    """Apply a validated training template to the tokenizer used for formatting."""
    if chat_template is None:
        return
    if tokenizer is None:
        raise ValueError("Cannot apply a chat template before the tokenizer is loaded.")
    tokenizer.chat_template = chat_template


def add_mlx_chat_template_config(
    config: MutableMapping[str, Any],
    supported_fields: Mapping[str, Any],
    chat_template: Optional[str],
) -> None:
    """Forward a requested template or fail instead of silently ignoring it."""
    if chat_template is None:
        return
    if "chat_template" not in supported_fields:
        raise RuntimeError(
            "This Unsloth Zoo version does not support custom MLX training chat templates. "
            "Update Unsloth Studio and try again."
        )
    config["chat_template"] = chat_template
