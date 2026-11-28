# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 / H3-Image LoRAs saved by musubi-tuner (kohya ``lora_unet_blocks_0_attn_qkv_proj``, every dot of the
original module path written as an underscore) read as the same adapter as the dotted original names."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from core.inference import video_lora  # noqa: E402

H3 = "minimax-h3"
# (original module path, in features, out features)
MODULES = (
    ("blocks.0.attn.qkv_proj", 8, 3 * 128),
    ("blocks.0.attn.out_proj", 128, 8),
    ("blocks.0.mlp.fc1", 8, 12),
    ("blocks.0.mlp.fc2", 6, 8),
    ("blocks.1.adaln_proj.linear", 8, 24),
    ("token_refiner.blocks.0.attn.qkv_proj", 8, 3 * 128),
    ("token_refiner.blocks.0.mlp.fc2", 6, 8),
)


def _state(lead, alpha = 2.0):
    g = torch.Generator().manual_seed(0)
    out = {}
    for module, in_f, out_f in MODULES:
        name = lead(module)
        out[f"{name}.lora_down.weight"] = torch.randn(4, in_f, generator = g)
        out[f"{name}.lora_up.weight"] = torch.randn(out_f, 4, generator = g)
        out[f"{name}.alpha"] = torch.tensor(alpha)
    return out


def _files(tmp_path):
    kohya = tmp_path / "kohya.safetensors"
    dotted = tmp_path / "dotted.safetensors"
    save_file(_state(lambda m: "lora_unet_" + m.replace(".", "_")), str(kohya))
    save_file(_state(lambda m: "diffusion_model." + m), str(dotted))
    return str(kohya), str(dotted)


def _same(got, want):
    assert sorted(got) == sorted(want)
    for module in want:
        assert len(got[module]) == len(want[module]), module
        for (a0, b0, s0), (a1, b1, s1) in zip(want[module], got[module]):
            assert torch.equal(a1, a0) and torch.equal(b1, b0) and s1 == s0, module


@pytest.mark.parametrize(
    "flat, dotted",
    [
        ("blocks_0_attn_qkv_proj", "blocks.0.attn.qkv_proj"),
        ("blocks_12_attn_out_proj", "blocks.12.attn.out_proj"),
        ("blocks_3_mlp_fc1", "blocks.3.mlp.fc1"),
        ("blocks_3_adaln_proj_linear", "blocks.3.adaln_proj.linear"),
        ("token_refiner_blocks_1_attn_qkv_proj", "token_refiner.blocks.1.attn.qkv_proj"),
        ("token_refiner_blocks_1_mlp_fc2", "token_refiner.blocks.1.mlp.fc2"),
        ("video_patch_proj", "video_patch_proj"),
        ("condition_proj", "condition_proj"),
        ("time_embedder_proj_in", "time_embedder.proj_in"),
        ("final_layer_adaln_proj_linear", "final_layer.adaln_proj.linear"),
        ("final_layer_video_out", "final_layer.video_out"),
    ],
)
def test_kohya_module_names(flat, dotted):
    assert video_lora.h3_kohya_module("lora_unet_" + flat) == dotted


def test_video_kohya_file_reads_as_the_dotted_one(tmp_path):
    kohya, dotted = _files(tmp_path)
    want = video_lora.read_deltas([(dotted, 0.7)], family = H3)
    got = video_lora.read_deltas([(kohya, 0.7)], family = H3)
    _same(got, want)
    assert "transformer_blocks.0.attn.to_q" in got


def test_video_kohya_names_stay_as_they_are_for_other_families(tmp_path):
    kohya, _ = _files(tmp_path)
    got = video_lora.read_deltas([(kohya, 1.0)], family = "wan")
    assert all(m.startswith("lora_unet_") for m in got)


def test_h3_image_kohya_file_reads_as_the_dotted_one(tmp_path):
    from core.inference import diffusion_minimax_h3_image as h3i

    kohya, dotted = _files(tmp_path)
    want = h3i.native_lora_deltas(h3i.read_lora_deltas([(dotted, 0.7)]), 1, 128)
    got = h3i.native_lora_deltas(h3i.read_lora_deltas([(kohya, 0.7)]), 1, 128)
    _same(got, want)
    assert "transformer_blocks.0.attn.to_q" in got


def test_an_unknown_kohya_name_is_left_for_the_loader_to_refuse():
    assert video_lora.h3_kohya_module("lora_unet_blocks_0_attn_norm") is None
    assert video_lora.h3_kohya_module("blocks.0.attn.qkv_proj") is None
