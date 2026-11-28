# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unset speed on a ROCm load that never compiles resolves `eager`, not `off` (no GPU)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import diffusion_rocm_tunableop as tunableop
from core.inference import diffusion_speed as speed

ROCM = SimpleNamespace(backend = "rocm", device = "cuda", ordinal = 0)
CUDA = SimpleNamespace(backend = "cuda", device = "cuda", ordinal = 0)


@pytest.fixture
def gfx1151(monkeypatch):
    monkeypatch.delenv(speed.ROCM_AUTO_EAGER_ENV, raising = False)
    monkeypatch.setattr(tunableop, "_arch", lambda device = None: "gfx1151")


def _resolve(
    speed_mode = None,
    effective = speed.SPEED_OFF,
    target = ROCM,
    deferred = False,
    quantized = False,
):
    return speed.unset_speed_on_rocm(
        speed_mode, effective, target, deferred = deferred, quantized = quantized
    )


def test_unset_speed_with_no_compile_on_gfx1151_runs_eager(gfx1151):
    assert _resolve() == speed.SPEED_EAGER


@pytest.mark.parametrize("asked", ["off", "eager", "default", "max"])
def test_an_explicit_speed_mode_is_never_changed(gfx1151, asked):
    effective = speed.normalize_speed_mode(asked)
    assert _resolve(speed_mode = asked, effective = effective) == effective


def test_a_deferred_compile_keeps_off_until_it_engages(gfx1151):
    assert _resolve(deferred = True) == speed.SPEED_OFF


def test_a_compile_already_promoted_is_kept(gfx1151):
    assert _resolve(effective = speed.SPEED_DEFAULT) == speed.SPEED_DEFAULT


def test_a_quantized_denoiser_keeps_off(gfx1151):
    assert _resolve(quantized = True) == speed.SPEED_OFF


def test_cuda_keeps_off(gfx1151):
    assert _resolve(target = CUDA) == speed.SPEED_OFF


def test_an_unmeasured_rocm_arch_keeps_off(monkeypatch):
    monkeypatch.delenv(speed.ROCM_AUTO_EAGER_ENV, raising = False)
    monkeypatch.setattr(tunableop, "_arch", lambda device = None: "gfx1100")
    assert _resolve() == speed.SPEED_OFF


@pytest.mark.parametrize("value", ["0", "off", "false", "no"])
def test_the_env_switch_keeps_off(gfx1151, monkeypatch, value):
    monkeypatch.setenv(speed.ROCM_AUTO_EAGER_ENV, value)
    assert _resolve() == speed.SPEED_OFF


def test_an_arch_probe_that_raises_keeps_off(monkeypatch):
    monkeypatch.delenv(speed.ROCM_AUTO_EAGER_ENV, raising = False)

    def boom(device = None):
        raise RuntimeError("no device")

    monkeypatch.setattr(tunableop, "_arch", boom)
    assert _resolve() == speed.SPEED_OFF


def test_the_dense_load_resolves_it_after_the_deferred_check_and_before_the_attention_pick():
    src = (Path(speed.__file__).resolve().parent / "diffusion.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    calls = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "unset_speed_on_rocm"
    ]
    assert len(calls) == 1
    deferred = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "speed_deferred" for t in n.targets)
    ]
    attention = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "select_attention_backend"
    ]
    line = calls[0]
    assert any(d < line for d in deferred)
    after = [a for a in attention if a > line]
    assert after and min(after) - line < 60
