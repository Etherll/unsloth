# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 dev fused guidance: each fused row equals the same row of a call with the pass's global flag, bit for
bit; only batch shape differs from the separate calls. Tiny real Diffusers LTX-2 on CPU; skips without LTX-2."""

from __future__ import annotations

import pytest

from core.inference import video_ltx25_fused_guidance as fg

_STG = [1]


def _transformer(perturbed: bool = True):
    try:
        import torch

        from core.inference.ltx2_import_compat import ensure_ltx2_pipelines_importable

        ensure_ltx2_pipelines_importable()
        from diffusers import LTX2VideoTransformer3DModel
    except Exception as exc:  # noqa: BLE001 -- an old or broken Diffusers
        pytest.skip(f"no Diffusers LTX-2 transformer here ({type(exc).__name__})")
    import inspect

    if "perturbed_attn" not in inspect.signature(LTX2VideoTransformer3DModel.__init__).parameters:
        pytest.skip("this Diffusers' LTX-2 DiT has no STG (perturbed_attn)")
    torch.manual_seed(0)
    return LTX2VideoTransformer3DModel(
        in_channels = 4,
        out_channels = 4,
        patch_size = 1,
        patch_size_t = 1,
        num_attention_heads = 2,
        attention_head_dim = 8,
        cross_attention_dim = 16,
        audio_in_channels = 4,
        audio_out_channels = 4,
        audio_num_attention_heads = 2,
        audio_attention_head_dim = 4,
        audio_cross_attention_dim = 8,
        num_layers = 3,
        qk_norm = "rms_norm_across_heads",
        caption_channels = 32,
        rope_double_precision = False,
        rope_type = "split",
        perturbed_attn = perturbed,
    ).eval()


class _Runner:
    """The three guidance flags the pipeline exposes, as properties (the router reads them off the class)."""

    def __init__(
        self,
        transformer,
        cfg = True,
        stg = True,
        mod = True,
    ):
        self.transformer = transformer
        self._flags = (cfg, stg, mod)

    @property
    def do_classifier_free_guidance(self):
        return self._flags[0]

    @property
    def do_spatio_temporal_guidance(self):
        return self._flags[1]

    @property
    def do_modality_isolation_guidance(self):
        return self._flags[2]

    def __call__(self, spatio_temporal_guidance_blocks = (1,)):
        raise AssertionError("not called")


def _main_kwargs(
    transformer,
    rows = 2,
    seed = 0,
):
    """A CFG call shaped as ``LTX2Pipeline`` makes it (latents and coords duplicated, prompts uncond then cond)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    frames, height, width, audio_frames, tokens = 2, 2, 2, 3, 5
    n = rows // 2
    latents = torch.randn(n, frames * height * width, 4, generator = g)
    audio = torch.randn(n, audio_frames, 4, generator = g)
    embeds = torch.randn(rows, tokens, 32, generator = g)
    audio_embeds = torch.randn(rows, tokens, 32, generator = g)
    mask = torch.ones(rows, tokens, dtype = torch.long)
    mask[0, :2] = 0  # a shorter negative prompt
    video_coords = transformer.rope.prepare_video_coords(
        n, frames, height, width, latents.device, fps = 24.0
    )
    audio_coords = transformer.audio_rope.prepare_audio_coords(n, audio_frames, latents.device)
    t = torch.tensor(0.6)
    return (
        dict(
            hidden_states = torch.cat([latents] * 2),
            audio_hidden_states = torch.cat([audio] * 2),
            encoder_hidden_states = embeds,
            audio_encoder_hidden_states = audio_embeds,
            timestep = t.expand(rows),
            sigma = t.expand(rows),
            encoder_attention_mask = mask,
            audio_encoder_attention_mask = mask,
            num_frames = frames,
            height = height,
            width = width,
            fps = 24.0,
            audio_num_frames = audio_frames,
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1)),
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1)),
            isolate_modalities = False,
            spatio_temporal_guidance_blocks = None,
            perturbation_mask = None,
            use_cross_timestep = True,
            attention_kwargs = None,
            return_dict = False,
        ),
        latents,
        audio,
    )


def _pass_kwargs(main, latents, audio, kind):
    """The STG / modality call the pipeline makes after the CFG call: cond rows, uncond-half coords and timestep."""
    out = dict(main)
    out.update(
        hidden_states = latents.clone(),
        audio_hidden_states = audio.clone(),
        encoder_hidden_states = main["encoder_hidden_states"].chunk(2)[1],
        audio_encoder_hidden_states = main["audio_encoder_hidden_states"].chunk(2)[1],
        encoder_attention_mask = main["encoder_attention_mask"].chunk(2)[1],
        audio_encoder_attention_mask = main["audio_encoder_attention_mask"].chunk(2)[1],
        timestep = main["timestep"].chunk(2)[0],
        sigma = main["timestep"].chunk(2)[0],
        video_coords = main["video_coords"].chunk(2)[0],
        audio_coords = main["audio_coords"].chunk(2)[0],
        isolate_modalities = kind == "modality",
        spatio_temporal_guidance_blocks = list(_STG) if kind == "stg" else None,
    )
    return out


def _batched(main):
    """The fused call's own inputs: the CFG rows, then a cond row for STG and one for modality isolation."""
    import torch

    out = dict(main)
    for key, value in main.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == 2:
            part = value[0:1] if key in fg._FIRST_HALF else value[1:2]
            out[key] = torch.cat([value, part, part])
    return out


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.setenv(fg.FUSED_GUIDANCE_ENV, "1")
    fg.reset_stats()


def test_off_installs_nothing(monkeypatch):
    monkeypatch.delenv(fg.FUSED_GUIDANCE_ENV)
    t = _transformer()
    cls = type(t)
    procs = [type(a.processor) for a in fg._cross_modal_attentions(t)]
    with fg.fused_guidance(_Runner(t), {}) as state:
        assert state is None
        assert type(t) is cls
    assert [type(a.processor) for a in fg._cross_modal_attentions(t)] == procs
    assert fg.FUSED_STATS["renders"] == 0


def test_each_fused_row_equals_its_pass_at_the_same_shape():
    import torch

    t = _transformer()
    main, latents, audio = _main_kwargs(t)
    big = _batched(main)
    with torch.no_grad():
        with fg.fused_guidance(_Runner(t), {"spatio_temporal_guidance_blocks": _STG}) as state:
            video, audio_out = t(**main)
            stg = state.pending["stg"].output
            mod = state.pending["modality"].output
        assert type(t).__name__ == "LTX2VideoTransformer3DModel" and not getattr(
            type(t), "_unsloth_fused_guidance_cls", False
        )
        # The same batched inputs, each pass's global flag: the fused rows must match bit for bit.
        ref_main = t(**big)
        ref_stg = t(**dict(big, spatio_temporal_guidance_blocks = _STG))
        ref_mod = t(**dict(big, isolate_modalities = True))
    assert fg.FUSED_STATS["fused_calls"] == 1 and fg.FUSED_STATS["fused_rows"] == 4
    assert torch.equal(video, ref_main[0][:2]) and torch.equal(audio_out, ref_main[1][:2])
    assert torch.equal(stg[0], ref_stg[0][2:3]) and torch.equal(stg[1], ref_stg[1][2:3])
    assert torch.equal(mod[0], ref_mod[0][3:4]) and torch.equal(mod[1], ref_mod[1][3:4])
    # Each pass really changes its row (a no-op mask would pass the checks above vacuously).
    assert not torch.equal(stg[0], ref_main[0][2:3])
    assert not torch.equal(mod[0], ref_main[0][3:4]) and not torch.equal(mod[1], ref_main[1][3:4])


def test_later_calls_are_served_and_match_the_separate_passes():
    import torch

    t = _transformer()
    main, latents, audio = _main_kwargs(t)
    blocks: list = []
    t.transformer_blocks[0].register_forward_pre_hook(
        lambda m, a, k: blocks.append(int(k["hidden_states"].shape[0])), with_kwargs = True
    )
    with torch.no_grad():
        plain_main = t(**main)
        plain_stg = t(**_pass_kwargs(main, latents, audio, "stg"))
        plain_mod = t(**_pass_kwargs(main, latents, audio, "modality"))
        assert blocks == [2, 1, 1]
        blocks.clear()
        with fg.fused_guidance(_Runner(t), {"spatio_temporal_guidance_blocks": _STG}):
            got_main = t(**main)
            got_stg = t(**_pass_kwargs(main, latents, audio, "stg"))
            got_mod = t(**_pass_kwargs(main, latents, audio, "modality"))
    assert blocks == [4]
    assert fg.FUSED_STATS["served_stg"] == 1 and fg.FUSED_STATS["served_modality"] == 1
    assert fg.FUSED_STATS["misses"] == 0
    for got, ref in ((got_main, plain_main), (got_stg, plain_stg), (got_mod, plain_mod)):
        for g, r in zip(got, ref):
            assert g.shape == r.shape
            torch.testing.assert_close(g, r, rtol = 1e-5, atol = 1e-5)


def test_a_call_that_differs_runs_plain():
    import torch

    t = _transformer()
    main, latents, audio = _main_kwargs(t)
    blocks: list = []
    t.transformer_blocks[0].register_forward_pre_hook(
        lambda m, a, k: blocks.append(int(k["hidden_states"].shape[0])), with_kwargs = True
    )
    with torch.no_grad():
        with fg.fused_guidance(_Runner(t), {"spatio_temporal_guidance_blocks": _STG}):
            t(**main)
            moved = _pass_kwargs(main, latents + 1.0, audio, "stg")
            got = t(**moved)
            other_blocks = dict(
                _pass_kwargs(main, latents, audio, "stg"), spatio_temporal_guidance_blocks = [0]
            )
            t(**other_blocks)
        ref = t(**moved)
    assert blocks == [4, 1, 1, 1]
    assert fg.FUSED_STATS["misses"] == 1 and fg.FUSED_STATS["served_stg"] == 0
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_only_the_passes_the_pipeline_runs_are_fused():
    import torch

    t = _transformer()
    main, latents, audio = _main_kwargs(t)
    with torch.no_grad():
        with fg.fused_guidance(
            _Runner(t, stg = False), {"spatio_temporal_guidance_blocks": _STG}
        ) as state:
            t(**main)
            assert set(state.pending) == {"modality"}
        with fg.fused_guidance(
            _Runner(t, mod = False), {"spatio_temporal_guidance_blocks": _STG}
        ) as state:
            t(**main)
            assert set(state.pending) == {"stg"}
        with fg.fused_guidance(_Runner(t, stg = False, mod = False), {}) as state:
            t(**main)
            assert state.pending == {}
    assert fg.FUSED_STATS["fused_rows"] == 3 + 3


def test_no_cfg_rows_fuse_too():
    """CFG 1 with STG and modality on: the cond call (one row) carries both passes."""
    import torch

    t = _transformer()
    main, latents, audio = _main_kwargs(t)
    one = _pass_kwargs(main, latents, audio, "main")
    one.update(isolate_modalities = False, spatio_temporal_guidance_blocks = None)
    with torch.no_grad():
        ref = [
            t(**one),
            t(**_pass_kwargs(main, latents, audio, "stg")),
            t(**_pass_kwargs(main, latents, audio, "modality")),
        ]
        with fg.fused_guidance(_Runner(t, cfg = False), {"spatio_temporal_guidance_blocks": _STG}):
            got = [
                t(**one),
                t(**_pass_kwargs(main, latents, audio, "stg")),
                t(**_pass_kwargs(main, latents, audio, "modality")),
            ]
    assert fg.FUSED_STATS["fused_rows"] == 3 and fg.FUSED_STATS["served_modality"] == 1
    for g, r in zip(got, ref):
        torch.testing.assert_close(g[0], r[0], rtol = 1e-5, atol = 1e-5)
        torch.testing.assert_close(g[1], r[1], rtol = 1e-5, atol = 1e-5)


def test_processor_reset_is_rewrapped_and_state_kept():
    t = _transformer()
    attn = fg._cross_modal_attentions(t)[0]
    attn.processor._attention_backend = "native"
    procs = fg.ensure_row_keep_processors(t)
    assert all(getattr(type(p), "_unsloth_row_keep_cls", False) for p in procs)
    assert procs[0]._attention_backend == "native"
    assert type(procs[0]).__name__ == "LTX2PerturbedAttnProcessor"
    stock = type(procs[1]).__mro__[1]()
    fg._cross_modal_attentions(t)[1].processor = stock
    again = fg.ensure_row_keep_processors(t)
    assert (
        again[0] is procs[0]
        and again[1] is not stock
        and getattr(type(again[1]), "_unsloth_row_keep_cls", False)
    )


def test_declines_a_step_cache_and_other_denoisers(monkeypatch):
    t = _transformer()
    assert fg.decline_reason(_Runner(t), object()) is not None
    monkeypatch.setattr(type(t), "is_cache_enabled", property(lambda self: True), raising = False)
    with fg.fused_guidance(_Runner(t), {}) as state:
        assert state is None
    assert "step cache" in fg.FUSED_STATS["declined"]


def test_dev_render_through_the_pipeline_one_call_per_step(monkeypatch):
    """The real ``LTX2Pipeline`` dev loop through Studio's render: 3 DiT calls per step become 1, same output."""
    import numpy as np
    import torch

    from . import test_video_batch_ltx25 as tiny
    from core.inference import video_ltx25

    pipe = tiny._tiny_ltx25([])
    if "stg_scale" not in __import__("inspect").signature(pipe.__call__).parameters:
        pytest.skip("this Diffusers' LTX2Pipeline has no STG / modality guidance")
    pipe.transformer = _transformer()
    video_ltx25.prepare_scheduler(pipe, dev = True)
    steps = 3
    blocks: list = []
    pipe.transformer.transformer_blocks[0].register_forward_pre_hook(
        lambda m, a, k: blocks.append(int(k["hidden_states"].shape[0])), with_kwargs = True
    )

    def run():
        kwargs = dict(
            prompt = "a red fox in the snow",
            negative_prompt = "blurry",
            width = 64,
            height = 64,
            num_frames = 9,
            num_inference_steps = steps,
            guidance_scale = 3.0,
            spatio_temporal_guidance_blocks = list(_STG),
            generator = torch.Generator().manual_seed(5),
            output_type = "np",
        )
        with torch.no_grad():
            out = video_ltx25.render(pipe, kwargs, dev = True)
        return np.asarray(out.frames), np.asarray(
            out.audio[0] if isinstance(out.audio, (list, tuple)) else out.audio
        )

    monkeypatch.delenv(fg.FUSED_GUIDANCE_ENV)
    frames_off, audio_off = run()
    assert blocks == [2, 1, 1] * steps
    blocks.clear()
    monkeypatch.setenv(fg.FUSED_GUIDANCE_ENV, "1")
    frames_on, audio_on = run()
    assert blocks == [4] * steps
    assert fg.FUSED_STATS["served_stg"] == steps and fg.FUSED_STATS["served_modality"] == steps
    assert fg.FUSED_STATS["misses"] == 0 and fg.FUSED_STATS["fallbacks"] == 0
    assert type(pipe.transformer).__name__ == "LTX2VideoTransformer3DModel"
    assert not getattr(type(pipe.transformer), "_unsloth_fused_guidance_cls", False)
    np.testing.assert_allclose(frames_on, frames_off, rtol = 0, atol = 1e-4)
    np.testing.assert_allclose(audio_on, audio_off, rtol = 0, atol = 1e-4)
