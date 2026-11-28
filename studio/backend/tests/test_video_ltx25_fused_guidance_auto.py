# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 dev fused guidance under Auto: UNSLOTH_LTX25_FUSED_GUIDANCE 1 / 0 force it, unset fuses only with a passing
G2c verdict for the card class and a promoted speed race for the load and canvas. Without either the render installs
nothing (bit-identical to main), the 4-row peak is priced in the planner, an OOM falls back and stops Auto retrying,
and the video status reports the block."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_batch_quality as bq
from core.inference import video_ltx25_fused_guidance as fg

A100_40 = "NVIDIA A100-SXM4-40GB"
A100_80 = "NVIDIA A100-SXM4-80GB"
L4 = "NVIDIA L4"
CANVAS = dict(width = 768, height = 512, frames = 49, steps = 30, guidance = 4.0)


class _Store:
    """The probe-cache namespace as the module uses it (path, get, put), in memory."""

    def __init__(self, path):
        self.path = path
        self.table: dict = {}
        self.gets = 0

    def _path(self):
        return self.path

    def get(self, card, key):
        self.gets += 1
        return self.table.get(key)

    def put(self, card, key, value):
        self.table[key] = value
        self.path.write_text("{}")
        return True


def _state(
    device = A100_40,
    optims = (),
    dev = True,
    family = "ltx-2.5",
    **over,
):
    pipe = types.SimpleNamespace(_unsloth_ltx25_dev = dev, transformer = None)
    fields = dict(
        family = types.SimpleNamespace(name = family),
        pipe = pipe,
        device = "cuda",
        gpu_ordinal = 0,
        placed_ordinal = None,
        device_name = device,
        speed_optims = tuple(optims),
        repo_id = "unsloth/LTX-2.5-GGUF",
        gguf_filename = "ltx-2.5-22b-dev-UD-Q4_K_XL.gguf",
        transformer_quant = None,
        text_encoder_quant = "int8",
        dtype = "torch.bfloat16",
        speed_mode = "default",
        attention_backend = None,
        offload_policy = "group",
        memory_mode = "auto",
        denoiser_streamed = False,
        transformer_cache = None,
        resolved = {},
    )
    fields.update(over)
    return types.SimpleNamespace(**fields)


SHIPPED_AT_IMPORT = dict(bq.SHIPPED)
# A passing G2c verdict for the A100 classes, so the Auto rules are tested apart from the measured entries.
PASSING = {"verdict": "PASS", "n": 48, "upper": 0.005, "mean": 0.0, "eps": bq.EPS, "path": "any"}


@pytest.fixture(autouse = True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.delenv(fg.FUSED_GUIDANCE_ENV, raising = False)
    for device in (A100_40, A100_80):
        monkeypatch.setitem(
            bq.SHIPPED, (getattr(fg, "QUALITY_FAMILY", "x"), device, "any"), dict(PASSING)
        )
    monkeypatch.setattr(bq, "_file", lambda: tmp_path / "batch_quality.json")
    store = _Store(tmp_path / "diffusion_ltx25_fused_guidance.json")
    monkeypatch.setattr(fg, "_STORE", [store], raising = False)
    memos = [getattr(fg, name, {}) for name in ("_OOM_DECLINED", "LAST_DECISION")]
    for memo in memos:
        memo.clear()
    fg.reset_stats()
    yield store
    for memo in memos:
        memo.clear()


def _promote(
    store,
    state,
    lb = 1.2,
    promoted = True,
    **canvas,
):
    c = dict(CANVAS, **canvas)
    store.put(
        "cuda:0",
        fg.verdict_key(state, **c),
        {"promoted": promoted, "speedup_lb": lb, "reason": "x"},
    )


def _decide(state, **over):
    return fg.decide(state, **dict(CANVAS, **over))


@pytest.mark.parametrize(
    "raw, want",
    [
        (None, None),
        ("", None),
        ("auto", None),
        ("1", True),
        ("on", True),
        ("0", False),
        ("off", False),
    ],
)
def test_env_is_tri_state(monkeypatch, raw, want):
    if raw is not None:
        monkeypatch.setenv(fg.FUSED_GUIDANCE_ENV, raw)
    assert fg.fused_guidance_enabled() is want


def test_env_on_forces_without_any_verdict(monkeypatch):
    monkeypatch.setenv(fg.FUSED_GUIDANCE_ENV, "1")
    d = _decide(_state(device = L4))
    assert d["on"] and d["mode"] == "forced"


def test_env_off_wins_over_both_verdicts(monkeypatch, _clean):
    state = _state()
    _promote(_clean, state)
    assert _decide(state)["on"]
    monkeypatch.setenv(fg.FUSED_GUIDANCE_ENV, "0")
    d = _decide(state)
    assert not d["on"] and fg.FUSED_GUIDANCE_ENV in d["reason"]


def test_auto_without_a_speed_verdict_is_off_and_reads_no_store(_clean):
    d = _decide(_state())
    assert not d["on"] and d["mode"] == "auto"
    assert "no speed race verdict" in d["reason"]
    assert _clean.gets == 0  # no race ever ran: the store file is absent, nothing is fingerprinted


def test_auto_with_a_losing_speed_verdict_is_off(_clean):
    state = _state()
    _promote(_clean, state, lb = 1.03, promoted = False)
    d = _decide(state)
    assert not d["on"] and "did not promote" in d["reason"]


def test_auto_ignores_a_promoted_flag_below_the_bound(_clean):
    state = _state()
    _promote(_clean, state, lb = 1.04, promoted = True)
    assert not _decide(state)["on"]


def test_auto_promoted_race_and_passing_quality_is_on(_clean):
    state = _state()
    _promote(_clean, state, lb = 1.21)
    d = _decide(state)
    assert d["on"] and d["mode"] == "auto" and d["speedup_lb"] == pytest.approx(1.21)


def test_auto_verdict_is_per_canvas(_clean):
    state = _state()
    _promote(_clean, state)
    assert not _decide(state, width = 1024)["on"]
    assert not _decide(state, steps = 40)["on"]
    assert not _decide(state, frames = 97)["on"]


def test_auto_verdict_is_per_load(_clean):
    state = _state()
    _promote(_clean, state)
    assert not _decide(_state(offload_policy = "none"))["on"]
    assert not _decide(_state(denoiser_streamed = True))["on"]
    assert not _decide(_state(attention_backend = "sage"))["on"]
    assert not _decide(_state(gguf_filename = "ltx-2.5-22b-dev-Q8_0.gguf"))["on"]


def test_auto_on_another_card_class_is_off(_clean):
    for device in (L4, "NVIDIA RTX PRO 6000 Blackwell Server Edition", None):
        state = _state(device = device)
        _promote(_clean, state)
        d = _decide(state)
        assert not d["on"] and "G2c" in d["reason"], device


def test_both_a100_classes_ship_the_measured_n48_verdict():
    for device in (A100_40, A100_80):
        found = SHIPPED_AT_IMPORT[(fg.QUALITY_FAMILY, device, fg.QUALITY_PATH)]
        assert found["n"] >= bq.MIN_PAIRS and found["eps"] == bq.EPS
        # above EPS on its own, inside the kernel-swap floor on both metrics: the floor rule admits it
        assert found["upper"] > bq.EPS and found["rule"] == "kernel_floor" and bq.passes(found)
        assert found["worst_upper"] <= found["kernel_floor_worst_upper"]
    assert (fg.QUALITY_FAMILY, L4, fg.QUALITY_PATH) not in SHIPPED_AT_IMPORT
    # the image side's batching families never read this token
    assert all(k[0] != "ltx-2.5" for k in SHIPPED_AT_IMPORT)


def test_the_kernel_floor_rule_needs_every_bound_inside_the_floor():
    base = dict(SHIPPED_AT_IMPORT[(fg.QUALITY_FAMILY, A100_40, fg.QUALITY_PATH)])
    assert bq.passes(base)
    for field, value in (
        ("upper", 0.032),  # past the floor's own bound
        ("worst_upper", 0.036),
        ("vs_kernel_floor_upper", 0.011),  # worse than the kernel control by more than EPS
        ("vs_kernel_floor_worst_upper", 0.011),
        ("n", 12),
    ):
        assert not bq.passes(dict(base, **{field: value})), field
    assert not bq.passes({k: v for k, v in base.items() if k != "kernel_floor_upper"})
    plain = {k: v for k, v in base.items() if k != "rule"}
    assert not bq.passes(plain), "without the rule, the plain EPS bound applies"


def test_a_shipped_failing_verdict_keeps_auto_off(monkeypatch, _clean):
    monkeypatch.setitem(
        bq.SHIPPED, (fg.QUALITY_FAMILY, A100_40, "any"), dict(PASSING, verdict = "FAIL", upper = 0.0149)
    )
    state = _state()
    _promote(_clean, state)
    d = _decide(state)
    assert not d["on"] and "does not pass" in d["reason"]


def test_a_failing_local_quality_verdict_turns_auto_off(_clean):
    state = _state()
    _promote(_clean, state)
    bq.record(fg.QUALITY_FAMILY, A100_40, "any", {"verdict": "FAIL", "n": 48, "upper": 0.02})
    d = _decide(state)
    assert not d["on"] and "does not pass" in d["reason"]


@pytest.mark.parametrize(
    "over, why",
    [
        (dict(conditioned = True), "image-conditioned"),
        (dict(rows = 2), "batched"),
    ],
)
def test_auto_declines_ungated_requests(_clean, over, why):
    state = _state()
    _promote(_clean, state)
    d = _decide(state, **over)
    assert not d["on"] and why in d["reason"]


def test_auto_declines_a_compiled_denoiser(_clean):
    state = _state(optims = ("compiled", "tf32"))
    _promote(_clean, state)
    d = _decide(state)
    assert not d["on"] and "compile" in d["reason"]


def test_not_an_ltx25_dev_load_is_always_off(monkeypatch):
    monkeypatch.setenv(fg.FUSED_GUIDANCE_ENV, "1")
    assert not _decide(_state(dev = False))["on"]
    assert not _decide(_state(family = "wan2.2-ti2v-5b"))["on"]


def test_the_status_lever_does_not_change_the_load_key():
    assert fg.load_key(_state(optims = ("tf32",))) == fg.load_key(
        _state(optims = ("tf32", fg.RACE_KIND))
    )


def test_oom_in_the_4_row_call_falls_back_and_stops_auto_retrying(_clean):
    import torch

    from . import test_video_ltx25_fused_guidance as base

    t = base._transformer()
    runner = base._Runner(t)
    main, _, _ = base._main_kwargs(t)
    state = _state()
    _promote(_clean, state)
    decision = _decide(state)
    assert decision["on"]
    fused = fg.FusedGuidanceState(runner, tuple(base._STG), None, decision["key"])
    real = type(t).__call__

    def call(module, *args, **kwargs):
        if int(kwargs["hidden_states"].shape[0]) == 4:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(module, *args, **kwargs)

    with torch.no_grad():
        out = fused.call(t, call, (), dict(main))
        want = real(t, **main)
    assert torch.equal(out[0], want[0]) and torch.equal(out[1], want[1])
    assert fg.FUSED_STATS["fallbacks"] == 1 and fg.FUSED_STATS["oom_fallbacks"] == 1
    assert fused.disabled and not fused.pending
    again = _decide(state)
    assert not again["on"] and "out of memory" in again["reason"]


def test_extra_rows_scale_with_the_canvas():
    small = fg.extra_rows_mib(768, 512, 49)
    assert 287 < small < 1024  # above the measured +287 MiB peak at this canvas
    # linear in video tokens: 7 latent frames at 49, 13 at 97; 4x the pixels is 4x the tokens
    slack = fg._EXTRA_SLACK_MIB
    assert (fg.extra_rows_mib(768, 512, 97) - slack) / (small - slack) == pytest.approx(
        13 / 7, rel = 0.01
    )
    assert (fg.extra_rows_mib(1536, 1024, 49) - slack) / (small - slack) == pytest.approx(
        4, rel = 0.01
    )


def test_planner_prices_the_4_row_peak_and_declines_when_it_does_not_fit():
    from core.inference import video as vm

    on = {"on": True, "mode": "auto", "reason": "x", "key": "k"}
    kw = dict(
        family = "ltx-2.5",
        floor_mib = 20000,
        placement = "'group' offload",
        width = 768,
        height = 512,
        frames = 49,
        decode_floor_mib = 1000,
        measured_extra_mib = None,
        denoise_mib = None,
    )
    extra = fg.extra_rows_mib(768, 512, 49)
    roomy = 20000 + vm.DEFAULT_BASE_OVERHEAD_MIB + extra + 64
    decision, denoise = vm._ltx25_fused_budget(dict(on), available_mib = roomy, **kw)
    assert decision["on"] and denoise == vm.DEFAULT_BASE_OVERHEAD_MIB + extra
    # Room for the unfused render only: the request runs unfused (never refused for the lever).
    tight = 20000 + vm.DEFAULT_BASE_OVERHEAD_MIB + 1
    assert (
        vm.video_offload_shortfall_message(available_mib = tight, **{k: v for k, v in kw.items()})
        is None
    )
    decision, denoise = vm._ltx25_fused_budget(dict(on), available_mib = tight, **kw)
    assert not decision["on"] and "does not fit" in decision["reason"] and denoise is None
    assert fg.LAST_DECISION["on"] is False


def test_the_fused_budget_prices_the_decode_like_admission():
    from core.inference import video as vm

    on = {"on": True, "mode": "auto", "reason": "x", "key": "k"}
    kw = dict(
        family = "ltx-2.5",
        floor_mib = 20000,
        placement = "'group' offload",
        width = 1920,
        height = 1088,
        frames = 241,
        decode_floor_mib = 20000,  # the weights stay through the decode, so the decode phase binds
        measured_extra_mib = None,
    )
    fused = vm.DEFAULT_BASE_OVERHEAD_MIB + fg.extra_rows_mib(1920, 1088, 241)
    tiled = 256

    def short(available, **extra):
        return vm.video_offload_shortfall_message(
            available_mib = available, denoise_mib = fused, **kw, **extra
        )

    room = next(a for a in range(20000, 400000, 64) if short(a, decode_mib = tiled) is None)
    assert (
        short(room) is not None
    ), "the flat decode estimate must not fit here for this to test anything"
    decision, denoise = vm._ltx25_fused_budget(
        dict(on), available_mib = room, denoise_mib = None, decode_mib = tiled, **kw
    )
    assert (
        decision["on"] and denoise == fused
    )  # the tiled decode admission uses also admits the 4-row call


def test_generate_wires_the_decision_into_the_render_and_the_planner():
    import inspect

    from core.inference import video as vm

    src = inspect.getsource(vm.VideoBackend.generate)
    assert "ltx25_fused_decide(" in src and "fused_guidance = ltx25_fused" in src
    assert "_ltx25_fused_budget(" in src and '"fused_guidance", 4' in src
    # Auto never fuses where the planner did not price it; a fused fallback does not record a fused peak
    assert (
        "not ltx25_fused_priced" in src
        and "vram_split_cards" in src
        and 'ltx25_fused.get("fell_back")' in src
    )


def test_status_block_reports_mode_decision_quality_and_counters(monkeypatch, _clean):
    state = _state()
    block = fg.status(state)
    assert block["mode"] == "auto" and block["engaged"] is False and block["decision"] is None
    assert block["quality"]["verdict"] == "PASS"
    assert set(block["stats"]) >= {
        "fused_calls",
        "served_stg",
        "served_modality",
        "fallbacks",
        "oom_fallbacks",
    }
    _promote(_clean, state)
    d = _decide(state)
    fg.FUSED_STATS["fused_calls"] += 30
    fg._note_outcome(dict(d, engaged = True))
    block = fg.status(state)
    assert (
        block["engaged"] is True
        and block["decision"]["on"] is True
        and block["stats"]["fused_calls"] == 30
    )
    # the next clip ran unfused: engaged follows the last clip, not the process counters
    fg._note_outcome(dict(d, engaged = False))
    assert fg.status(state)["engaged"] is False
    # another load sees no decision of this one
    other = _state()
    assert fg.status(other)["decision"] is None and fg.status(other)["engaged"] is False
    monkeypatch.setenv(fg.FUSED_GUIDANCE_ENV, "0")
    assert fg.status(state)["mode"] == "off"
    assert fg.status(_state(dev = False)) is None


def test_video_status_carries_the_block():
    import inspect

    from core.inference import video as vm
    from models.inference import VideoStatusResponse

    assert "ltx25_fused_guidance" in VideoStatusResponse.model_fields
    src = inspect.getsource(vm.VideoBackend.status)
    assert '"ltx25_fused_guidance": ltx25_fused' in src and '"ltx25_fused_guidance"]' in src


def _race(
    state,
    *,
    off_s = 3.0,
    on_s = 2.0,
    engage = True,
    idle = None,
    repeats = 3,
    budget_declines = False,
):
    clock = [0.0]
    renders: list = []

    def render(prompt, seed, n):
        n = n or CANVAS["steps"]  # None: the load's default, as generate resolves it
        d = fg.decide(state, **dict(CANVAS, steps = n))
        if d["on"] and budget_declines:
            d = fg.decline_for_budget(d, "test")
        renders.append((d["on"], n))
        if d["on"] and engage:
            fg.FUSED_STATS["fused_calls"] += n
        clock[0] += (on_s if d["on"] else off_s) * (1 + 0.01 * (len(renders) % 3))

    report = fg.run_race(
        render,
        state,
        idle = idle or (lambda: True),
        timer = lambda: clock[0],
        repeats = repeats,
    )
    return report, renders


def test_race_promotes_records_and_auto_then_fuses(_clean):
    state = _state()
    assert not _decide(state)["on"]
    report, renders = _race(state)
    v = report["verdict"]
    assert v["promoted"] and v["serves"] and v["speedup_lb"] >= 1.05 and report["recorded"]
    # warm both arms at few steps first, then alternate the order per repeat
    assert renders[:2] == [(False, fg.RACE_WARM_STEPS), (True, fg.RACE_WARM_STEPS)]
    assert [r[0] for r in renders[2:]] == [False, True, True, False, False, True]
    assert _decide(state)["on"]


def test_race_losing_records_a_verdict_auto_keeps_off(_clean):
    state = _state()
    report, _ = _race(state, off_s = 3.0, on_s = 2.95)
    assert not report["verdict"]["promoted"] and report["recorded"]
    d = _decide(state)
    assert not d["on"] and "did not promote" in d["reason"]


def test_race_on_a_card_without_quality_does_not_serve(_clean):
    state = _state(device = L4)
    report, _ = _race(state)
    assert report["verdict"]["promoted"] and not report["verdict"]["serves"]
    assert "G2c" in report["verdict"]["quality_gate"]
    assert not _decide(state)["on"]


def test_race_whose_fused_arm_did_not_engage_is_not_promoted(_clean):
    state = _state()
    report, _ = _race(state, engage = False)
    assert not report["verdict"]["promoted"] and "did not run" in report["verdict"]["reason"]


def test_race_whose_fused_arm_the_planner_declined_is_not_promoted(_clean):
    state = _state()
    report, renders = _race(state, budget_declines = True)
    assert not any(on for on, _ in renders)
    assert not report["verdict"]["promoted"] and not _decide(state)["on"]


def test_race_records_nothing_when_a_request_arrives_during_the_last_render(_clean):
    from core.inference.diffusion_batch_race import RaceRefused

    state = _state()
    calls = [0]

    def idle():
        calls[0] += 1
        return calls[0] <= 8  # 2 warm + 6 timed renders pass; the check before recording does not

    with pytest.raises(RaceRefused, match = "last render"):
        _race(state, idle = idle)
    assert not _clean.table


def test_race_stops_when_a_request_arrives_and_records_nothing(_clean):
    from core.inference.diffusion_batch_race import RaceRefused

    state = _state()
    calls = [0]

    def idle():
        calls[0] += 1
        return calls[0] < 4

    with pytest.raises(RaceRefused):
        _race(state, idle = idle)
    assert not _clean.table


@pytest.mark.parametrize(
    "over, why",
    [
        (dict(optims = ("compiled",)), "compiled"),
        (dict(dev = False), "dev only"),
        (dict(family = "wan2.2-ti2v-5b"), "dev only"),
        (dict(placed_ordinal = None, gpu_ordinal = None, device_kind = "cpu"), "card identity"),
    ],
)
def test_race_refuses_loads_auto_never_fuses(over, why):
    from core.inference.diffusion_batch_race import RaceRefused

    kind = over.pop("device_kind", None)
    state = _state(**over)
    if kind is not None:
        state.device = kind
    with pytest.raises(RaceRefused, match = why):
        _race(state)


def test_race_route_is_registered():
    from routes.video import router
    assert any(getattr(r, "path", "") == "/video/fused-guidance/race" for r in router.routes)


def test_auto_without_verdicts_renders_main_bit_for_bit(monkeypatch):
    import numpy as np
    import torch

    from . import test_video_batch_ltx25 as tiny
    from . import test_video_ltx25_fused_guidance as base
    from core.inference import video_ltx25

    pipe = tiny._tiny_ltx25([])
    if "stg_scale" not in __import__("inspect").signature(pipe.__call__).parameters:
        pytest.skip("this Diffusers' LTX2Pipeline has no STG / modality guidance")
    pipe.transformer = base._transformer()
    pipe._unsloth_ltx25_dev = True
    # A separate-pass lever (as STG prefix reuse would be) wired through the render's one seam.
    entered: list = []
    real_seam = fg.dev_guidance

    @__import__("contextlib").contextmanager
    def lever(runner, call, logger):
        entered.append(type(runner.transformer).__name__)
        yield

    monkeypatch.setattr(
        fg, "dev_guidance", lambda *a, **k: real_seam(*a, **dict(k, separate = (lever,)))
    )
    video_ltx25.prepare_scheduler(pipe, dev = True)
    steps = 2
    blocks: list = []
    pipe.transformer.transformer_blocks[0].register_forward_pre_hook(
        lambda m, a, k: blocks.append(int(k["hidden_states"].shape[0])), with_kwargs = True
    )
    procs = [type(a.processor) for a in fg._cross_modal_attentions(pipe.transformer)]

    def run(decision):
        kwargs = dict(
            prompt = "a red fox in the snow",
            negative_prompt = "blurry",
            width = 64,
            height = 64,
            num_frames = 9,
            num_inference_steps = steps,
            guidance_scale = 3.0,
            spatio_temporal_guidance_blocks = list(base._STG),
            generator = torch.Generator().manual_seed(5),
            output_type = "np",
        )
        with torch.no_grad():
            out = video_ltx25.render(pipe, kwargs, dev = True, fused_guidance = decision)
        return np.asarray(out.frames), np.asarray(
            out.audio[0] if isinstance(out.audio, (list, tuple)) else out.audio
        )

    frames_main, audio_main = run(None)  # the env unset, no decision: main's render
    blocks.clear()
    state = _state(pipe = pipe)
    decision = _decide(state, width = 64, height = 64, frames = 9, steps = steps, guidance = 3.0)
    assert not decision["on"]
    frames_auto, audio_auto = run(decision)
    assert blocks == [2, 1, 1] * steps
    assert np.array_equal(frames_auto, frames_main) and np.array_equal(audio_auto, audio_main)
    assert (
        len(entered) == 2
    )  # main's render and Auto-off both ran the separate passes with the lever
    assert fg.FUSED_STATS["renders"] == 0
    assert decision["engaged"] is False and decision["fell_back"] is False
    assert fg.status(state)["engaged"] is False
    assert [type(a.processor) for a in fg._cross_modal_attentions(pipe.transformer)] == procs
    # forced on, the same decision path engages and the status says so
    monkeypatch.setenv(fg.FUSED_GUIDANCE_ENV, "1")
    forced = _decide(state, width = 64, height = 64, frames = 9, steps = steps, guidance = 3.0)
    blocks.clear()
    run(forced)
    assert (
        blocks == [4] * steps and forced["engaged"] is True and fg.status(state)["engaged"] is True
    )
    assert (
        len(entered) == 2
    )  # fused engaged: the separate-pass lever was not entered on this render
    assert type(pipe.transformer).__name__ == "LTX2VideoTransformer3DModel"


def test_the_render_has_one_seam_for_fused_or_separate_passes():
    import inspect

    from core.inference import video_ltx25

    src = inspect.getsource(video_ltx25._render)
    assert src.count("dev_guidance(") == 1 and "fused_guidance(" not in src.replace(
        "dev_guidance(", ""
    )


def test_seam_enters_separate_levers_only_when_fused_does_not_engage(monkeypatch):
    import contextlib

    from . import test_video_ltx25_fused_guidance as base

    entered: list = []

    @contextlib.contextmanager
    def lever(runner, call, logger):
        entered.append(getattr(type(runner.transformer), "_unsloth_fused_guidance_cls", False))
        yield

    t = base._transformer()
    runner = base._Runner(t)
    with fg.dev_guidance(runner, {}, decision = {"on": True, "key": None}, separate = (lever,)) as path:
        assert path == "fused" and getattr(type(t), "_unsloth_fused_guidance_cls", False)
    assert entered == []
    with fg.dev_guidance(runner, {}, decision = {"on": False}, separate = (lever,)) as path:
        assert path == "separate"
    # the lever sees the stock class: nothing of fused guidance is installed under it
    assert entered == [False]
    monkeypatch.setenv(fg.FUSED_GUIDANCE_ENV, "0")
    with fg.dev_guidance(runner, {}, separate = (lever,)) as path:
        assert path == "separate"
    assert entered == [False, False]
    # declined at install (a step cache): the separate passes and their levers run
    monkeypatch.setenv(fg.FUSED_GUIDANCE_ENV, "1")
    monkeypatch.setattr(type(t), "is_cache_enabled", property(lambda self: True), raising = False)
    with fg.dev_guidance(runner, {}, separate = (lever,)) as path:
        assert path == "separate"
    assert len(entered) == 3
