# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The race measures batch size 3, and a seeded request joins only at sizes measured seeded-exact."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from core.inference import diffusion_batch_quality as quality
from core.inference import diffusion_batch_race as race
from core.inference import diffusion_batch_trace as trace
from core.inference import image_request_batching as batching

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_image_batch_infra import _Image, _engine, _fake_generate, _verdicts  # noqa: E402
from test_image_concurrency import (  # noqa: E402,F401  (fixture)
    _batched_calls,
    _create,
    _post_while_rendering,
    studio,
)

CARD = "NVIDIA A100-SXM4-80GB"
EXACT = {"promoted": True, "neighbour_exact": True, "replay_exact": True, "b_exact": True}


@pytest.fixture(autouse = True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(quality, "_file", lambda: tmp_path / quality.FILE_NAME)
    monkeypatch.setattr(quality, "SHIPPED", {})
    monkeypatch.delenv("UNSLOTH_STUDIO_IMAGE_BATCH_SEEDED", raising = False)
    trace.reset()
    yield
    trace.reset()


def _race(generate, clock, **kw):
    return race.run_race(
        generate,
        types.SimpleNamespace(cuda_graphs = ()),
        idle = lambda: True,
        width = 512,
        height = 512,
        repeats = 3,
        persist = False,
        clock = lambda: clock["t"],
        **kw,
    )


def test_the_race_measures_three_by_default():
    assert 3 in race.RACE_SIZES and {2, 4} <= set(race.RACE_SIZES)
    assert race.invalid_race_sizes(list(race.RACE_SIZES)) is None


def test_the_engine_races_the_default_sizes_including_three(monkeypatch):
    seen = {}
    monkeypatch.setattr(race, "run_race", lambda *a, **k: seen.update(k) or {})
    _engine().batch_race(width = 512, height = 512)
    assert tuple(seen["sizes"]) == race.RACE_SIZES and 3 in seen["sizes"]
    seen.clear()
    _engine().batch_race(width = 512, height = 512, sizes = (2,))
    assert tuple(seen["sizes"]) == (2,), "an explicit size list is kept"


def test_a_three_row_race_runs_and_promotes_a_real_win():
    generate, clock, calls = _fake_generate(lambda rows: 1.0 + 0.4 * (rows - 1))
    verdict = _race(generate, clock, sizes = (3,))["sizes"]["3"]
    assert verdict["promoted"] is True and verdict["size"] == 3
    assert 3 in calls


def test_a_graphed_three_row_win_never_claims_a_batch_graph_that_was_not_recorded():
    """Graphs exist only at BATCH_GRAPH_SIZES (2, 4), so a size-3 verdict serves only via the eager arm."""
    handle = types.SimpleNamespace(
        cache = {"k2": object(), "k4": object()},
        _batched = {"k2": 2, "k4": 4},
        _batched_canvas = {},
        drop_batched = lambda size: 0,
    )
    state = types.SimpleNamespace(cuda_graphs = (handle,))
    won = dict(
        EXACT,
        arm = "batched",
        path = "graph",
        layout = race.layout_token(()),
        graph_canvas = "c",
        arms = {"batched_eager": {"promoted": False}},
    )
    got = race.effective_verdict(won, 3, state, layout = ())
    assert got["promoted"] is False and got["graph_missing"] is True
    eager_won = dict(won, arms = {"batched_eager": {"promoted": True}})
    got = race.effective_verdict(eager_won, 3, state, layout = ())
    assert got["promoted"] is True and got["path"] == "eager"
    assert (
        got["b_exact"] is False and got["neighbour_exact"] is False
    ), "the neighbour check ran on the graphed arm: it vouches for nothing eager"


def test_a_streamed_load_names_the_sizes_a_seeded_request_may_join_at(monkeypatch):
    streamed = _engine(offload_policy = "sequential")
    _verdicts(monkeypatch, {2: EXACT, 3: dict(EXACT, b_exact = False), 4: {"promoted": False}})
    monkeypatch.setattr(quality, "quality_ok", lambda state, path: True)
    assert streamed.coalesce_seeded_ok(1024, 1024, 4.0, 1) is False
    assert streamed.coalesce_seeded_sizes(1024, 1024, 4.0, 1) == frozenset({2})
    assert (
        streamed.coalesce_seeded_sizes(1024, 1024, 4.0, 2) == frozenset()
    ), "a size must exceed its own images"
    _verdicts(monkeypatch, {})
    assert streamed.coalesce_seeded_sizes(1024, 1024, 4.0, 1) == frozenset()


def test_a_seeded_size_must_also_pass_the_quality_gate(monkeypatch):
    streamed = _engine(offload_policy = "sequential", device_name = CARD)
    _verdicts(monkeypatch, {2: dict(EXACT, arm = "batched")})
    assert streamed.coalesce_seeded_sizes(1024, 1024, 4.0, 1) == frozenset(), "no G2c verdict"
    quality.record("flux.2-klein", CARD, "graph", {"verdict": "PASS", "n": 24, "upper": 0.004})
    assert streamed.coalesce_seeded_sizes(1024, 1024, 4.0, 1) == frozenset({2})


def test_a_stored_seeded_quality_verdict_admits_a_neighbour_exact_size(monkeypatch):
    streamed = _engine(offload_policy = "sequential")
    _verdicts(monkeypatch, {2: dict(EXACT, b_exact = False)})
    monkeypatch.setattr(quality, "quality_ok", lambda state, path: True)
    monkeypatch.setattr(quality, "seeded_quality_ok", lambda state, path: False)
    assert streamed.coalesce_seeded_sizes(1024, 1024, 4.0, 1) == frozenset()
    monkeypatch.setattr(quality, "seeded_quality_ok", lambda state, path: True)
    # same per-size rule as coalesce_seeded_ok: a G2c verdict with ``seeded`` admits a neighbour-exact win
    assert streamed.coalesce_seeded_sizes(1024, 1024, 4.0, 1) == frozenset({2})


def test_the_coalescer_fires_a_batch_holding_a_seeded_member_only_at_a_seeded_size():
    member = lambda seeded: batching._Member("p", [1], None, None, seeded)  # noqa: E731
    unseeded, seeded = member(False), member(True)
    two = {2}.__contains__
    assert batching._fitting_prefix([unseeded, unseeded, unseeded], None, two) == [unseeded] * 3
    assert batching._fitting_prefix([unseeded, seeded, unseeded], None, two) == [unseeded, seeded]
    assert batching._fitting_prefix([unseeded, unseeded, seeded], None, two) == [unseeded, unseeded]
    assert batching._fitting_prefix([seeded, unseeded], lambda total: False, two) == []
    assert batching._fitting_prefix([seeded, unseeded, unseeded], None, None) == [
        seeded,
        unseeded,
        unseeded,
    ], "no seeded_fits: today's rule"


def _seeded_route(studio, monkeypatch, seeded_sizes, order):
    app, backend, _saved = studio
    backend.coalesce_seeded_sizes = lambda w, h, g, images, steps = None, layout = None: frozenset(
        seeded_sizes
    )
    seeds = {"alice": None, "bob": 22, "carol": None}
    posts = [
        (
            "/api/inference/images/generate",
            name,
            _create(f"{name} prompt", **({"seed": seeds[name]} if seeds[name] is not None else {})),
        )
        for name in order
    ]
    waiting = len(posts) if seeded_sizes else len(posts) - 1
    responses = _post_while_rendering(app, backend, posts, waiting)
    assert all(r.status_code == 200 for r in responses), [r.text for r in responses]
    [bob] = [c for c in backend.calls if 22 in (c["seeds"] or [c["seed"]])]
    return bob, backend


def test_a_seeded_request_joins_the_route_at_a_measured_exact_size(studio, monkeypatch):
    bob, backend = _seeded_route(studio, monkeypatch, {2}, ["alice", "bob", "carol"])
    assert bob["prompts"] == ["alice prompt", "bob prompt"], "bob joins at the exact size 2"
    assert bob["seeds"][1] == 22


def test_a_seeded_request_never_renders_at_a_size_without_a_seeded_verdict(studio, monkeypatch):
    for sizes in ({2}, {4}, set()):
        bob, backend = _seeded_route(studio, monkeypatch, sizes, ["alice", "carol", "bob"])
        size = len(bob["prompts"] or [None])
        assert size == 1 or size in sizes, (sizes, bob)
        backend.calls.clear()


def test_a_backend_without_seeded_sizes_keeps_the_all_sizes_answer(studio, monkeypatch):
    app, backend, _saved = studio
    backend.coalesce_seeded_ok = lambda w, h, g, images, steps = None, layout = None: False
    posts = [
        ("/api/inference/images/generate", "alice", _create("a fox")),
        ("/api/inference/images/generate", "bob", _create("a cat", seed = 22)),
    ]
    _post_while_rendering(app, backend, posts, 1)
    [bob] = [c for c in backend.calls if 22 in (c["seeds"] or [c["seed"]])]
    assert bob["prompts"] is None, "no seeded verdict: the seeded request renders alone"
