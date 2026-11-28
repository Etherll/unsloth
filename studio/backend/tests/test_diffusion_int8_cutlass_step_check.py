# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The sustained whole-step check of the CUTLASS race winners (CPU).

Drives the real deferred image-load path (``_plan_or_defer``, the deferred race units, ``_run``'s dispatch, the
finalizer and the step check) with the timers replaced by a card model (``_Card``). The model's per-shape timings
(short, sustained, paired) let the CUTLASS tiles win, as the race lines on the RTX 6000 Ada imply (150 Linears on
CUTLASS after #19879). Only a whole step of every GEMM, sustained, runs at the clock a render ran at (R10: 1340 MHz on
CUTLASS against 1480 MHz on Triton)."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import types

import pytest

from core.inference import diffusion_int8_cutlass as cut
from core.inference import diffusion_int8_cutlass_defer as defer
from core.inference import diffusion_int8_gemm as g8

torch = pytest.importorskip("torch")


@dataclasses.dataclass(frozen = True)
class _Power:
    boost_mhz: float
    # one weight shape run back to back (the race's sustained / paired timings)
    shape_mhz: dict
    # a whole step of every int8 GEMM, sustained: all on today's path vs all on the CUTLASS tiles
    step_mhz: dict


POWER = {
    # RTX 6000 Ada, 300 W: the tiles win every per-shape timing; the whole step droops on them (tester R10)
    "ada_droop": _Power(
        2505.0,
        {"today": 1480.0, "cutlass": 1470.0, "paired": 1475.0},
        {"today": 1480.0, "cutlass": 1340.0},
    ),
    # a power-capped card where the tiles win sustained too (RTX PRO 6000 / 3090 style)
    "capped_wins": _Power(
        2100.0,
        {"today": 1800.0, "cutlass": 1780.0, "paired": 1790.0},
        {"today": 1800.0, "cutlass": 1770.0},
    ),
    # clock-stable card (A100 style)
    "stable": _Power(
        1410.0,
        {"today": 1410.0, "cutlass": 1410.0, "paired": 1410.0},
        {"today": 1410.0, "cutlass": 1410.0},
    ),
}
# tile speed at equal clock relative to today's path, and rows per tile
_TILE_SPEED = {0: 0.95, 1: 0.96, 2: 0.98}
_TILE_ROWS = {0: 128, 1: 128, 2: 256}

H3I = "MiniMaxH3ImageTransformer2DModel"
# (label, N, K, Linears, rows per call): MiniMax-H3-Image at 2048x2048 (~4100 image + text rows)
INVENTORY = (
    ("qkv", 7168, 5376, 150, 4100),
    ("out", 5376, 7168, 50, 4100),
    ("ff_in", 28672, 5376, 50, 4100),
    ("ff_out", 5376, 14336, 50, 4100),
    ("context", 5376, 5120, 1, 512),
)
IMAGE_LINEARS = sum(c for _l, _n, _k, c, r in INVENTORY if r >= 1024)


class _Card:
    def __init__(self, power: _Power):
        self.p = power
        self.shape = (0, 0, 0)
        self.pairing = False
        self.rounds = 0
        self.recording = None
        self.launches = {"cutlass": 0, "today": 0}

    def begin(self, m, n, k):
        self.shape = (int(m), int(n), int(k))

    def _work(self, name, m, n, k):
        speed = 1.0 if name in ("today", "cublas") else _TILE_SPEED[int(name)]
        return m * n * k * 1e-6 * (speed + (1.0 if self.pairing else 0.0))

    def time_us(
        self,
        fns,
        rounds = 3,
        reps = 3,
        warm = 1,
    ):
        return {name: self._work(name, *self.shape) / self.p.boost_mhz for name in fns}

    def time_sustained_us(
        self,
        fns,
        settle_s = 0.25,
        measure_s = 0.25,
    ):
        out = {}
        for name in fns:
            if self.shape[0] < cut._LARGE_M:
                mhz = self.p.boost_mhz
            elif name in ("today", "cublas"):
                mhz = self.p.shape_mhz["today"]
            else:
                mhz = self.p.shape_mhz["paired" if self.pairing else "cutlass"]
            out[name] = self._work(name, *self.shape) / mhz
        return out

    def time_round_us(
        self,
        fn,
        index = None,
        settle_s = 1.0,
        measure_s = 1.0,
    ):
        self.rounds += 1
        self.recording = []
        fn()
        calls, self.recording = self.recording, None
        work = {kind: 0.0 for kind in ("today", "cutlass")}
        for tile, m, n, k in calls:
            work["today" if tile is None else "cutlass"] += self._work(
                "today" if tile is None else tile, m, n, k
            )
        share = work["cutlass"] / max(1e-12, sum(work.values()))
        lo, hi = self.p.step_mhz["cutlass"], self.p.step_mhz["today"]
        mhz = hi - (hi - lo) * share
        return {"us": sum(work.values()) / mhz, "mhz": round(mhz), "w": 287.0 if share else 299.0}

    def note(self, tile, a, w):
        if self.recording is not None:
            self.recording.append((tile, int(a.shape[0]), int(w.shape[0]), int(w.shape[1])))
        else:
            self.launches["today" if tile is None else "cutlass"] += 1


def _ns(*shape, dtype = torch.int8):
    t = types.SimpleNamespace(shape = tuple(shape), dtype = dtype, is_cuda = True)
    t.device = types.SimpleNamespace(index = 0, type = "cuda")
    t.stride = lambda dim = None: 1
    t.contiguous = lambda: t
    return t


def _all_state():
    """Every per-device ``_CUT_*`` dict diffusion_int8_gemm keeps (also ones added later), and the probe cache."""
    return [v for k, v in vars(g8).items() if k.startswith("_CUT_") and isinstance(v, dict)] + [
        cut._PROBED
    ]


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for env in (
        getattr(cut, "STEP_CHECK_ENV", "UNSLOTH_DIFFUSION_INT8_CUTLASS_STEP_CHECK"),
        cut.PAIRED_ENV,
        cut.SUSTAINED_ENV,
        cut.CUTLASS_ENV,
        cut.ROW_RACE_ENV,
        cut.LIVE_RACE_ENV,
        cut.HUGE_RACE_ENV,
        cut.FILL_TIE_ENV,
        defer.DEFER_ENV,
    ):
        monkeypatch.delenv(env, raising = False)
    saved = [(d, dict(d)) for d in _all_state()]
    paired = dict(getattr(cut, "_PAIRED_ROWS", {}))
    defer._reset_for_tests()
    yield
    defer._reset_for_tests()
    getattr(defer, "_AFTER_RENDER", []).clear()
    for d, old in saved:
        d.clear()
        d.update(old)
    if hasattr(cut, "_PAIRED_ROWS"):
        cut._PAIRED_ROWS.clear()
        cut._PAIRED_ROWS.update(paired)


class _Studio:
    """One Studio process on a modelled card: image loads, renders, the deferred queue run in order, a disk cache."""

    def __init__(
        self,
        monkeypatch,
        card: str,
        disk: dict,
        image: bool = True,
    ):
        self.mp, self.sim, self.disk, self.image = monkeypatch, _Card(POWER[card]), disk, image
        self.jobs: list = []
        self.rendering = False
        self.races = {"in_render": 0, "outside": 0}
        for d in _all_state():
            d.clear()
        self._patch()

    def _patch(self):
        mp, sim = self.mp, self.sim
        real_paired = cut.paired_cost_us
        real_race = cut.race

        def race(*a, **k):
            self.races["in_render" if self.rendering else "outside"] += 1
            return real_race(*a, **k)

        mp.setattr(cut, "race", race)

        def paired_cost_us(fns, base_name):
            sim.pairing = True
            try:
                return real_paired(fns, base_name)
            finally:
                sim.pairing = False

        def eligible(a, w, xs, ws, bias, out_dtype):
            sim.begin(a.shape[0], w.shape[0], w.shape[1])
            return True

        def launch(
            a,
            w,
            xs,
            ws,
            bias,
            fam,
            tile,
            out_dtype,
            epilogue = 0,
        ):
            sim.note(tile, a, w)

        def today(index, a, w, xs, ws, bias):
            sim.note(None, a, w)

        def schedule(
            index,
            units,
            logger,
            before,
            after,
            owner = None,
        ):
            self.jobs.append([before, *units, after])

        tiles = tuple(sorted(_TILE_SPEED))
        to = torch.Tensor.to
        gen = torch.Generator

        def to_device(t, *a, **k):
            if a and isinstance(a[0], (torch.device, str)) and str(a[0]).startswith("cuda"):
                return t
            return to(t, *a, **k)

        def lazy(
            *size,
            generator = None,
            dtype = None,
            device = None,
            **kw,
        ):
            size = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
            return torch.zeros(*size, dtype = dtype or torch.float32)

        def lazy_randint(
            low,
            high,
            size,
            generator = None,
            dtype = None,
            device = None,
            **kw,
        ):
            return torch.zeros(size, dtype = dtype or torch.int64)

        mp.setattr(cut, "paired_cost_us", paired_cost_us)
        mp.setattr(cut, "time_us", sim.time_us)
        mp.setattr(cut, "time_sustained_us", sim.time_sustained_us)
        if hasattr(cut, "time_round_us"):
            mp.setattr(cut, "time_round_us", sim.time_round_us)
        mp.setattr(cut, "eligible", eligible)
        mp.setattr(cut, "launch", launch)
        mp.setattr(cut, "has_kernel", lambda *a, **k: True)
        mp.setattr(cut, "probed_tiles", lambda index: tiles)
        mp.setattr(cut, "_stored_tiles", lambda index, fam: tiles)
        mp.setattr(
            cut, "tiles", lambda fam: tuple((_TILE_ROWS[t], 128, 64, 0, 0, 0, 0, 0) for t in tiles)
        )
        mp.setattr(g8, "cutlass_wanted", lambda cap: True)
        mp.setattr(g8, "_cut_cache_key", lambda index: "modelled card|race key")
        mp.setattr(g8, "_cut_cache_read", lambda: json.loads(json.dumps(self.disk)))
        mp.setattr(g8, "_cut_cache_write", self._write)
        mp.setattr(g8, "_xs_dtype", lambda *a, **k: torch.float32)
        mp.setattr(g8, "_huge_race_fits", lambda *a, **k: True)
        mp.setattr(g8, "_launch_or_ref", today)
        mp.setattr(g8, "reference", lambda a, w, xs, ws, bias: sim.note(None, a, w))
        mp.setattr(defer, "allowed", lambda: self.image)
        mp.setattr(defer, "schedule", schedule)
        mp.setattr(defer, "pending", lambda index = None: bool(self.jobs))
        mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
        mp.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        mp.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
        mp.setattr(torch.cuda, "empty_cache", lambda: None)
        mp.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (40 << 30, 48 << 30))
        mp.setattr(torch.cuda, "device", lambda *a, **k: contextlib.nullcontext())
        mp.setattr(torch, "Generator", lambda device = None: gen(device = "cpu"))
        mp.setattr(torch, "randint", lazy_randint)
        mp.setattr(torch, "rand", lazy)
        mp.setattr(torch, "randn", lazy)
        mp.setattr(torch.Tensor, "to", to_device)

    def _write(self, table: dict) -> None:
        """The JSON file round trip of ``_cut_cache_write``."""
        table = json.loads(json.dumps(table))
        self.disk.clear()
        self.disk.update(table)

    def load(self, dit: str = H3I):
        """``_swap_on``'s CUTLASS part for this inventory: the family switch, the step-call reset, the plan."""
        if hasattr(cut, "set_paired_rows"):
            cut.set_paired_rows(0, type(dit, (), {})())
        g8._CUT_TRITON_ON[0] = True
        if hasattr(g8, "_step_reset"):
            g8._step_reset(0)
        self.weights, recs = {}, []
        for _label, n, k, _count, _rows in INVENTORY:
            w = types.SimpleNamespace(
                qdata = torch.empty(n, k, dtype = torch.int8),
                scale = torch.empty(n, 1, dtype = torch.bfloat16),
            )
            self.weights[(n, k)] = w
            recs.append((types.SimpleNamespace(bias = None), ("v2", None, False, w, True)))
        with defer.allow() if self.image else contextlib.nullcontext():
            g8._plan_or_defer(0, recs, True, False, None)

    def render(
        self,
        steps: int = 2,
        rows_at: int = 0,
    ) -> dict:
        """One render: every Linear called once per step through ``_run``; returns the step's launches.
        ``rows_at``: the image-row calls' row count (another resolution); 0 keeps the inventory's."""
        self.sim.launches = {"cutlass": 0, "today": 0}
        defer.render_begin()
        self.rendering = True
        try:
            self._steps(steps, rows_at)
        finally:
            self.rendering = False
            defer.render_end()
        return {k_: v // steps for k_, v in self.sim.launches.items()}

    def _steps(
        self,
        steps: int,
        rows_at: int = 0,
    ) -> None:
        for _ in range(steps):
            for _label, n, k, count, rows in INVENTORY:
                rows = rows_at if rows_at and rows >= 1024 else rows
                for _ in range(count):
                    g8._run(
                        _ns(rows, k),
                        _ns(n, k),
                        _ns(rows, dtype = torch.float32),
                        _ns(n, dtype = torch.bfloat16),
                        None,
                    )

    def idle(self):
        """The deferred queue between renders, in order (a finalizer may queue the next job)."""
        while self.jobs:
            for fn in self.jobs.pop(0):
                for _ in range(5):
                    if fn() is not False:
                        break


def _session(
    monkeypatch,
    card: str,
    disk: dict,
    renders: int = 2,
) -> tuple:
    """Load, first render, idle (deferred race + step check), then ``renders`` renders: (studio, CUTLASS calls a step)."""
    s = _Studio(monkeypatch, card, disk)
    s.load()
    first = s.render()
    s.idle()
    after = [s.render()["cutlass"] for _ in range(renders)]
    return s, first["cutlass"], after


def _small_rows_cutlass(studio: _Studio) -> int:
    """CUTLASS calls a step below 1024 rows (the 512-row race, untouched by the step check)."""
    entry = (g8._CUT_PLAN.get(0) or {}).get(
        g8._cut_key(5376, 5120, torch.float32, torch.bfloat16, False)
    ) or {}
    return 1 if entry and cut.slot(512, entry) is not None else 0


def test_droop_card_keeps_triton_on_h3_image_rows(monkeypatch):
    """Ada-style droop: the tiles win the race (alone and paired), the sustained whole step loses on them, so every
    image-row Linear goes back to Triton (origin/main: 300 of 300 on CUTLASS)."""
    disk: dict = {}
    s, first, after = _session(monkeypatch, "ada_droop", disk)
    assert first == 0  # first image: the deferred shapes run Triton
    small = _small_rows_cutlass(s)
    assert after == [small, small], after
    assert s.sim.rounds == 6  # 3 interleaved pairs, one CUTLASS and one Triton round each
    step = g8._CUT_STEP[0]
    assert step["keep"] is False and step["ratio"] > 1.04, step
    # the race itself still picked the tiles: only the step check moved them
    races = g8._CUT_RACES[0]
    assert all(r[4160]["tile"] is not None for k_, r in races.items() if 4160 in r), races


def test_droop_card_without_the_check_keeps_cutlass(monkeypatch):
    """``UNSLOTH_DIFFUSION_INT8_CUTLASS_STEP_CHECK=0``: origin/main's routing, nothing timed."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_CUTLASS_STEP_CHECK", "0")
    s, _first, after = _session(monkeypatch, "ada_droop", {})
    assert after[-1] == IMAGE_LINEARS + _small_rows_cutlass(s), after
    assert s.sim.rounds == 0


def test_stored_veto_applies_before_the_first_image(monkeypatch):
    """The next process on that card serves Triton from the first image and times nothing."""
    disk: dict = {}
    _session(monkeypatch, "ada_droop", disk)
    s = _Studio(monkeypatch, "ada_droop", disk)
    s.load()
    small = _small_rows_cutlass(s)
    assert s.render()["cutlass"] == small
    s.idle()
    assert s.render()["cutlass"] == small
    assert s.sim.rounds == 0
    assert g8._CUT_STEP[0]["how"] == "stored"


def test_upgrade_with_a_cached_race_checks_after_the_first_image(monkeypatch):
    """A card whose race verdicts were stored before this check (the tester's Ada after an update): the first image
    still runs CUTLASS (no timing on a request), the check runs once it ends, and the next render runs Triton."""
    disk: dict = {}
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_CUTLASS_STEP_CHECK", "0")
    _session(monkeypatch, "ada_droop", disk)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_CUTLASS_STEP_CHECK")
    s = _Studio(monkeypatch, "ada_droop", disk)
    s.load()
    assert not s.jobs  # nothing queued at load
    small = _small_rows_cutlass(s)
    assert s.render()["cutlass"] == IMAGE_LINEARS + small
    assert s.sim.rounds == 0 and len(s.jobs) == 1  # queued as the first image ended
    s.idle()
    assert s.sim.rounds == 6 and g8._CUT_STEP[0]["keep"] is False
    assert s.render()["cutlass"] == small


@pytest.mark.parametrize("card", ["capped_wins", "stable"])
def test_cards_where_cutlass_wins_sustained_keep_it(monkeypatch, card):
    """The step check runs and keeps every winner: same routing as without it."""
    s, _first, after = _session(monkeypatch, card, {})
    assert s.sim.rounds == 6
    assert g8._CUT_STEP[0]["keep"] is True, g8._CUT_STEP[0]
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_CUTLASS_STEP_CHECK", "0")
    off, _f, off_after = _session(monkeypatch, card, {})
    assert after == off_after and after[-1] >= IMAGE_LINEARS, (after, off_after)


def test_the_check_leaves_the_race_verdicts_alone(monkeypatch):
    """The race cache entries are the same with and without the check (only the step table is new)."""
    on: dict = {}
    off: dict = {}
    _session(monkeypatch, "ada_droop", on)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_CUTLASS_STEP_CHECK", "0")
    _session(monkeypatch, "ada_droop", off)
    step_keys = [k_ for k_ in on if "|step=" in k_]
    assert len(step_keys) == 1
    assert {k_: v for k_, v in on.items() if k_ not in step_keys} == off


def test_video_loads_race_at_load_and_skip_the_check(monkeypatch):
    """Video loads do not defer (no image arm): no step check, routing as before."""
    s = _Studio(monkeypatch, "ada_droop", {}, image = False)
    s.load("MiniMaxH3VideoTransformer3DModel")
    got = s.render()
    s.idle()
    assert s.sim.rounds == 0 and not s.jobs
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_CUTLASS_STEP_CHECK", "0")
    s2 = _Studio(monkeypatch, "ada_droop", {}, image = False)
    s2.load("MiniMaxH3VideoTransformer3DModel")
    assert s2.render() == got


def test_a_new_row_bucket_races_after_the_render_not_inside_it(monkeypatch):
    """A render at another resolution meets a row bucket with no verdict: it keeps the nearest slot and the bucket
    races once it ends, through the deferred queue (origin/main raced it inside the render, before the first call)."""
    disk: dict = {}
    _session(monkeypatch, "capped_wins", disk)
    s = _Studio(monkeypatch, "capped_wins", disk)
    s.load()
    s.render()
    s.idle()
    before = s.races["outside"]
    s.render(rows_at = 2100)  # a smaller image: 2112-row bucket, never raced
    assert s.races["in_render"] == 0
    assert s.jobs  # queued as the render ended
    s.idle()
    assert s.races["outside"] > before
    key = g8._cut_key(28672, 5376, torch.float32, torch.bfloat16, False)
    assert 2112 in g8._CUT_PLAN[0][key]
    assert s.races["in_render"] == 0


def test_video_loads_keep_the_live_race_at_the_first_call(monkeypatch):
    """Video loads do not defer: a new row bucket races at its first call, as before."""
    s = _Studio(monkeypatch, "capped_wins", {}, image = False)
    s.load("MiniMaxH3VideoTransformer3DModel")
    s.render()
    assert s.races["in_render"] > 0


def test_step_verdict_median_and_margin():
    assert cut.step_verdict([(1.05, 1.0), (1.06, 1.0), (0.90, 1.0)]) == {
        "keep": False,
        "ratio": 1.05,
    }
    assert cut.step_verdict([(1.009, 1.0)] * 3)["keep"] is True  # a near tie keeps the winners
    assert cut.step_verdict([(0.95, 1.0), (0.97, 1.0)])["ratio"] == 0.96
    assert cut.step_verdict([]) == {"keep": True, "ratio": None}


def test_a_new_row_bucket_that_loses_re_records_the_graphs(monkeypatch):
    """The render captured the nearest CUTLASS slot for the new bucket; a deferred race that picks Triton still
    changes that route, so the next render re-records its graphs."""
    disk: dict = {}
    _session(monkeypatch, "capped_wins", disk)
    s = _Studio(monkeypatch, "capped_wins", disk)
    s.load()
    s.render()
    s.idle()
    refreshed: list = []
    monkeypatch.setattr(defer, "request_refresh", lambda: refreshed.append(1))

    def lose(index, a, w, xs, ws, bias):
        key = g8._cut_key(w.shape[0], w.shape[1], xs.dtype, ws.dtype, bias is not None)
        b = cut.row_bucket(int(a.shape[0]))
        g8._CUT_LIVE.setdefault(index, set()).add((key, b))
        g8._CUT_PLAN[index][key][b] = None

    monkeypatch.setattr(g8, "_live_row_race", lose)
    s.render(rows_at = 2100)
    s.idle()
    key = g8._cut_key(28672, 5376, torch.float32, torch.bfloat16, False)
    assert g8._CUT_PLAN[0][key][2112] is None
    assert refreshed


def test_a_merged_deferred_job_runs_every_finalizer(monkeypatch):
    """The step check and a live race queued under one (card, owner) merge into one job: both finalizers run, in
    queue order, so the live race's own step check is not dropped."""
    monkeypatch.setattr(defer, "_JOBS", {})
    monkeypatch.setattr(defer, "_STATUS", dict(defer._STATUS))
    monkeypatch.setattr(defer, "_STATE", dict(defer._STATE))
    calls: list = []
    defer.schedule(
        0, [lambda: 0], None, before = lambda: 0, after = lambda: calls.append("step"), owner = "t"
    )
    defer.schedule(
        0, [lambda: 0], None, before = lambda: 0, after = lambda: calls.append("live"), owner = "t"
    )
    job = defer._JOBS[(0, "t")]
    assert len(job.units) == 3  # one ``before``, both units
    job.after()
    assert calls == ["step", "live"]


def test_a_render_between_step_units_still_completes_the_check(monkeypatch):
    """A render at another already-raced resolution lands between two step units and changes what the step runs:
    the pairs taken so far are dropped, and the units left still finish a full check on the new signature."""
    disk: dict = {}
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_CUTLASS_STEP_CHECK", "0")
    s0 = _Studio(monkeypatch, "ada_droop", disk)
    s0.load()
    s0.render()
    s0.idle()
    s0.render(rows_at = 2100)
    s0.idle()  # both row buckets raced and stored
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_CUTLASS_STEP_CHECK")
    s = _Studio(monkeypatch, "ada_droop", disk)
    s.load()
    s.render()
    (job,) = s.jobs
    s.jobs.clear()
    before, first, *rest = job
    before()
    assert first() is not False
    s.render(rows_at = 2100)  # between units: the step signature moves
    for fn in rest:
        for _ in range(5):
            if fn() is not False:
                break
    step = g8._CUT_STEP.get(0)
    assert step is not None and step.get("how") == "measured", step
    assert len(step.get("us") or []) == cut._STEP_PAIRS, step


def test_another_cached_resolution_gets_its_own_step_check(monkeypatch):
    """Both resolutions' buckets come from the race cache: after the first resolution is checked, a render at the
    other one runs a new step signature, and that gets its own check instead of serving unchecked tiles."""
    disk: dict = {}
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_CUTLASS_STEP_CHECK", "0")
    s0 = _Studio(monkeypatch, "ada_droop", disk)
    s0.load()
    s0.render()
    s0.idle()
    s0.render(rows_at = 2100)
    s0.idle()
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_CUTLASS_STEP_CHECK")
    s = _Studio(monkeypatch, "ada_droop", disk)
    s.load()
    s.render()
    s.idle()
    assert "@4160" in g8._CUT_STEP[0]["sig"]
    s.render(rows_at = 2100)
    assert s.jobs, "the new signature queued a check"
    s.idle()
    step = g8._CUT_STEP[0]
    assert "@2112" in step["sig"] and step["how"] == "measured", step


def test_a_live_unit_allocates_nothing_while_a_capture_records(monkeypatch):
    from core.inference import diffusion_cuda_graph as graphs

    @contextlib.contextmanager
    def recording(index):
        yield False

    monkeypatch.setattr(graphs, "hold_device_off_capture", recording)

    def no_alloc(*a, **k):
        raise AssertionError("allocated during a capture")

    monkeypatch.setattr(torch, "randint", no_alloc)
    key = g8._cut_key(28672, 5376, torch.float32, torch.bfloat16, False)
    monkeypatch.setitem(g8._CUT_PLAN, 0, {key: {4160: 0}})
    assert g8._live_unit(0, key, 2112) is False
