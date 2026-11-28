# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 dev guidance in one denoiser forward per step (``UNSLOTH_LTX25_FUSED_GUIDANCE``: 1 on, 0 off, unset Auto).

The CFG call also runs the STG row (via ``perturbation_mask``, exact at 0 and 1) and the modality-isolation row (a2v /
v2a outputs zeroed per row, since ``isolate_modalities`` is one bool per call); the later two calls are answered from
it only when their inputs match. Batch-shaped kernels make rows not bit-identical, so this is drift-gated. Auto fuses
only with a passing G2c verdict for the card class and a persisted speed race that promoted this load and canvas;
otherwise nothing is installed and the render is the unfused one, bit for bit."""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import os
import threading
from typing import Any, Optional

FUSED_GUIDANCE_ENV = "UNSLOTH_LTX25_FUSED_GUIDANCE"
_ON = ("1", "on", "true", "yes", "force")
_OFF = ("0", "off", "false", "no", "disable", "disabled")
_TRANSFORMER_CLS = "LTX2VideoTransformer3DModel"
_STATE_ATTR = "_unsloth_fused_guidance_state"
_KEEP_ATTR = "_unsloth_row_keep"

# Rows taken from the CFG call's uncond half (the pipeline reuses ``chunk(2)[0]`` for these).
_FIRST_HALF = frozenset(
    {"timestep", "sigma", "audio_timestep", "audio_sigma", "video_coords", "audio_coords"}
)
_GUIDANCE_KEYS = ("isolate_modalities", "spatio_temporal_guidance_blocks", "perturbation_mask")

FUSED_STATS: dict[str, Any] = {
    "renders": 0,
    "fused_calls": 0,
    "fused_rows": 0,
    "served_stg": 0,
    "served_modality": 0,
    "misses": 0,
    "fallbacks": 0,
    "oom_fallbacks": 0,
    "declined": None,
}


def fused_guidance_enabled() -> Optional[bool]:
    """True: forced on (``1``), False: forced off (``0``), None: Auto (unset or any other value)."""
    raw = (os.environ.get(FUSED_GUIDANCE_ENV) or "").strip().lower()
    if raw in _ON:
        return True
    if raw in _OFF:
        return False
    return None


def reset_stats() -> None:
    for k in FUSED_STATS:
        FUSED_STATS[k] = None if k == "declined" else 0


_KEEP_CLASSES: dict[type, type] = {}


def _keep_class(base: type) -> type:
    """``base`` with a per-row output keep mask (None = every row, the stock processor)."""
    cls = _KEEP_CLASSES.get(base)
    if cls is not None:
        return cls

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        attention_mask: Any = None,
        query_rotary_emb: Any = None,
        key_rotary_emb: Any = None,
    ) -> Any:
        out = base.__call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            query_rotary_emb,
            key_rotary_emb,
        )
        keep = getattr(self, _KEEP_ATTR, None)
        if keep is not None and keep.shape[0] == out.shape[0]:
            import torch
            out = torch.where(keep, out, torch.zeros((), dtype = out.dtype, device = out.device))
        return out

    cls = type(
        base.__name__,
        (base,),
        {"__call__": __call__, _KEEP_ATTR: None, "_unsloth_row_keep_cls": True},
    )
    cls.__qualname__ = base.__qualname__
    cls.__module__ = base.__module__
    _KEEP_CLASSES[base] = cls
    return cls


def _cross_modal_attentions(transformer: Any) -> list:
    out = []
    for block in getattr(transformer, "transformer_blocks", None) or ():
        for name in ("audio_to_video_attn", "video_to_audio_attn"):
            attn = getattr(block, name, None)
            if attn is None or getattr(attn, "processor", None) is None:
                return []
            out.append(attn)
    return out


def ensure_row_keep_processors(transformer: Any) -> list:
    """Every a2v / v2a processor of ``transformer`` as its keep-mask subclass (state copied, so a pinned attention
    backend stays); re-wraps one a reset replaced. Empty when the layout is not LTX-2's."""
    attns = _cross_modal_attentions(transformer)
    procs = []
    for attn in attns:
        proc = attn.processor
        if not getattr(type(proc), "_unsloth_row_keep_cls", False):
            new = object.__new__(_keep_class(type(proc)))
            new.__dict__.update(getattr(proc, "__dict__", {}))
            setter = getattr(attn, "set_processor", None)
            if callable(setter):
                setter(new)
            else:
                attn.processor = new
            proc = attn.processor
        procs.append(proc)
    return procs


def _set_keep(procs: list, keep: Any) -> None:
    for proc in procs:
        setattr(proc, _KEEP_ATTR, keep)


def _same(a: Any, b: Any) -> bool:
    import torch

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
            return False
        if a.data_ptr() == b.data_ptr() and a.stride() == b.stride():
            return True
        return bool(torch.equal(a, b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 -- uncomparable: not the same call
        return False


def _blocks(value: Any) -> tuple:
    return tuple(int(b) for b in (value or ()))


class _Pending:
    __slots__ = ("kind", "inputs", "output")

    def __init__(self, kind: str, inputs: dict, output: tuple):
        self.kind = kind
        self.inputs = inputs
        self.output = output


class FusedGuidanceState:
    """Per render: the pipeline (its guidance flags), the STG blocks it was called with, and the rows owed."""

    def __init__(
        self,
        runner: Any,
        stg_blocks: tuple,
        logger: Any = None,
        decline_key: Optional[str] = None,
    ):
        self.runner = runner
        self.stg_blocks = stg_blocks
        self.logger = logger
        self.decline_key = decline_key
        self.pending: dict[str, _Pending] = {}
        self.disabled: Optional[str] = None
        self._masks: dict = {}

    def call(self, module: Any, base_call: Any, args: tuple, kwargs: dict) -> Any:
        kind = self._kind(args, kwargs)
        if kind == "main":
            self.pending.clear()
            if self.disabled is None:
                fused = self._fused(module, base_call, kwargs)
                if fused is not None:
                    return fused
            return base_call(module, *args, **kwargs)
        if kind in ("stg", "modality"):
            owed = self.pending.pop(kind, None)
            if owed is not None and self._matches(owed.inputs, kwargs):
                FUSED_STATS["served_stg" if kind == "stg" else "served_modality"] += 1
                return owed.output
            if owed is not None:
                FUSED_STATS["misses"] += 1
        return base_call(module, *args, **kwargs)

    def _kind(self, args: tuple, kwargs: dict) -> Optional[str]:
        if args or kwargs.get("return_dict", True):
            return None
        iso = bool(kwargs.get("isolate_modalities", False))
        blocks = _blocks(kwargs.get("spatio_temporal_guidance_blocks"))
        if kwargs.get("perturbation_mask") is not None:
            return None
        if not iso and not blocks:
            return "main"
        if not iso and blocks:
            return "stg" if blocks == self.stg_blocks else None
        if iso and not blocks:
            return "modality"
        return None

    def _passes(self) -> tuple[bool, bool, bool]:
        r = self.runner
        try:
            return (
                bool(r.do_classifier_free_guidance),
                bool(r.do_spatio_temporal_guidance) and bool(self.stg_blocks),
                bool(r.do_modality_isolation_guidance),
            )
        except Exception:  # noqa: BLE001 -- not the LTX-2 pipeline's guidance flags
            return False, False, False

    def _fused(self, module: Any, base_call: Any, kwargs: dict) -> Any:
        import torch

        cfg, stg, mod = self._passes()
        if not (stg or mod):
            return None
        hs = kwargs.get("hidden_states")
        if not isinstance(hs, torch.Tensor) or hs.ndim < 2:
            return None
        rows = int(hs.shape[0])
        if cfg and rows % 2:
            return None
        n = rows // 2 if cfg else rows
        cond = slice(n, rows) if cfg else slice(0, rows)
        first = slice(0, n)
        extra = int(stg) + int(mod)
        parts: dict = {}
        fused_kwargs: dict = {}
        for key, value in kwargs.items():
            if key in _GUIDANCE_KEYS:
                continue
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and int(value.shape[0]) == rows:
                part = value[first] if (cfg and key in _FIRST_HALF) else value[cond]
                parts[key] = part
                fused_kwargs[key] = torch.cat([value] + [part] * extra, dim = 0)
            else:
                parts[key] = value
                fused_kwargs[key] = value
        total = rows + extra * n
        stg_rows = slice(rows, rows + n) if stg else None
        mod_rows = slice(rows + n * int(stg), total) if mod else None
        procs: list = []
        keep = None
        if mod:
            procs = ensure_row_keep_processors(module)
            if not procs:
                self.disabled = "the denoiser has no a2v / v2a attention processors to mask per row"
                return None
            keep = self._mask("keep", total, mod_rows, hs.device, torch.bool)
        fused_kwargs["isolate_modalities"] = False
        if stg:
            fused_kwargs["spatio_temporal_guidance_blocks"] = list(self.stg_blocks)
            fused_kwargs["perturbation_mask"] = self._mask(
                "stg", total, stg_rows, hs.device, hs.dtype
            )
        else:
            fused_kwargs["spatio_temporal_guidance_blocks"] = None
            fused_kwargs["perturbation_mask"] = None
        _set_keep(procs, keep)
        try:
            out = base_call(module, **fused_kwargs)
        except Exception as exc:  # noqa: BLE001 -- the separate calls still run: a lever, never a failure
            self.disabled = f"{type(exc).__name__}: {exc}"
            FUSED_STATS["fallbacks"] += 1
            if _is_oom(exc):
                FUSED_STATS["oom_fallbacks"] += 1
                if self.decline_key is not None:
                    # Auto stops trying this load and canvas in this process: every later step would OOM again.
                    _OOM_DECLINED[self.decline_key] = self.disabled[:300]
            if self.logger is not None:
                self.logger.warning(
                    "ltx25.fused_guidance: off for this render (%s)", self.disabled[:300]
                )
            with contextlib.suppress(Exception):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return None
        finally:
            _set_keep(procs, None)
        if not isinstance(out, tuple) or len(out) != 2:
            self.disabled = f"unexpected denoiser output {type(out).__name__}"
            return None
        video, audio = out
        FUSED_STATS["fused_calls"] += 1
        FUSED_STATS["fused_rows"] += total
        if stg:
            self.pending["stg"] = _Pending("stg", parts, (video[stg_rows], audio[stg_rows]))
        if mod:
            self.pending["modality"] = _Pending(
                "modality", parts, (video[mod_rows], audio[mod_rows])
            )
        return (video[:rows], audio[:rows])

    def _mask(self, kind: str, total: int, rows: slice, device: Any, dtype: Any) -> Any:
        import torch

        key = (kind, total, rows.start, rows.stop, str(device), dtype)
        mask = self._masks.get(key)
        if mask is None:
            if kind == "keep":
                mask = torch.ones((total, 1, 1), dtype = torch.bool, device = device)
                mask[rows] = False
            else:
                mask = torch.ones((total,), dtype = dtype, device = device)
                mask[rows] = 0
            self._masks[key] = mask
        return mask

    @staticmethod
    def _matches(expected: dict, kwargs: dict) -> bool:
        for key, value in kwargs.items():
            if key in _GUIDANCE_KEYS:
                continue
            if key not in expected or not _same(expected[key], value):
                return False
        return all(k in kwargs or k in _GUIDANCE_KEYS for k in expected)


@contextlib.contextmanager
def dev_guidance(
    runner: Any,
    call: dict,
    logger: Any = None,
    decision: Optional[dict] = None,
    separate: tuple = (),
):
    """Fused guidance or the separate passes with each ``separate`` lever (``(runner, call, logger)`` factory)
    entered; never both. Yields ``"fused"`` or ``"separate"``. An OOM fallback finishes without those levers."""
    with fused_guidance(runner, call, logger, decision = decision) as state:
        if state is not None:
            yield "fused"
            return
        with contextlib.ExitStack() as stack:
            for lever in separate or ():
                stack.enter_context(lever(runner, call, logger))
            yield "separate"


_ROUTED_CLASSES: dict[type, type] = {}


def _routed_class(base: type) -> type:
    """``base`` whose ``__call__`` consults the render's state first (same name, so name checks still match)."""
    cls = _ROUTED_CLASSES.get(base)
    if cls is not None:
        return cls
    base_call = base.__call__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        state = self.__dict__.get(_STATE_ATTR)
        if state is None:
            return base_call(self, *args, **kwargs)
        return state.call(self, base_call, args, kwargs)

    cls = type(base.__name__, (base,), {"__call__": __call__, "_unsloth_fused_guidance_cls": True})
    cls.__qualname__ = base.__qualname__
    cls.__module__ = base.__module__
    _ROUTED_CLASSES[base] = cls
    return cls


def decline_reason(runner: Any, transformer: Any) -> Optional[str]:
    if transformer is None or type(transformer).__name__ != _TRANSFORMER_CLS:
        return f"denoiser is {type(transformer).__name__}, not {_TRANSFORMER_CLS}"
    if getattr(type(transformer), "_unsloth_fused_guidance_cls", False):
        return "already routed"
    for name in (
        "do_classifier_free_guidance",
        "do_spatio_temporal_guidance",
        "do_modality_isolation_guidance",
    ):
        if not hasattr(type(runner), name):
            return f"the pipeline has no {name}"
    if getattr(transformer, "is_cache_enabled", False):
        return "a step cache is on (each guidance pass keeps its own cache context)"
    override = transformer.__dict__.get("forward")
    if override is not None and type(override).__name__.endswith("Router"):
        return f"the denoiser forward is routed by {type(override).__name__}"
    return None


def _stg_blocks(runner: Any, call: dict) -> tuple:
    if "spatio_temporal_guidance_blocks" in call:
        return _blocks(call["spatio_temporal_guidance_blocks"])
    try:
        param = inspect.signature(runner.__call__).parameters.get("spatio_temporal_guidance_blocks")
    except (TypeError, ValueError):
        return ()
    if param is None or param.default is inspect.Parameter.empty:
        return ()
    return _blocks(param.default)


@contextlib.contextmanager
def fused_guidance(
    runner: Any,
    call: dict,
    logger: Any = None,
    enabled: Optional[bool] = None,
    decline_key: Optional[str] = None,
    decision: Optional[dict] = None,
):
    """Without ``decision`` only the env (forced on) decides. Records ``engaged`` / ``fell_back`` on ``decision``."""
    if isinstance(decision, dict):
        enabled = bool(decision.get("on"))
        decline_key = decision.get("key")
    if enabled is None:
        enabled = fused_guidance_enabled() is True
    outcome = decision if isinstance(decision, dict) else {}
    outcome.update(engaged = False, fell_back = False)
    _note_outcome(outcome)
    if not enabled:
        yield None
        return
    transformer = getattr(runner, "transformer", None)
    reason = decline_reason(runner, transformer)
    if reason is not None:
        FUSED_STATS["declined"] = reason
        if logger is not None:
            logger.info("ltx25.fused_guidance: declined (%s)", reason)
        yield None
        return
    base = type(transformer)
    state = FusedGuidanceState(runner, _stg_blocks(runner, call), logger, decline_key)
    FUSED_STATS["renders"] += 1
    calls, falls = FUSED_STATS["fused_calls"], FUSED_STATS["fallbacks"]
    transformer.__class__ = _routed_class(base)
    transformer.__dict__[_STATE_ATTR] = state
    try:
        yield state
    finally:
        transformer.__dict__.pop(_STATE_ATTR, None)
        state.pending.clear()
        if type(transformer) is _ROUTED_CLASSES.get(base):
            transformer.__class__ = base
        outcome.update(
            engaged = FUSED_STATS["fused_calls"] > calls,
            fell_back = FUSED_STATS["fallbacks"] > falls,
        )
        _note_outcome(outcome)


QUALITY_FAMILY = "ltx-2.5-dev-fused-guidance"
QUALITY_PATH = "any"
RACE_KIND = "ltx25_fused_guidance"
_DEV_ATTR = "_unsloth_ltx25_dev"
_OOM_DECLINED: dict[str, str] = {}
_STORE: list = []
LAST_DECISION: dict[str, Any] = {}
_DECISION_LOCK = threading.Lock()

# Measured A100-40 768x512x49 (2688 tokens): +143.5 MiB per extra row; priced at 1.5x plus slack until measured.
_EXTRA_MIB_PER_ROW_KTOK = 53.4
_EXTRA_MARGIN = 1.5
_EXTRA_SLACK_MIB = 128


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001
        pass
    return "out of memory" in str(exc).lower()


def extra_rows_mib(
    width: Any,
    height: Any,
    frames: Any,
    rows: int = 2,
) -> int:
    """Activation headroom the fused call needs above the CFG pair's (``rows`` extra rows) at this canvas."""
    try:
        tokens = (
            max(1, int(width) // 32)
            * max(1, int(height) // 32)
            * ((max(1, int(frames)) - 1) // 8 + 1)
        )
    except (TypeError, ValueError):
        return 0
    import math

    return (
        int(math.ceil(rows * tokens / 1000.0 * _EXTRA_MIB_PER_ROW_KTOK * _EXTRA_MARGIN))
        + _EXTRA_SLACK_MIB
    )


def _store() -> Any:
    if not _STORE:
        from .diffusion_probe_cache import Namespace
        _STORE.append(Namespace("ltx25_fused_guidance", sources = ("video_ltx25_fused_guidance.py",)))
    return _STORE[0]


def _optims(state: Any) -> tuple:
    return tuple(
        sorted(str(o) for o in (getattr(state, "speed_optims", ()) or ()) if o != RACE_KIND)
    )


def load_key(state: Any) -> str:
    """Load fields a speed verdict depends on (card, driver, torch come from the probe fingerprint)."""
    pipe = getattr(state, "pipe", None)
    resolved = getattr(state, "resolved", None) or {}
    revision = ""
    if isinstance(resolved, dict):
        revision = str(
            resolved.get("revision") or resolved.get("sha") or resolved.get("commit") or ""
        )
    try:
        import torch
        cards = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        cards = 0
    parts = (
        str(getattr(state, "repo_id", None) or ""),
        revision,
        str(getattr(state, "gguf_filename", None) or ""),
        str(getattr(getattr(state, "family", None), "name", None) or ""),
        type(getattr(pipe, "transformer", None)).__name__,
        str(getattr(state, "transformer_quant", None) or "bf16"),
        str(getattr(state, "text_encoder_quant", None) or ""),
        str(getattr(state, "dtype", None) or ""),
        str(getattr(state, "speed_mode", None) or ""),
        ",".join(_optims(state)),
        str(getattr(state, "attention_backend", None) or ""),
        str(getattr(state, "offload_policy", None) or ""),
        str(getattr(state, "memory_mode", None) or ""),
        "streamed" if getattr(state, "denoiser_streamed", False) else "resident",
        str(getattr(state, "transformer_cache", None) or ""),
        f"cards{cards}",
    )
    return "|".join(parts)


def canvas_key(width: Any, height: Any, frames: Any, steps: Any, guidance: Any) -> str:
    try:
        g = "" if guidance is None else f"{float(guidance):.6g}"
    except (TypeError, ValueError):
        g = str(guidance)
    return f"{int(width or 0)}x{int(height or 0)}x{int(frames or 0)}|s{int(steps or 0)}|g{g}|t2v"


def verdict_key(state: Any, width: Any, height: Any, frames: Any, steps: Any, guidance: Any) -> str:
    return f"{load_key(state)}#{canvas_key(width, height, frames, steps, guidance)}"


def _card(state: Any) -> Optional[str]:
    from .diffusion_batch_race import card_of
    return card_of(state)


def lookup_speed(
    state: Any, width: Any, height: Any, frames: Any, steps: Any, guidance: Any
) -> Optional[dict]:
    """The persisted race verdict for this load and canvas, None without one (or without a card identity)."""
    card = _card(state)
    if card is None:
        return None
    try:
        store = _store()
        path = store._path()
        if path is None or not path.is_file():
            return None  # no race ever ran here: no fingerprint read
        found = store.get(card, verdict_key(state, width, height, frames, steps, guidance))
        return found if isinstance(found, dict) else None
    except Exception:  # noqa: BLE001 - an unreadable store is no verdict
        return None


def record_speed(
    state: Any, width: Any, height: Any, frames: Any, steps: Any, guidance: Any, verdict: dict
) -> bool:
    card = _card(state)
    if card is None:
        return False
    try:
        return bool(
            _store().put(
                card, verdict_key(state, width, height, frames, steps, guidance), dict(verdict)
            )
        )
    except Exception:  # noqa: BLE001 - a cache write never fails the race
        return False


class _QualityState:
    """What ``diffusion_batch_quality.gate`` reads off a state: the family token and the card class."""

    def __init__(self, device: Optional[str]):
        import types
        self.family = types.SimpleNamespace(name = QUALITY_FAMILY)
        self.device_name = device


def card_class(state: Any) -> Optional[str]:
    from .diffusion_batch_quality import device_class
    try:
        return device_class(state)
    except Exception:  # noqa: BLE001
        return None


def quality_verdict(state: Any) -> Optional[dict]:
    from .diffusion_batch_quality import lookup
    return lookup(QUALITY_FAMILY, card_class(state), QUALITY_PATH)


def gated(verdict: Optional[dict], state: Any) -> Optional[dict]:
    """The race verdict, promoted only with a passing G2c verdict for this card class."""
    if not isinstance(verdict, dict):
        return verdict
    from .diffusion_batch_quality import gate

    return gate(dict(verdict, path = QUALITY_PATH), _QualityState(card_class(state)))


_RACE_ARM: contextvars.ContextVar = contextvars.ContextVar("ltx25_fused_race_arm", default = None)


def _arm_override() -> Optional[bool]:
    return _RACE_ARM.get()


@contextlib.contextmanager
def race_arm(on: bool):
    """Inside the race only: this arm's renders fuse (True) or not (False), whatever the verdicts say."""
    token = _RACE_ARM.set(bool(on))
    try:
        yield
    finally:
        _RACE_ARM.reset(token)


def _decision(
    on: bool,
    mode: str,
    reason: str,
    key: Optional[str] = None,
    **extra: Any,
) -> dict:
    out = {"on": bool(on), "mode": mode, "reason": reason, "key": key, **extra}
    with _DECISION_LOCK:
        LAST_DECISION.clear()
        LAST_DECISION.update({k: v for k, v in out.items() if k != "key"})
    return out


def _note_outcome(decision: dict) -> None:
    """The render's outcome on the last decision (status reads ``engaged`` from the last clip, not the process)."""
    with _DECISION_LOCK:
        if (
            decision
            and decision.get("pipe") is not None
            and decision.get("pipe") == LAST_DECISION.get("pipe")
        ):
            LAST_DECISION.update(
                engaged = bool(decision.get("engaged")), fell_back = bool(decision.get("fell_back"))
            )


def decide(
    state: Any,
    *,
    width: Any,
    height: Any,
    frames: Any,
    steps: Any,
    guidance: Any,
    conditioned: bool = False,
    rows: int = 1,
) -> dict:
    """Whether this request's dev render fuses: ``{"on", "mode" (forced | off | auto | race), "reason", "key"}``.
    Auto reads only stored verdicts: nothing runs on a request."""
    out = _decide(
        state,
        width = width,
        height = height,
        frames = frames,
        steps = steps,
        guidance = guidance,
        conditioned = conditioned,
        rows = rows,
    )
    out["pipe"] = id(getattr(state, "pipe", None))
    with _DECISION_LOCK:
        LAST_DECISION["pipe"] = out["pipe"]
    return out


def _decide(
    state: Any,
    *,
    width: Any,
    height: Any,
    frames: Any,
    steps: Any,
    guidance: Any,
    conditioned: bool = False,
    rows: int = 1,
) -> dict:
    pipe = getattr(state, "pipe", None)
    if getattr(getattr(state, "family", None), "name", None) != "ltx-2.5" or not getattr(
        pipe, _DEV_ATTR, False
    ):
        return _decision(False, "off", "not an LTX-2.5 dev load")
    arm = _arm_override()
    if arm is not None:
        return _decision(
            arm,
            "race",
            "race arm",
            verdict_key(state, width, height, frames, steps, guidance),
            canvas = [int(width), int(height), int(frames), int(steps), guidance],
        )
    forced = fused_guidance_enabled()
    if forced is True:
        return _decision(True, "forced", f"{FUSED_GUIDANCE_ENV}=1")
    if forced is False:
        return _decision(False, "off", f"{FUSED_GUIDANCE_ENV}=0")
    if conditioned:
        return _decision(False, "auto", "image-conditioned renders are not gated")
    if int(rows or 1) != 1:
        return _decision(False, "auto", "batched clips")
    optims = tuple(getattr(state, "speed_optims", ()) or ())
    if "compiled" in optims or getattr(pipe, "_unsloth_cuda_graphs", ()):
        return _decision(
            False,
            "auto",
            "the denoiser is compiled or graphed: a 4-row call would compile on a request",
        )
    quality = quality_verdict(state)
    from .diffusion_batch_quality import passes

    if not passes(quality):
        device = card_class(state)
        return _decision(
            False,
            "auto",
            f"no passing G2c verdict for {device}"
            if quality is None
            else f"G2c verdict for {device} does not pass",
        )
    key = verdict_key(state, width, height, frames, steps, guidance)
    if key in _OOM_DECLINED:
        return _decision(
            False, "auto", f"4-row call ran out of memory here: {_OOM_DECLINED[key]}", key
        )
    speed = lookup_speed(state, width, height, frames, steps, guidance)
    if speed is None:
        return _decision(False, "auto", "no speed race verdict for this load and canvas", key)
    from .diffusion_batch_race import PROMOTE_LB

    try:
        lb = float(speed.get("speedup_lb") or 0.0)
    except (TypeError, ValueError):
        lb = 0.0
    if not speed.get("promoted") or lb < PROMOTE_LB:
        return _decision(
            False, "auto", f"race did not promote: {speed.get('reason')}", key, speedup_lb = lb
        )
    served = gated(speed, state)
    if not served.get("promoted"):
        return _decision(False, "auto", str(served.get("quality_gate")), key, speedup_lb = lb)
    return _decision(True, "auto", f"race speedup lower bound {lb:.3f}", key, speedup_lb = lb)


def decline_for_budget(decision: Optional[dict], reason: str) -> Optional[dict]:
    """The planner found the 4-row peak does not fit beside this load: the render runs unfused."""
    if not isinstance(decision, dict) or not decision.get("on"):
        return decision
    extra = {k: v for k, v in decision.items() if k not in ("on", "mode", "reason", "key")}
    out = _decision(
        False,
        decision.get("mode") or "auto",
        f"4-row peak does not fit: {reason}",
        decision.get("key"),
        **extra,
    )
    return out


def status(state: Any) -> Optional[dict]:
    """The video status block: mode, the last request's decision, the card's G2c verdict and the counters."""
    pipe = getattr(state, "pipe", None)
    if getattr(getattr(state, "family", None), "name", None) != "ltx-2.5" or not getattr(
        pipe, _DEV_ATTR, False
    ):
        return None
    forced = fused_guidance_enabled()
    with _DECISION_LOCK:
        # this load's last request only: a decision made on an earlier load says nothing about this one
        mine = dict(LAST_DECISION) if LAST_DECISION.get("pipe") == id(pipe) else {}
    quality = None
    try:
        q = quality_verdict(state)
        if isinstance(q, dict):
            quality = {k: q.get(k) for k in ("verdict", "n", "upper", "worst_upper", "source")}
    except Exception:  # noqa: BLE001
        quality = None
    return {
        "mode": "on" if forced is True else "off" if forced is False else "auto",
        "engaged": bool(mine.get("engaged")),
        "decision": {k: v for k, v in mine.items() if k != "pipe"} or None,
        "quality": quality,
        "stats": dict(FUSED_STATS),
    }


RACE_PROMPTS = (
    "A red fox trots through fresh snow in a pine forest at dawn; soft crunch of snow and distant birdsong.",
    "Waves crash on a rocky cliff under a lighthouse at sunset; booming surf and seagulls calling.",
    "A steam train crosses a stone viaduct through autumn mountains; the whistle blows and wheels clatter.",
)
RACE_WARM_STEPS = 2


def race_refusal(state: Any) -> Optional[str]:
    """Why this load cannot be raced (Auto would never fuse it), else None."""
    pipe = getattr(state, "pipe", None)
    if state is None or pipe is None:
        return "no video model is loaded"
    if getattr(getattr(state, "family", None), "name", None) != "ltx-2.5" or not getattr(
        pipe, _DEV_ATTR, False
    ):
        return "fused guidance is LTX-2.5 dev only"
    optims = tuple(getattr(state, "speed_optims", ()) or ())
    if "compiled" in optims or getattr(pipe, "_unsloth_cuda_graphs", ()):
        return "the denoiser is compiled or graphed; Auto does not fuse it"
    if _card(state) is None:
        return "no CUDA card identity to key the verdict on"
    return None


def run_race(
    render: Any,
    state: Any,
    *,
    idle: Any,
    steps: Optional[int] = None,
    repeats: int = 3,
    timer: Any = None,
) -> dict:
    """Time interleaved unfused vs fused renders (alternating order, each arm warmed first) and persist the verdict.
    ``idle()`` is checked before every render and before recording: a request arriving stops the race unrecorded."""
    import time

    from .diffusion_batch_race import MIN_REPEATS, RaceRefused, judge

    timer = timer or time.perf_counter
    reason = race_refusal(state)
    if reason is not None:
        raise RaceRefused(reason)
    repeats = max(MIN_REPEATS, int(repeats))

    def _go(on: bool, prompt: str, seed: int, n: int) -> tuple[float, int, int]:
        if not idle():
            raise RaceRefused(
                "a request holds or waits for the engine; the race stopped and recorded nothing"
            )
        calls, falls = FUSED_STATS["fused_calls"], FUSED_STATS["fallbacks"]
        t0 = timer()
        with race_arm(on):
            render(prompt, seed, n)
        return timer() - t0, FUSED_STATS["fused_calls"] - calls, FUSED_STATS["fallbacks"] - falls

    for on in (False, True):
        _go(on, RACE_PROMPTS[0], 0, RACE_WARM_STEPS)
    pairs, engaged, fell_back, canvases = [], True, 0, set()
    for r in range(repeats):
        times: dict = {}
        for on in (False, True) if r % 2 == 0 else (True, False):
            dt, calls, falls = _go(on, RACE_PROMPTS[r % len(RACE_PROMPTS)], 100 + r, steps)
            times[on] = dt
            canvas = LAST_DECISION.get("canvas")
            canvases.add(tuple(canvas) if isinstance(canvas, (list, tuple)) else None)
            if on:
                ran = int(canvas[3]) if isinstance(canvas, (list, tuple)) else -1
                engaged = engaged and calls == ran
                fell_back += falls
            elif calls:
                engaged = False  # the unfused arm fused: the race timed nothing
        pairs.append(([times[False]], times[True]))
    if len(canvases) != 1 or None in canvases:
        raise RaceRefused(
            "the race renders did not reach the fused-guidance decision at one canvas"
        )
    width, height, frames, steps, guidance = next(iter(canvases))
    verdict = judge(pairs)
    if not engaged or fell_back:
        verdict = dict(
            verdict,
            promoted = False,
            reason = f"the fused arm did not run one 4-row call per step (fallbacks {fell_back})",
        )
    verdict.update(
        kind = RACE_KIND,
        path = QUALITY_PATH,
        rows = 4,
        width = int(width),
        height = int(height),
        frames = int(frames),
        steps = int(steps),
        serial_s = [round(p[0][0], 3) for p in pairs],
        fused_s = [round(p[1], 3) for p in pairs],
        t = time.time(),
    )
    if not idle():
        raise RaceRefused("a request arrived during the last render; the race recorded nothing")
    recorded = record_speed(state, width, height, frames, steps, guidance, verdict)
    served = gated(verdict, state) or {}
    return {
        "kind": RACE_KIND,
        "canvas": canvas_key(width, height, frames, steps, guidance),
        "card": card_class(state),
        "recorded": recorded,
        "verdict": dict(
            verdict,
            serves = bool(served.get("promoted")),
            **({"quality_gate": served["quality_gate"]} if served.get("quality_gate") else {}),
        ),
    }
