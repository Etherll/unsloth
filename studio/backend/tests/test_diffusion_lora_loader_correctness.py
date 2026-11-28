# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LoRA files on the families diffusers converts (Z-Image, Qwen-Image, Qwen-Image-2.1, FLUX.1, FLUX.3): a file that maps
nothing raises, alpha values diffusers cannot use raise instead of hanging, and FLUX.3 BFL / Kohya and Qwen-Image-2.1
musubi files reach every layer they name. Files that loaded correctly before keep the same adapter tensors."""

from __future__ import annotations

import signal
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("safetensors")
diffusers = pytest.importorskip("diffusers")

from safetensors.torch import save_file  # noqa: E402

from core.inference import diffusion_lora_files as lf  # noqa: E402

RANK = 4
FLUX3 = "flux.3-klein"


def _pipe(mixin_name, model):
    mixin = getattr(diffusers.loaders, mixin_name, None)
    if mixin is None:
        pytest.skip(f"this diffusers has no {mixin_name}")

    class Pipe(mixin):
        hf_device_map = None
        text_encoder = None

        @property
        def components(self):
            return {"transformer": self.transformer}

    pipe = Pipe()
    pipe.transformer = model
    return pipe


def _model(name, **kwargs):
    cls = getattr(diffusers, name, None)
    if cls is None:
        pytest.skip(f"this diffusers has no {name}")
    torch.manual_seed(0)
    return cls(**kwargs)


def zimage():
    return _pipe(
        "ZImageLoraLoaderMixin",
        _model(
            "ZImageTransformer2DModel",
            dim = 32,
            n_layers = 1,
            n_refiner_layers = 1,
            n_heads = 2,
            n_kv_heads = 2,
            cap_feat_dim = 16,
            in_channels = 4,
            axes_dims = [4, 6, 6],
            axes_lens = [16, 16, 16],
        ),
    )


def qwen():
    return _pipe(
        "QwenImageLoraLoaderMixin",
        _model(
            "QwenImageTransformer2DModel",
            num_layers = 1,
            attention_head_dim = 8,
            num_attention_heads = 2,
            joint_attention_dim = 16,
            in_channels = 4,
            out_channels = 4,
            axes_dims_rope = (2, 2, 4),
        ),
    )


def qwen21():
    return _pipe(
        "QwenImageLoraLoaderMixin",
        _model(
            "QwenImage21Transformer2DModel",
            num_layers = 1,
            attention_head_dim = 8,
            num_attention_heads = 2,
            context_in_dim = 16,
            in_channels = 4,
            out_channels = 4,
            axes_dims_rope = (2, 2, 4),
        ),
    )


def flux3():
    return _pipe(
        "Flux3LoraLoaderMixin",
        _model(
            "Flux3Transformer2DModel",
            in_channels = 4,
            cond_in_channels = 4,
            num_layers = 1,
            num_single_layers = 1,
            attention_head_dim = 8,
            num_attention_heads = 2,
            joint_attention_dim = 16,
            timestep_channels = 16,
            axes_dims_rope = (2, 2, 2, 2),
        ),
    )


def flux1():
    return _pipe(
        "FluxLoraLoaderMixin",
        _model(
            "FluxTransformer2DModel",
            patch_size = 1,
            in_channels = 4,
            num_layers = 1,
            num_single_layers = 1,
            attention_head_dim = 8,
            num_attention_heads = 2,
            joint_attention_dim = 16,
            pooled_projection_dim = 8,
            axes_dims_rope = (2, 2, 4),
        ),
    )


def _pair(in_f, out_f, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(RANK, in_f, generator = g), torch.randn(out_f, RANK, generator = g)


def _save(
    tmp_path,
    state,
    name = "lora.safetensors",
):
    path = tmp_path / name
    save_file({k: v.contiguous() for k, v in state.items()}, str(path))
    return str(path)


def _adapters(pipe, name = "x"):
    """``{module: (lora_A, lora_B, scaling)}`` for every layer PEFT wrapped under ``name``."""
    out = {}
    for module_name, module in pipe.transformer.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if isinstance(lora_a, torch.nn.ModuleDict) and name in lora_a:
            out[module_name] = (
                lora_a[name].weight.detach().clone(),
                module.lora_B[name].weight.detach().clone(),
                module.scaling[name],
            )
    return out


def _assert_same(got, want):
    assert sorted(got) == sorted(want)
    for module in want:
        a0, b0, s0 = want[module]
        a1, b1, s1 = got[module]
        assert torch.equal(a1, a0), module
        assert torch.equal(b1, b0), module
        assert s1 == s0, module


class _Deadline:
    """Raises in the test instead of hanging the run when a load never returns."""

    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        def _expired(*_args):
            raise TimeoutError(f"the load did not return within {self.seconds} s")

        self.previous = signal.signal(signal.SIGALRM, _expired)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous)
        return False


# ------------------------------------------------------------------- a file that maps nothing raises


_FOREIGN = {
    "z-image": (zimage, 32),
    "qwen-image": (qwen, 16),
    "qwen-image-2.1": (qwen21, 16),
    FLUX3: (flux3, 16),
}


@pytest.mark.parametrize("family", sorted(_FOREIGN))
def test_a_file_naming_no_layer_raises(tmp_path, family):
    """Keys outside every component diffusers loads into were dropped with only a log line: no adapter attached."""
    factory, width = _FOREIGN[family]
    a, b = _pair(width, width, 1)
    path = _save(tmp_path, {"unet.down.0.lora_A.weight": a, "unet.down.0.lora_B.weight": b})
    pipe = factory()
    with pytest.raises(ValueError, match = "none of its LoRA keys"):
        lf.load_adapter(pipe, path, "x", family)
    assert _adapters(pipe) == {}


def _diffusers_file(family):
    if family == "z-image":
        return zimage, {
            "transformer.layers.0.attention.to_q": (32, 32),
            "transformer.layers.0.feed_forward.w1": (32, None),
        }
    if family == "qwen-image":
        return qwen, {
            "transformer.transformer_blocks.0.attn.to_q": (16, 16),
            "transformer.transformer_blocks.0.img_mlp.net.0.proj": (16, None),
        }
    if family == "qwen-image-2.1":
        return qwen21, {
            "transformer.transformer_blocks.0.attn.to_q": (16, 16),
            "transformer.transformer_blocks.0.img_mlp.gate_layer": (16, 48),
        }
    return flux3, {
        "transformer.single_transformer_blocks.0.attn.to_q": (16, 16),
        "transformer.norm_out.linear": (16, 32),
    }


@pytest.mark.parametrize("family", sorted(_FOREIGN))
def test_a_diffusers_file_loads_as_diffusers_loads_it(tmp_path, family):
    factory, modules = _diffusers_file(family)
    probe = factory().transformer
    state = {}
    for i, (stem, (in_f, out_f)) in enumerate(modules.items()):
        out_f = out_f or int(probe.get_submodule(stem[len("transformer.") :]).weight.shape[0])
        a, b = _pair(in_f, out_f, 10 + i)
        state[f"{stem}.lora_A.weight"], state[f"{stem}.lora_B.weight"] = a, b
    path = _save(tmp_path, state)
    want = factory()
    want.load_lora_weights(path, adapter_name = "x")
    got = factory()
    lf.load_adapter(got, path, "x", family)
    _assert_same(_adapters(got), _adapters(want))
    assert len(_adapters(got)) == len(modules)


# ------------------------------------------------------------------- alpha


_KOHYA_Q = {
    "z-image": (zimage, "lora_unet_layers_0_attention_to_q", 32),
    "qwen-image": (qwen, "lora_unet_transformer_blocks_0_attn_to_q", 16),
}


@pytest.mark.parametrize("family", sorted(_KOHYA_Q))
@pytest.mark.parametrize("alpha", [-4.0, float("nan")])
def test_an_unusable_alpha_raises_at_once(tmp_path, family, alpha):
    """diffusers' converter halves a negative scale forever."""
    factory, stem, width = _KOHYA_Q[family]
    a, b = _pair(width, width, 2)
    path = _save(
        tmp_path,
        {
            f"{stem}.lora_down.weight": a,
            f"{stem}.lora_up.weight": b,
            f"{stem}.alpha": torch.tensor(alpha),
        },
    )
    pipe = factory()
    start = time.perf_counter()
    with _Deadline(10), pytest.raises(ValueError, match = "alpha"):
        lf.load_adapter(pipe, path, "x", family)
    assert time.perf_counter() - start < 5.0
    assert _adapters(pipe) == {}


@pytest.mark.parametrize("family", sorted(_KOHYA_Q))
def test_a_positive_alpha_loads_as_diffusers_loads_it(tmp_path, family):
    factory, stem, width = _KOHYA_Q[family]
    a, b = _pair(width, width, 3)
    path = _save(
        tmp_path,
        {
            f"{stem}.lora_down.weight": a,
            f"{stem}.lora_up.weight": b,
            f"{stem}.alpha": torch.tensor(2.0),
        },
    )
    want = factory()
    want.load_lora_weights(path, adapter_name = "x")
    got = factory()
    lf.load_adapter(got, path, "x", family)
    _assert_same(_adapters(got), _adapters(want))
    assert len(_adapters(got)) == 1


def _flux1_mixture(alpha):
    stem = "lora_transformer_transformer_blocks_0_attn_to_q"
    a, b = _pair(16, 16, 4)
    state = {f"{stem}.lora_down.weight": a, f"{stem}.lora_up.weight": b}
    if alpha is not None:
        state[f"{stem}.alpha"] = torch.tensor(float(alpha))
    return state


def test_a_kohya_file_without_alpha_loads_at_alpha_equal_rank(tmp_path):
    """diffusers' FLUX.1 mixture converter pops ``.alpha`` without a default: a file that omits it raised KeyError.
    The trainer convention for a missing alpha is alpha = rank (scale 1)."""
    path = _save(tmp_path, _flux1_mixture(None), "no_alpha.safetensors")
    ref = _save(tmp_path, _flux1_mixture(RANK), "alpha_rank.safetensors")
    want = flux1()
    want.load_lora_weights(ref, adapter_name = "x")
    got = flux1()
    lf.load_adapter(got, path, "x", "flux.1")
    _assert_same(_adapters(got), _adapters(want))
    assert len(_adapters(got)) == 1


def test_a_kohya_file_with_alpha_loads_by_path(tmp_path, monkeypatch):
    """Every module carries its alpha: the file goes to diffusers untouched, as before."""
    path = _save(tmp_path, _flux1_mixture(2.0))
    seen = []
    real = lf._load_lora_weights
    monkeypatch.setattr(lf, "_load_lora_weights", lambda p, s, n: (seen.append(s), real(p, s, n)))
    want = flux1()
    want.load_lora_weights(path, adapter_name = "x")
    got = flux1()
    lf.load_adapter(got, path, "x", "flux.1")
    assert seen == [path]
    _assert_same(_adapters(got), _adapters(want))


# ------------------------------------------------------------------- FLUX.3 BFL / Kohya


# BFL checkpoint module names of the tiny FLUX.3 model, one per diffusers Linear.
_FLUX3_BFL = (
    "time_in.in_layer",
    "time_in.out_layer",
    "emb_in.image",
    "emb_in.image_cond",
    "txt_in",
    "early_stream_modulations.image.lin",
    "early_stream_modulations.image_cond.lin",
    "early_stream_modulations.txt.lin",
    "single_stream_modulations.image.lin",
    "single_stream_modulations.image_cond.lin",
    "single_stream_modulations.txt.lin",
    "final_layer.image.linear",
    "final_layer.image.adaLN_modulation.1",
) + tuple(
    f"{block}.{leaf}"
    for block in (
        "content_mode_blocks.image.0",
        "content_mode_blocks.image_cond.0",
        "txt_mode_blocks.0",
        "single_blocks.0",
    )
    for leaf in ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")
)


def test_flux3_bfl_names_map_as_the_diffusers_checkpoint_converter_maps_them():
    from diffusers.loaders.single_file_utils import (
        convert_flux3_transformer_checkpoint_to_diffusers,
    )

    linears = {n for n, m in flux3().transformer.named_modules() if isinstance(m, torch.nn.Linear)}
    mapped = set()
    for stem in _FLUX3_BFL:
        (key,) = convert_flux3_transformer_checkpoint_to_diffusers(
            {stem + ".weight": torch.zeros(2, 1)}
        )
        target, swap = lf.flux3_module_name(stem)
        assert target == key[: -len(".weight")], stem
        assert swap == stem.endswith("adaLN_modulation.1"), stem
        mapped.add(target)
    assert mapped == linears


def _flux3_adapter():
    pipe = flux3()
    out = {}
    for i, stem in enumerate(_FLUX3_BFL):
        target, _ = lf.flux3_module_name(stem)
        weight = pipe.transformer.get_submodule(target).weight
        out[stem] = _pair(int(weight.shape[1]), int(weight.shape[0]), 100 + i)
    return out


def _flux3_expected(pairs, scale = 1.0):
    """The adapter in diffusers names: B scaled, and norm_out's [shift; scale] rows turned to [scale; shift]."""
    out = {}
    for stem, (a, b) in pairs.items():
        target, swap = lf.flux3_module_name(stem)
        b = (b.float() * scale).to(b.dtype) if scale != 1.0 else b
        if swap:
            shift, sc = b.chunk(2, dim = 0)
            b = torch.cat([sc, shift], dim = 0)
        out[f"transformer.{target}.lora_A.weight"] = a
        out[f"transformer.{target}.lora_B.weight"] = b
    return out


def _load_expected(tmp_path, state):
    path = _save(tmp_path, state, "expected.safetensors")
    pipe = flux3()
    pipe.load_lora_weights(path, adapter_name = "x")
    return _adapters(pipe)


@pytest.mark.parametrize(
    "layout", ["bfl", "bfl_kohya_suffix", "kohya", "kohya_no_alpha", "kohya_zero_alpha"]
)
def test_flux3_non_diffusers_files_reach_every_layer(tmp_path, layout):
    """BFL ``diffusion_model.`` files lost every layer diffusers' converter does not rename (norm_out among them, unswapped),
    ``lora_down`` / ``.alpha`` ones were refused and Kohya ``lora_unet_`` ones attached nothing."""
    pairs = _flux3_adapter()
    alpha = 2.0
    state = {}
    for stem, (a, b) in pairs.items():
        if layout == "bfl":
            state[f"diffusion_model.{stem}.lora_A.weight"] = a
            state[f"diffusion_model.{stem}.lora_B.weight"] = b
            continue
        lead = (
            f"diffusion_model.{stem}"
            if layout == "bfl_kohya_suffix"
            else "lora_unet_" + stem.replace(".", "_")
        )
        state[f"{lead}.lora_down.weight"] = a
        state[f"{lead}.lora_up.weight"] = b
        if layout == "kohya_zero_alpha":
            state[f"{lead}.alpha"] = torch.tensor(0.0)  # kohya reads 0 as alpha = rank
        elif layout != "kohya_no_alpha":
            state[f"{lead}.alpha"] = torch.tensor(alpha)
    path = _save(tmp_path, state)
    scale = 1.0 if layout in ("bfl", "kohya_no_alpha", "kohya_zero_alpha") else alpha / RANK
    want = _load_expected(tmp_path, _flux3_expected(pairs, scale))
    got = flux3()
    lf.load_adapter(got, path, "x", FLUX3)
    _assert_same(_adapters(got), want)
    assert len(want) == len(_FLUX3_BFL)


def test_flux3_bfl_file_diffusers_already_maps_loads_by_path(tmp_path, monkeypatch):
    pairs = {k: v for k, v in _flux3_adapter().items() if k.startswith("single_blocks.")}
    state = {}
    for stem, (a, b) in pairs.items():
        state[f"diffusion_model.{stem}.lora_A.weight"] = a
        state[f"diffusion_model.{stem}.lora_B.weight"] = b
    path = _save(tmp_path, state)
    seen = []
    real = lf._load_lora_weights
    monkeypatch.setattr(lf, "_load_lora_weights", lambda p, s, n: (seen.append(s), real(p, s, n)))
    want = flux3()
    want.load_lora_weights(path, adapter_name = "x")
    got = flux3()
    lf.load_adapter(got, path, "x", FLUX3)
    assert seen == [path]
    _assert_same(_adapters(got), _adapters(want))


def test_flux3_kohya_name_of_no_layer_raises(tmp_path):
    a, b = _pair(16, 16, 5)
    path = _save(
        tmp_path,
        {
            "lora_unet_single_blocks_7_q_proj.lora_down.weight": a,
            "lora_unet_single_blocks_7_q_proj.lora_up.weight": b,
        },
    )
    with pytest.raises(ValueError, match = "single_blocks_7_q_proj"):
        lf.load_adapter(flux3(), path, "x", FLUX3)


# ------------------------------------------------------------------- Qwen-Image-2.1 musubi / Kohya


_QWEN21_MODULES = (
    ("transformer_blocks.0.attn.to_q", 16, 16),
    ("transformer_blocks.0.attn.to_out.0", 16, 16),
    ("transformer_blocks.0.img_mlp.gate_layer", 16, 48),
    ("transformer_blocks.0.img_mlp.proj", 16, 48),
    ("transformer_blocks.0.img_mlp.out", 48, 16),
    ("txt_in.in_layer", None, None),
    ("modulation.1", None, None),
)


def _qwen21_pairs():
    model = qwen21().transformer
    out = {}
    for i, (stem, in_f, out_f) in enumerate(_QWEN21_MODULES):
        weight = model.get_submodule(stem).weight
        out[stem] = _pair(int(weight.shape[1]), int(weight.shape[0]), 200 + i)
    return out


def _kohya_state(pairs, lead, alpha):
    state = {}
    for stem, (a, b) in pairs.items():
        name = lead(stem)
        state[f"{name}.lora_down.weight"] = a
        state[f"{name}.lora_up.weight"] = b
        state[f"{name}.alpha"] = torch.tensor(alpha)
    return state


def test_qwen21_musubi_file_reaches_gate_layer(tmp_path):
    """diffusers splits ``img_mlp_gate_layer`` into ``img_mlp.gate.layer`` (and ``txt_in_in_layer`` / ``modulation_1``
    into nothing), so those layers were dropped with a warning while the rest loaded."""
    pairs = _qwen21_pairs()
    path = _save(tmp_path, _kohya_state(pairs, lambda s: "lora_unet_" + s.replace(".", "_"), 2.0))
    ref = _save(
        tmp_path, _kohya_state(pairs, lambda s: "diffusion_model." + s, 2.0), "dotted.safetensors"
    )
    want = qwen21()
    want.load_lora_weights(ref, adapter_name = "x")
    assert len(_adapters(want)) == len(pairs)
    got = qwen21()
    lf.load_adapter(got, path, "x", "qwen-image-2.1")
    _assert_same(_adapters(got), _adapters(want))


def test_qwen21_kohya_file_diffusers_already_maps_loads_by_path(tmp_path, monkeypatch):
    pairs = {k: v for k, v in _qwen21_pairs().items() if ".attn." in k}
    path = _save(tmp_path, _kohya_state(pairs, lambda s: "lora_unet_" + s.replace(".", "_"), 2.0))
    seen = []
    real = lf._load_lora_weights
    monkeypatch.setattr(lf, "_load_lora_weights", lambda p, s, n: (seen.append(s), real(p, s, n)))
    want = qwen21()
    want.load_lora_weights(path, adapter_name = "x")
    got = qwen21()
    lf.load_adapter(got, path, "x", "qwen-image-2.1")
    assert seen == [path]
    _assert_same(_adapters(got), _adapters(want))
    assert len(_adapters(got)) == len(pairs)


@pytest.mark.parametrize("family", sorted(_KOHYA_Q))
def test_a_zero_alpha_loads_as_kohya_reads_it(tmp_path, family):
    """kohya reads alpha 0 as alpha = rank: the adapter equals the one from an explicit alpha = rank."""
    factory, stem, width = _KOHYA_Q[family]
    a, b = _pair(width, width, 2)
    loaded = []
    for alpha in (0.0, float(a.shape[0])):
        path = _save(
            tmp_path,
            {
                f"{stem}.lora_down.weight": a,
                f"{stem}.lora_up.weight": b,
                f"{stem}.alpha": torch.tensor(alpha),
            },
            name = f"lora_{int(alpha)}.safetensors",
        )
        pipe = factory()
        lf.load_adapter(pipe, path, "x", family)
        loaded.append(_adapters(pipe))
    assert loaded[0] and loaded[0].keys() == loaded[1].keys()
    for k, (a0, b0, s0) in loaded[0].items():
        a1, b1, s1 = loaded[1][k]
        assert torch.equal(a0, a1) and torch.equal(b0, b1) and s0 == s1
