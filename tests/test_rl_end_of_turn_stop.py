# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO / RLOO rollouts stop on the chat template's end-of-turn token on TRL <= 1.19.

A Qwen base checkpoint stops only on <|endoftext|> while ChatML ends turns with <|im_end|>, so without the
end-of-turn stop every rollout decodes to max_completion_length. CPU only; the helpers are lifted from
unsloth/models/rl_replacements.py via ast so no GPU import is needed.
"""

from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path

import pytest
import torch


SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
NAMES = (
    "_unsloth_chat_template_end_of_turn_id",
    "_unsloth_rl_end_of_turn_stop",
    "rl_trainer_end_of_turn_stop",
    "grpo_update_SamplingParams",
)
TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
CHATML = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _load():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"), filename = str(SOURCE_PATH))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    found = {n.name for n in nodes}
    assert found == set(NAMES), f"missing helpers: {set(NAMES) - found}"
    namespace = {"inspect": inspect}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace


@pytest.fixture(scope = "module")
def helpers():
    return _load()


@pytest.fixture(scope = "module")
def tiny_tokenizer():
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(TINY)
    except Exception as exc:  # offline runner without the tiny repo cached
        pytest.skip(f"tiny tokenizer unavailable: {exc}")
    return tokenizer


def _qwen_base_tokenizer(tokenizer):
    """The upstream Qwen base layout: EOS <|endoftext|>, ChatML turns ending in <|im_end|>."""
    import copy

    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.chat_template = CHATML
    return tokenizer


_TRL_SOURCES = {
    # TRL 1.15 - 1.19: stops and masks come from model.generation_config
    "aware": (
        "class {name}:\n"
        "    def __init__(self, model):\n"
        "        self.eos_token_ids = list(model.generation_config.eos_token_id)\n"
    ),
    # TRL < 1.15: tokenizer.eos_token_id alone
    "old": (
        "class {name}:\n"
        "    def __init__(self, tokenizer):\n"
        "        self.eos_token_id = tokenizer.eos_token_id\n"
    ),
}
_TRL_SOURCES["native"] = _TRL_SOURCES["aware"] + (
    "def get_chat_template_end_of_turn_token_id(processing_class, chat_template = None):\n"
    "    return None\n"
)


def _fake_trl(
    monkeypatch,
    tmp_path,
    kind,
    file = "grpo_trainer",
    name = "GRPOTrainer",
):
    """A trl.trainer.<file> module backed by a real file, so inspect.getsource works on it."""
    import importlib.util

    path = tmp_path / f"fake_{file}_{kind}.py"
    path.write_text(_TRL_SOURCES[kind].format(name = name), encoding = "utf-8")
    spec = importlib.util.spec_from_file_location(f"trl.trainer.{file}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Unsloth rebinds the module's trainer class to its own subclass; detection must not depend on it.
    setattr(module, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, f"trl.trainer.{file}", module)


def _model(eos):
    from transformers import GenerationConfig
    return types.SimpleNamespace(generation_config = GenerationConfig(eos_token_id = eos))


def _args(
    use_vllm = False,
    generation_kwargs = None,
    vllm_sampling_params = None,
):
    return types.SimpleNamespace(
        use_vllm = use_vllm,
        generation_kwargs = generation_kwargs,
        vllm_sampling_params = vllm_sampling_params,
    )


CONV = [{"prompt": [{"role": "user", "content": "hi"}]}]
PLAIN = [{"prompt": "hi"}]


def test_end_of_turn_id_matches_template(helpers, tiny_tokenizer):
    tokenizer = _qwen_base_tokenizer(tiny_tokenizer)
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert tokenizer.eos_token_id != eot
    assert helpers["_unsloth_chat_template_end_of_turn_id"](tokenizer) == eot
    tokenizer.chat_template = None
    assert helpers["_unsloth_chat_template_end_of_turn_id"](tokenizer) is None


def _trl_hf_stops(model, tokenizer):
    """TRL 1.15 - 1.19 GRPO / RLOO: HF stop set = tokenizer EOS + model.generation_config EOS ids."""
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    return eos if tokenizer.eos_token_id in eos else [tokenizer.eos_token_id, *eos]


def _decoded_tokens(
    stops,
    eot,
    max_new_tokens = 24,
):
    """Greedy decode with a tiny Qwen2 whose head always picks the end-of-turn token."""
    from transformers import AutoModelForCausalLM

    try:
        lm = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype = torch.float32)
    except Exception as exc:
        pytest.skip(f"tiny model unavailable: {exc}")
    head = torch.nn.Linear(lm.config.hidden_size, lm.config.vocab_size, bias = True)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[eot] = 10.0
    lm.lm_head = head
    out = lm.generate(
        torch.tensor([[1, 2, 3]]),
        attention_mask = torch.ones(1, 3, dtype = torch.long),
        max_new_tokens = max_new_tokens,
        do_sample = False,
        eos_token_id = stops,
        pad_token_id = stops[0],
    )
    return out.shape[1] - 3


def test_hf_rollout_stops_on_end_of_turn(helpers, tiny_tokenizer, monkeypatch, tmp_path):
    tokenizer = _qwen_base_tokenizer(tiny_tokenizer)
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos = tokenizer.eos_token_id
    _fake_trl(monkeypatch, tmp_path, "aware")
    model = _model(eos)
    before = _decoded_tokens(_trl_hf_stops(model, tokenizer), eot)
    assert before == 24  # keeps decoding past <|im_end|> to the cap
    args = _args()
    added = helpers["_unsloth_rl_end_of_turn_stop"]("grpo_trainer", model, tokenizer, CONV, args)
    assert added == eot
    assert model.generation_config.eos_token_id == [eos, eot]
    assert args.generation_kwargs is None  # HF path: no vLLM-only kwargs leak into GenerationConfig
    assert _decoded_tokens(_trl_hf_stops(model, tokenizer), eot) == 1
    # Idempotent: a second trainer on the same model adds nothing.
    assert (
        helpers["_unsloth_rl_end_of_turn_stop"]("grpo_trainer", model, tokenizer, CONV, args)
        is None
    )
    assert model.generation_config.eos_token_id == [eos, eot]


def test_vllm_stop_token_ids_merge_user_stops(helpers, tiny_tokenizer, monkeypatch, tmp_path):
    tokenizer = _qwen_base_tokenizer(tiny_tokenizer)
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos = tokenizer.eos_token_id
    _fake_trl(monkeypatch, tmp_path, "aware", file = "rloo_trainer", name = "RLOOTrainer")
    model = _model(eos)
    args = _args(use_vllm = True, generation_kwargs = {"stop_token_ids": [7], "top_k": 5})
    user_kwargs = args.generation_kwargs
    helpers["_unsloth_rl_end_of_turn_stop"]("rloo_trainer", model, tokenizer, CONV, args)
    assert args.generation_kwargs == {"stop_token_ids": [7, eot], "top_k": 5}
    assert user_kwargs == {"stop_token_ids": [7], "top_k": 5}  # caller's dict untouched


@pytest.mark.parametrize(
    "case",
    [
        "native",
        "old",
        "plain",
        "off",
        "no_template",
        "already",
        "tokenizer_eos",
        "ignore_eos_kwargs",
        "ignore_eos_sampling_params",
    ],
)
def test_no_op_cases(helpers, tiny_tokenizer, monkeypatch, tmp_path, case):
    tokenizer = _qwen_base_tokenizer(tiny_tokenizer)
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos = tokenizer.eos_token_id
    data = CONV
    gen_eos = eos
    if case == "native":  # a TRL that already adds it itself
        _fake_trl(monkeypatch, tmp_path, "native")
    elif case == "old":  # TRL < 1.15 stops on tokenizer.eos_token_id alone
        _fake_trl(monkeypatch, tmp_path, "old")
    else:
        _fake_trl(monkeypatch, tmp_path, "aware")
    if case == "plain":
        data = PLAIN
    if case == "off":
        monkeypatch.setenv("UNSLOTH_RL_END_OF_TURN_STOP", "0")
    if case == "no_template":
        tokenizer.chat_template = None
    if case == "already":
        gen_eos = [eos, eot]
    if case == "tokenizer_eos":  # get_chat_template(map_eos_token = True): TRL already stops on it
        tokenizer.eos_token = "<|im_end|>"
    model = _model(gen_eos)
    # HF path for these two: TRL already stops on the id there (vLLM is covered below).
    args = _args(use_vllm = case not in ("already", "tokenizer_eos"))
    if case == "ignore_eos_kwargs":
        args.generation_kwargs = {"ignore_eos": True}
    if case == "ignore_eos_sampling_params":
        args.vllm_sampling_params = types.SimpleNamespace(ignore_eos = True)
    user_kwargs = args.generation_kwargs
    assert (
        helpers["_unsloth_rl_end_of_turn_stop"]("grpo_trainer", model, tokenizer, data, args)
        is None
    )
    assert model.generation_config.eos_token_id == gen_eos
    assert args.generation_kwargs is user_kwargs


@pytest.mark.parametrize("case", ["already", "tokenizer_eos"])
def test_vllm_stops_even_when_in_memory_config_has_it(
    helpers, tiny_tokenizer, monkeypatch, tmp_path, case
):
    """vLLM reads the checkpoint's generation_config.json, not the in-memory one TRL masks with."""
    tokenizer = _qwen_base_tokenizer(tiny_tokenizer)
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos = tokenizer.eos_token_id
    _fake_trl(monkeypatch, tmp_path, "aware")
    gen_eos = [eos, eot] if case == "already" else eos
    if case == "tokenizer_eos":
        tokenizer.eos_token = "<|im_end|>"
    model = _model(gen_eos)
    args = _args(use_vllm = True)
    assert (
        helpers["_unsloth_rl_end_of_turn_stop"]("grpo_trainer", model, tokenizer, CONV, args) == eot
    )
    assert model.generation_config.eos_token_id == gen_eos
    assert args.generation_kwargs == {"stop_token_ids": [eot]}
    # A user stop list that already has it is left alone.
    args = _args(use_vllm = True, generation_kwargs = {"stop_token_ids": [eot]})
    assert (
        helpers["_unsloth_rl_end_of_turn_stop"]("grpo_trainer", model, tokenizer, CONV, args)
        is None
    )


def test_extra_args_code_runs_in_trainer_init(helpers, tiny_tokenizer, monkeypatch, tmp_path):
    tokenizer = _qwen_base_tokenizer(tiny_tokenizer)
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    _fake_trl(monkeypatch, tmp_path, "aware")
    code = helpers["rl_trainer_end_of_turn_stop"]("grpo_trainer")(
        "model = model, processing_class = processing_class, train_dataset = train_dataset, args = args",
        "",
    )
    assert "_unsloth_rl_end_of_turn_stop" in code
    model = _model(tokenizer.eos_token_id)
    scope = {
        "_unsloth_rl_end_of_turn_stop": helpers["_unsloth_rl_end_of_turn_stop"],
        "model": model,
        "processing_class": None,
        "tokenizer": tokenizer,
        "train_dataset": CONV,
        "args": _args(),
    }
    exec(code, scope, scope)
    assert eot in model.generation_config.eos_token_id
    assert helpers["rl_trainer_end_of_turn_stop"]("grpo_trainer")("model = model", "") == ""


def test_registered_for_grpo_and_rloo():
    text = SOURCE_PATH.read_text(encoding = "utf-8")
    assert 'for _eot_trainer_file in ("grpo_trainer", "rloo_trainer"):' in text
    assert (
        "RL_EXTRA_ARGS[_eot_trainer_file].append(rl_trainer_end_of_turn_stop(_eot_trainer_file))"
        in text
    )


def test_user_vllm_sampling_params_keep_trainer_stops(helpers):
    class SamplingParams:
        def __init__(
            self,
            stop_token_ids = None,
            temperature = 1.0,
            max_tokens = 16,
        ):
            self.stop_token_ids = stop_token_ids
            self.temperature = temperature
            self.max_tokens = max_tokens

    user = SamplingParams(stop_token_ids = [7])
    user._set_kwargs = {"stop_token_ids": [7]}
    out = helpers["grpo_update_SamplingParams"](
        SamplingParams, {"stop_token_ids": [1, 2], "max_tokens": 8}, user
    )
    assert out == {"stop_token_ids": [7, 1, 2], "max_tokens": 8}
    out = helpers["grpo_update_SamplingParams"](SamplingParams, {"max_tokens": 8}, user)
    assert out == {"stop_token_ids": [7], "max_tokens": 8}
