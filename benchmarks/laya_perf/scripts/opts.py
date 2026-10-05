# Candidate optimizations as monkeypatches, toggled by name from bench/parity/prof.
import torch

PRE, LOAD, POST, TRAINER = {}, {}, {}, {}


def pre(names, D, args):
    for n in names:
        if n in PRE:
            PRE[n](D, args)
        elif n not in LOAD and n not in POST and n not in TRAINER:
            raise SystemExit(f"unknown opt {n}")


def load_kwargs(names, args):
    kw = {}
    for n in names:
        if n in LOAD:
            kw.update(LOAD[n](args))
    return kw


def post(names, model, D, args):
    for n in names:
        if n in POST:
            model = POST[n](model, D, args) or model
    return model


def trainer(names, tr, D, args):
    for n in names:
        if n in TRAINER:
            TRAINER[n](tr, D, args)


def dropout_off(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if isinstance(m, torch.nn.MultiheadAttention):
            m.dropout = 0.0
        if hasattr(m, "attention_dropout") and isinstance(m.attention_dropout, float):
            m.attention_dropout = 0.0
    cfg = getattr(model.encoder, "config", None)
    for k in ("attention_dropout", "embedding_dropout", "mlp_dropout", "classifier_dropout"):
        if cfg is not None and hasattr(cfg, k):
            setattr(cfg, k, 0.0)


def _base_encoder(model):
    enc = model.encoder
    return enc.get_base_model() if hasattr(enc, "get_base_model") else enc


# C1: keep cuDNN SDPA out of the decision forward (head + encoder) and _logits.
def c1_cudnn_guard(D, args):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    compute_loss, logits = D.DecisionTrainer.compute_loss, D._logits

    def guarded_loss(self, *a, **k):
        with sdpa_kernel(backends):
            return compute_loss(self, *a, **k)

    def guarded_logits(*a, **k):
        with sdpa_kernel(backends):
            return logits(*a, **k)

    D.DecisionTrainer.compute_loss = guarded_loss
    D._logits = guarded_logits


PRE["cudnn_guard"] = c1_cudnn_guard


# C1 also covers the backward, where gradient checkpointing reruns the forward (as the perf tree now does).
def c1_cudnn_guard_backward(tr, D, args):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    backward = tr.accelerator.backward

    def guarded(loss, **kw):
        with sdpa_kernel(backends):
            return backward(loss, **kw)

    tr.accelerator.backward = guarded


TRAINER["cudnn_guard"] = c1_cudnn_guard_backward


# C2: pad micro-batch length to a multiple of m.
def _pad_multiple(m):
    def apply(D, args):
        call = D.DecisionDataCollator.__call__

        def padded(self, items):
            batch = call(self, items)
            length = batch["input_ids"].shape[1]
            extra = -(-length // m) * m - length
            if extra:
                batch["input_ids"] = torch.nn.functional.pad(batch["input_ids"], (0, extra), value = self.pad_token_id)
                batch["attention_mask"] = torch.nn.functional.pad(batch["attention_mask"], (0, extra), value = 0)
            return batch

        D.DecisionDataCollator.__call__ = padded
    return apply


PRE["pad8"] = _pad_multiple(8)
PRE["pad64"] = _pad_multiple(64)


# C7: LoRA's frozen encoder stored in the autocast dtype (bf16 here) instead of fp16.
def c7_lora_bf16(args):
    return {"dtype": torch.bfloat16} if getattr(args, "mode", "full") == "lora" and getattr(args, "precision", "bf16") != "fp32" else {}


LOAD["lora_bf16"] = c7_lora_bf16


# C4: regional torch.compile of each ModernBERT encoder layer.
def _compile_layers(dynamic):
    def apply(model, D, args):
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        import os
        if os.environ.get("LP_STRICT") == "1":
            torch._dynamo.config.suppress_errors = False
        for layer in _base_encoder(model).layers:
            layer.compile(dynamic = dynamic)
        return model
    return apply


POST["compile"] = _compile_layers(True)
POST["compile_auto"] = _compile_layers(None)


# C8: _logits over length-sorted batches, inference_mode, results in input order.
def c8_sorted_logits(D, args):
    @torch.inference_mode()
    def _logits(model, items, pad_token_id, batch_size = 16):
        device = next(model.parameters()).device
        amp_dtype = D._amp_dtype(device)
        collate = D.DecisionDataCollator(pad_token_id)
        was_training = model.training
        model.eval()
        order = sorted(range(len(items)), key = lambda i: len(items[i]["input_ids"]))
        out = [None] * len(items)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            chunk = [items[i] for i in index]
            batch = collate(chunk)
            batch.pop("target")
            with torch.autocast(device.type, dtype = amp_dtype, enabled = amp_dtype is not None):
                logits, _ = model(**{k: v.to(device, non_blocking = True) for k, v in batch.items()})
            logits = logits.float().cpu()
            for row, i in enumerate(index):
                out[i] = logits[row, :len(items[i]["markers"])].clone()
        model.train(was_training)
        return out

    D._logits = _logits


PRE["sorted_logits"] = c8_sorted_logits


# C3: encoder unpadding. Valid tokens are packed to [1, T]; attention runs per sequence through
# FA2 varlen / packed SDPA mask / flex_attention, with ModernBERT's symmetric |i-j| <= 64 window on
# local layers; last_hidden_state is scattered back to [B, L, d] (pad rows zero; masked and never read).
_PACK = "_laya_packed"
_ATTN = "laya_packed"


def _packed_attention(module, query, key, value, attention_mask, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    info = kwargs.pop(_PACK, None)
    if info is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    dropout, scale = kwargs.get("dropout", 0.0), kwargs.get("scaling")
    window = kwargs.get("sliding_window")
    backend = info["backend"]
    if backend in ("varlen", "xf") and not (backend == "xf" and not window):
        # torch-native FA2 varlen (aten flash kernels), windowed on local layers; xf uses it for local layers.
        from torch.nn.attention.varlen import varlen_attn
        q, k, v = (x.squeeze(0).transpose(0, 1) for x in (query, key, value))
        size = (window - 1, window - 1) if window else (-1, -1)
        out = varlen_attn(q, k, v, info["cu"], info["cu"], info["max"], info["max"], scale = scale, window_size = size)
        return out.unsqueeze(0), None
    if backend == "xf":
        # Global layers: xformers block-diagonal varlen (cutlass Blackwell FMHA on sm_100 when fbgemm genai is present).
        import xformers.ops as xops
        q, k, v = (x.transpose(1, 2) for x in (query, key, value))
        return xops.memory_efficient_attention(q, k, v, attn_bias = info["xf_bias"], scale = scale), None
    if backend == "flash":
        from flash_attn import flash_attn_varlen_func
        q, k, v = (x.squeeze(0).transpose(0, 1) for x in (query, key, value))
        size = (window - 1, window - 1) if window else (-1, -1)
        out = flash_attn_varlen_func(q, k, v, info["cu"], info["cu"], info["max"], info["max"],
                                     dropout_p = dropout, softmax_scale = scale, causal = False, window_size = size)
        return out.unsqueeze(0), None
    if backend == "flex":
        mask = info["local_bm"] if window else info["global_bm"]
        out = info["flex"](query, key, value, block_mask = mask, scale = scale)
        return out.transpose(1, 2).contiguous(), None
    mask = info["local"] if window else info["global"]
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask = mask, dropout_p = dropout, scale = scale)
    return out.transpose(1, 2).contiguous(), None


_FLEX = {}
_UNPAD_BUCKET = 256


def _packed_forward(self, input_ids = None, attention_mask = None, **kwargs):
    orig = self._laya_orig_forward
    backend = self._laya_backend
    half = torch.is_autocast_enabled() or self.embeddings.tok_embeddings.weight.dtype in (torch.float16, torch.bfloat16)
    if (not self.training or input_ids is None or attention_mask is None or kwargs.get("inputs_embeds") is not None
            or (backend in ("flash", "varlen", "varlen_b", "xf") and not half)):
        return orig(input_ids = input_ids, attention_mask = attention_mask, **kwargs)
    batch, width = input_ids.shape
    keep = attention_mask.bool()
    lengths = keep.sum(1, dtype = torch.int32)
    row_lengths = lengths.tolist()
    if backend not in ("flash", "varlen", "varlen_b", "xf") and sum(row_lengths) == batch * width:
        return orig(input_ids = input_ids, attention_mask = attention_mask, **kwargs)
    index = keep.flatten().nonzero().flatten()
    ids = input_ids.flatten().index_select(0, index).unsqueeze(0)
    pos = torch.arange(width, device = input_ids.device).expand(batch, width).flatten().index_select(0, index)
    cu = torch.nn.functional.pad(lengths.cumsum(0, dtype = torch.int32), (1, 0))
    info = {"backend": backend, "cu": cu, "max": max(row_lengths)}
    window = self.config.sliding_window
    if backend == "xf":
        from xformers.ops.fmha.attn_bias import BlockDiagonalMask
        info["xf_bias"] = BlockDiagonalMask.from_seqlens(row_lengths)
    if backend in ("sdpa", "flex"):
        doc = torch.repeat_interleave(torch.arange(batch, device = ids.device), lengths, output_size = int(sum(row_lengths)))
    if backend == "sdpa":
        same = doc[:, None] == doc[None, :]
        info["global"] = same[None, None]
        info["local"] = (same & ((pos[:, None] - pos[None, :]).abs() <= window))[None, None]
    elif backend == "flex":
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        if "flex" not in _FLEX:
            _FLEX["flex"] = torch.compile(flex_attention, dynamic = True)
        total = int(sum(row_lengths))

        def global_mod(b, h, q, kv):
            return doc[q] == doc[kv]

        def local_mod(b, h, q, kv):
            return (doc[q] == doc[kv]) & ((pos[q] - pos[kv]).abs() <= window)

        info["global_bm"] = create_block_mask(global_mod, None, None, total, total, device = ids.device)
        info["local_bm"] = create_block_mask(local_mod, None, None, total, total, device = ids.device)
        info["flex"] = _FLEX["flex"]
    if backend == "varlen_b":
        total = int(sum(row_lengths))
        extra = -(-(total + 1) // _UNPAD_BUCKET) * _UNPAD_BUCKET - total
        ids = torch.nn.functional.pad(ids, (0, extra), value = self.config.pad_token_id or 0)
        pos = torch.nn.functional.pad(pos, (0, extra))
        cu = torch.cat([cu, cu.new_tensor([total + extra])])
        info.update(backend = "varlen", cu = cu, max = 1 << (max(max(row_lengths), extra) - 1).bit_length())
    out = orig(input_ids = ids, attention_mask = {"full_attention": None, "sliding_attention": None},
               position_ids = pos.unsqueeze(0), **kwargs, **{_PACK: info})
    hidden = out.last_hidden_state.squeeze(0)[:index.numel()]
    out.last_hidden_state = hidden.new_zeros((batch * width, hidden.shape[-1])).index_copy(0, index, hidden).view(batch, width, -1)
    return out


def _unpad(backend):
    def apply(model, D, args):
        from types import MethodType
        from transformers import AttentionInterface, AttentionMaskInterface
        from transformers.masking_utils import sdpa_mask
        AttentionInterface.register(_ATTN, _packed_attention)
        AttentionMaskInterface.register(_ATTN, sdpa_mask)
        base = _base_encoder(model)
        base.config._attn_implementation = _ATTN
        base._laya_backend = backend
        base._laya_orig_forward = base.forward
        base.forward = MethodType(_packed_forward, base)
        return model
    return apply


POST["unpad_flash"] = _unpad("flash")
POST["unpad_sdpa"] = _unpad("sdpa")
POST["unpad_flex"] = _unpad("flex")
POST["unpad_varlen"] = _unpad("varlen")
POST["unpad_xf"] = _unpad("xf")
POST["unpad_vb"] = _unpad("varlen_b")


# C3a: stock transformers FlashAttention-2 on the padded batch (unpads inside attention only, windowed).
def c3a_fa2(model, D, args):
    base = _base_encoder(model)
    base.set_attn_implementation("flash_attention_2")
    return model


POST["fa2"] = c3a_fa2


# C13: ModernBertModel.forward computes rotary cos/sin once per layer (28x) though there are 2 layer types.
# Memoize per forward call; exact (same values).
def c13_rope_once(model, D, args):
    base = _base_encoder(model)
    rotary = base.rotary_emb
    rotary_forward = rotary.forward
    rotary._laya_cache = None

    def cached(x, position_ids, layer_type = None):
        cache = rotary._laya_cache
        if cache is None:
            return rotary_forward(x, position_ids, layer_type)
        key = (layer_type, x.dtype)
        if key not in cache:
            cache[key] = rotary_forward(x, position_ids, layer_type)
        return cache[key]

    rotary.forward = cached
    forward = base.forward

    def wrapped(*a, **k):
        rotary._laya_cache = {}
        try:
            return forward(*a, **k)
        finally:
            rotary._laya_cache = None

    base.forward = wrapped
    return model


POST["rope_once"] = c13_rope_once


# C4 variant: compile each layer's bound forward instead of nn.Module.compile.
def c4_compile_fn(model, D, args):
    import os
    if os.environ.get("LP_STRICT") == "1":
        torch._dynamo.config.suppress_errors = False
    for layer in _base_encoder(model).layers:
        layer.forward = torch.compile(layer.forward, dynamic = True)
    return model


POST["compile_fn"] = c4_compile_fn


# Bench runs are short: let the perf tree's compile gate fire regardless of step count.
def compile_always(D, args):
    if hasattr(D, "COMPILE_MIN_DECISIONS"):
        D.COMPILE_MIN_DECISIONS = 0


def no_compile(D, args):
    if hasattr(D, "COMPILE_MIN_DECISIONS"):
        D.COMPILE_MIN_DECISIONS = 10**12


PRE["compile_always"] = compile_always
PRE["no_compile"] = no_compile


# C14: lean LoRA forward for plain single-adapter LoRA Linear (no dropout / variant / merge):
# base(x) + B(A(x)) * s without PEFT's per-call checks and its input cast to the fp32 adapter dtype.
def _lean_lora_forward(self, x, *args, **kwargs):
    if self.disable_adapters or self.merged or args or kwargs:
        return self._laya_peft_forward(x, *args, **kwargs)
    name = self._laya_adapter
    return self.base_layer(x) + self.lora_B[name](self.lora_A[name](x)) * self.scaling[name]


def c14_lean_lora(model, D, args):
    from types import MethodType
    from peft.tuners.lora.layer import Linear as LoraLinear
    count = 0
    for module in model.encoder.modules():
        if type(module) is not LoraLinear or len(module.lora_A) != 1 or module.lora_variant:
            continue
        name = next(iter(module.lora_A.keys()))
        if not isinstance(module.lora_dropout[name], torch.nn.Identity):
            continue
        module._laya_adapter = name
        module._laya_peft_forward = module.forward
        module.forward = MethodType(_lean_lora_forward, module)
        count += 1
    print("lean_lora modules", count)
    return model


POST["lean_lora"] = c14_lean_lora


# C12: CUDA graphs via compile mode="reduce-overhead" per layer; static shapes, use with pad64 buckets.
def c12_cudagraphs(model, D, args):
    import os
    if os.environ.get("LP_STRICT") == "1":
        torch._dynamo.config.suppress_errors = False
    for layer in _base_encoder(model).layers:
        layer.compile(mode = "reduce-overhead", dynamic = False)
    return model


POST["cudagraphs"] = c12_cudagraphs


# C7b: only the frozen Linear weights (which autocast casts to bf16 for every matmul anyway) stored in bf16;
# embeddings and LayerNorm weights stay fp16, so nothing autocast reads directly loses precision.
def c7b_lora_bf16_linear(model, D, args):
    if getattr(args, "mode", "full") != "lora" or getattr(args, "precision", "bf16") == "fp32":
        return model
    for module in model.encoder.modules():
        if type(module) is torch.nn.Linear and not module.weight.requires_grad and module.weight.dtype == torch.float16:
            module.weight.data = module.weight.data.to(torch.bfloat16)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.bfloat16)
    return model


POST["lora_bf16_linear"] = c7b_lora_bf16_linear


# C14b: lean LoRA + one shared autocast cast of x (base and lora_A both read it) + no multiply when scaling == 1.
def _lean_lora_forward2(self, x, *args, **kwargs):
    if self.disable_adapters or self.merged or args or kwargs:
        return self._laya_peft_forward(x, *args, **kwargs)
    if x.is_cuda and torch.is_autocast_enabled("cuda"):
        x = x.to(torch.get_autocast_dtype("cuda"))
    name = self._laya_adapter
    lora = self.lora_B[name](self.lora_A[name](x))
    scaling = self.scaling[name]
    return self.base_layer(x) + (lora if scaling == 1 else lora * scaling)


def c14b_lean_lora2(model, D, args):
    from types import MethodType
    from peft.tuners.lora.layer import Linear as LoraLinear
    for module in model.encoder.modules():
        if type(module) is not LoraLinear or len(module.lora_A) != 1 or module.lora_variant:
            continue
        name = next(iter(module.lora_A.keys()))
        if not isinstance(module.lora_dropout[name], torch.nn.Identity):
            continue
        module._laya_adapter = name
        module._laya_peft_forward = module.forward
        module.forward = MethodType(_lean_lora_forward2, module)
    return model


POST["lean_lora2"] = c14b_lean_lora2


# C14c: lean LoRA without the shared cast (autocast casts x per matmul, as with PEFT), scaling skip kept.
def _lean_lora_forward3(self, x, *args, **kwargs):
    if self.disable_adapters or self.merged or args or kwargs:
        return self._laya_peft_forward(x, *args, **kwargs)
    name = self._laya_adapter
    if not torch.is_autocast_enabled(x.device.type) and x.dtype != self.lora_A[name].weight.dtype:
        return self._laya_peft_forward(x)
    lora = self.lora_B[name](self.lora_A[name](x))
    scaling = self.scaling[name]
    return self.base_layer(x) + (lora if scaling == 1 else lora * scaling)


def c14c_lean_lora3(model, D, args):
    from types import MethodType
    from peft.tuners.lora.layer import Linear as LoraLinear
    for module in model.encoder.modules():
        if type(module) is not LoraLinear or len(module.lora_A) != 1 or module.lora_variant:
            continue
        name = next(iter(module.lora_A.keys()))
        if not isinstance(module.lora_dropout[name], torch.nn.Identity):
            continue
        module._laya_adapter = name
        module._laya_peft_forward = module.forward
        module.forward = MethodType(_lean_lora_forward3, module)
    return model


POST["lean_lora3"] = c14c_lean_lora3


# The perf tree's own compile path (private SDPA + regional compile), as DecisionTrainer.train runs it.
# parity.py never calls train(), so G1 needs this to exercise the compiled layers.
def trainer_compile(model, D, args):
    fn = getattr(D, "_compile_encoder_layers", None)
    print("trainer_compile", fn(model) if fn else "n/a")
    return model


POST["trainer_compile"] = trainer_compile


# D1: Inductor C++ wrapper for the compiled layers (same kernels, launched from C++ instead of Python).
def cpp_wrapper(D, args):
    import torch._inductor.config as ic
    ic.cpp_wrapper = True


PRE["cpp_wrapper"] = cpp_wrapper


# C12b: CUDA graphs on the compiled layers (cudagraph trees), static shapes from pad64 buckets.
# Outputs made under no_grad (the GC forward) are cloned, since a later replay may reuse their memory;
# each micro-batch starts a new cudagraph step.
def _clone_out(fn):
    from torch.utils._pytree import tree_map

    def call(*a, **k):
        out = fn(*a, **k)
        if not torch.is_grad_enabled():
            out = tree_map(lambda t: t.clone() if isinstance(t, torch.Tensor) else t, out)
        return out
    return call


def cg_layers(D, args):
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
    torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 4096)
    orig = D._compile_encoder_layers
    mode = getattr(args, "cg_mode", None) or "reduce-overhead"

    def comp(model):
        if not orig(model):
            return False
        for layer in model.encoder.layers:
            layer._compiled_call_impl = _clone_out(torch.compile(layer._call_impl, mode = mode, dynamic = False))
        return True

    D._compile_encoder_layers = comp
    compute_loss = D.DecisionTrainer.compute_loss

    def step_loss(self, *a, **k):
        torch.compiler.cudagraph_mark_step_begin()
        return compute_loss(self, *a, **k)

    D.DecisionTrainer.compute_loss = step_loss
    _pad_multiple(64)(D, args)


PRE["cg"] = cg_layers


# C4 static variant without cudagraphs (isolates what the pad64 static shapes alone do).
def static_layers(D, args):
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
    orig = D._compile_encoder_layers

    def comp(model):
        if not orig(model):
            return False
        for layer in model.encoder.layers:
            layer._compiled_call_impl = torch.compile(layer._call_impl, dynamic = False)
        return True

    D._compile_encoder_layers = comp
    _pad_multiple(64)(D, args)


PRE["static64"] = static_layers


# cg needs stable .grad buffers: the first backward would otherwise adopt cudagraph memory as .grad.
def cg_stable_grads(tr, D, args):
    model = tr.model
    for p in model.parameters():
        if p.requires_grad and p.grad is None:
            p.grad = torch.zeros_like(p)
    model.zero_grad = lambda set_to_none = True: torch._foreach_zero_([p.grad for p in model.parameters() if p.grad is not None])


TRAINER["cg"] = cg_stable_grads


# D2: the whole encoder as one compiled graph with torch's non-reentrant checkpointing inside it
# (no CPU offload: changes the user's "unsloth" GC mode, so report-only), static pad64 shapes.
# enc_cg adds CUDA graphs (one fwd + one bwd graph per length bucket).
def _enc_graph(mode):
    def apply(D, args):
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        orig = D._compile_encoder_layers

        def comp(model):
            if not orig(model):
                return False
            from unsloth_zoo.gradient_checkpointing import unpatch_unsloth_smart_gradient_checkpointing
            enc = model.encoder
            for layer in enc.layers:
                layer._compiled_call_impl = None
            unpatch_unsloth_smart_gradient_checkpointing()
            import transformers.modeling_utils, torch.utils.checkpoint as ck
            transformers.modeling_utils.checkpoint = getattr(ck, "_old_checkpoint", ck.checkpoint)
            enc.gradient_checkpointing_disable()
            enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
            enc.compile(mode = mode, dynamic = False)
            print("enc_graph", mode, enc._gradient_checkpointing_func if hasattr(enc, "_gradient_checkpointing_func") else None)
            return True

        D._compile_encoder_layers = comp
        if mode == "reduce-overhead":
            compute_loss = D.DecisionTrainer.compute_loss

            def step_loss(self, *a, **k):
                torch.compiler.cudagraph_mark_step_begin()
                return compute_loss(self, *a, **k)

            D.DecisionTrainer.compute_loss = step_loss
        _pad_multiple(64)(D, args)
    return apply


PRE["enc_compile"] = _enc_graph(None)
PRE["enc_cg"] = _enc_graph("reduce-overhead")
TRAINER["enc_cg"] = cg_stable_grads


# Q6: run the perf tree's compile path for LoRA too (it is full-finetune only in the code).
def lora_compile(D, args):
    orig = D._compile_encoder_layers

    def comp(model):
        flag = getattr(model, "_unsloth_full_finetuning", False)
        model._unsloth_full_finetuning = True
        try:
            return orig(model)
        finally:
            model._unsloth_full_finetuning = flag

    D._compile_encoder_layers = comp


PRE["lora_compile"] = lora_compile


# Ablations for the LoRA long-run collapse: take one perf change back out of the perf tree.
def allow_cudnn(D, args):
    import contextlib
    if hasattr(D, "_no_cudnn_attention"):
        D._no_cudnn_attention = contextlib.nullcontext


def foreach_adam(D, args):
    get = D.DecisionTrainer.get_optimizer_cls_and_kwargs

    def patched(*a, **k):
        cls, kw = get(*a, **k)
        return cls, {**kw, "foreach": True}

    D.DecisionTrainer.get_optimizer_cls_and_kwargs = staticmethod(patched)


def peft_forward(model, D, args):
    n = 0
    for module in model.encoder.modules():
        if hasattr(module, "_unsloth_peft_forward"):
            module.forward = module._unsloth_peft_forward
            n += 1
    print("peft_forward restored", n)
    return model


PRE["allow_cudnn"] = allow_cudnn
PRE["foreach_adam"] = foreach_adam
POST["peft_forward"] = peft_forward


# X9: fused weight-gradient accumulation (Megatron's gradient_accumulation_fusion, stock torch only). Each trainable Linear
# weight keeps a persistent fp32 buffer as its .grad; the backward adds dy^T x into it with one GEMM (fp16/bf16 inputs,
# fp32 output, beta=1) instead of GEMM -> half dW -> cast to fp32 -> AccumulateGrad add. Same VRAM (.grad exists anyway),
# no new argument. Opaque custom op so the compiled layers call it in place on the buffer.
_ACCUM = {}


def _accum_ops():
    if _ACCUM:
        return _ACCUM["op"]

    # A raw Library op: torch.library.custom_op costs ~25 us of Python per call, this ~6 us (CPU-measured).
    lib = torch.library.Library("laya_bench", "DEF")
    lib.define("wgrad_accum_(Tensor(a!) main_grad, Tensor dy, Tensor x) -> ()")

    def impl(main_grad, dy, x):
        dy2, x2 = dy.reshape(-1, dy.shape[-1]), x.reshape(-1, x.shape[-1])
        if dy2.dtype == main_grad.dtype or not dy2.is_cuda:
            main_grad.addmm_(dy2.t().to(main_grad.dtype), x2.to(main_grad.dtype))
        else:
            torch.addmm(main_grad, dy2.t(), x2, out_dtype = main_grad.dtype, out = main_grad)

    lib.impl("wgrad_accum_", impl, "CompositeExplicitAutograd")
    torch.library.register_fake("laya_bench::wgrad_accum_", lambda main_grad, dy, x: None, lib = lib)
    _ACCUM["lib"] = lib
    wgrad_accum_ = torch.ops.laya_bench.wgrad_accum_.default

    class AccumLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, main_grad):
            dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
            xh, wh = x.to(dtype), weight.to(dtype)
            ctx.x_dtype, ctx.has_bias = x.dtype, bias is not None
            ctx.save_for_backward(xh, wh, main_grad)
            with torch.autocast(x.device.type, enabled = False):
                return torch.nn.functional.linear(xh, wh, None if bias is None else bias.to(dtype))

        @staticmethod
        def backward(ctx, dy):
            xh, wh, main_grad = ctx.saved_tensors
            dy = dy.to(wh.dtype)
            wgrad_accum_(main_grad, dy, xh)
            db = dy.reshape(-1, dy.shape[-1]).sum(0, dtype = torch.float32) if ctx.has_bias else None
            return (dy @ wh).to(ctx.x_dtype), None, db, None

    _ACCUM["op"] = AccumLinear
    return AccumLinear


def _accum_forward(self, x):
    return _ACCUM["op"].apply(x, self.weight, self.bias, self.weight._laya_main_grad)


def _attach_main_grads(model):
    for p in model._laya_accum_params:
        if p.grad is None:
            p._laya_main_grad.zero_()
            p.grad = p._laya_main_grad


def accum_fuse_model(model, D, args):
    import types
    _accum_ops()
    params = []
    for module in model.modules():
        if type(module) is torch.nn.Linear and module.weight.requires_grad:
            module.weight._laya_main_grad = torch.zeros_like(module.weight, dtype = torch.float32)
            module.forward = types.MethodType(_accum_forward, module)
            params.append(module.weight)
    model._laya_accum_params = params
    print("accum_fuse linears", len(params), "params", sum(p.numel() for p in params))
    return model


def accum_fuse(D, args):
    compute_loss = D.DecisionTrainer.compute_loss

    def patched(self, model, *a, **k):
        if getattr(model, "_laya_accum_params", None):
            _attach_main_grads(model)
        return compute_loss(self, model, *a, **k)

    D.DecisionTrainer.compute_loss = patched


PRE["accum_fuse"] = accum_fuse
POST["accum_fuse"] = accum_fuse_model


# Long-context screens: train up to 4,096 tokens (Unsloth's offloaded GC sizes its buffers from max_len), compile every
# length bucket instead of falling back to eager after torch's default 8 shapes, and coarser 256-token buckets.
LOAD["long_ctx"] = lambda args: {"max_seq_length": 4096}


def recompile_hi(D, args):
    import torch._dynamo
    cfg = torch._dynamo.config
    for name, value in (("recompile_limit", 256), ("cache_size_limit", 256), ("accumulated_recompile_limit", 8192), ("accumulated_cache_size_limit", 8192)):
        if hasattr(cfg, name):
            setattr(cfg, name, max(getattr(cfg, name), value))


def pad_multiple_256(D, args):
    orig = D._compile_encoder_layers

    def comp(model):
        done = orig(model)
        if done:
            model._unsloth_pad_multiple = 256
        return done

    D._compile_encoder_layers = comp


PRE["recompile_hi"] = recompile_hi
PRE["pad256"] = pad_multiple_256


# Long-context fixes for D1's per-bucket recompiles. dyn_layers: the shipped regional compile with dynamic shapes (one
# graph for every length). geo_buckets: keep static shapes, but pad to geometric buckets above 512 so long data needs
# about 7 shapes (under torch's default limit of 8) instead of one per 64 tokens.
def dyn_layers(D, args):
    orig = D._compile_encoder_layers

    def comp(model):
        if not orig(model):
            return False
        for layer in model.encoder.layers:
            layer._compiled_call_impl = D._training_only(layer, torch.compile(layer._call_impl, dynamic = True))
        return True

    D._compile_encoder_layers = comp


GEO_BUCKETS = [64 * k for k in range(1, 9)] + [768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]


def geo_buckets(D, args):
    import bisect

    def pad_length(model, inputs):
        if not (getattr(model, "_unsloth_pad_multiple", 0) and model.training):
            return inputs
        length = inputs["input_ids"].shape[1]
        target = GEO_BUCKETS[min(bisect.bisect_left(GEO_BUCKETS, length), len(GEO_BUCKETS) - 1)]
        extra = max(0, target - length)
        if not extra:
            return inputs
        pad = model.encoder.config.pad_token_id or 0
        return {**inputs, "input_ids": torch.nn.functional.pad(inputs["input_ids"], (0, extra), value = pad),
                "attention_mask": torch.nn.functional.pad(inputs["attention_mask"], (0, extra))}

    D._pad_length = pad_length


PRE["dyn_layers"] = dyn_layers
PRE["geo_buckets"] = geo_buckets


# X10: padding-free encoder, reusing transformers' own machinery: _get_unpad_data / _pad_input for the gather and
# scatter, the padding-free convention (packed ids + position_ids + cu_seq_lens_q/k, max_length_q/k kwargs; masks
# skipped with the dict form ModernBertModel.forward accepts), and the flash path's window conversion
# (window_size = sliding_window - 1 per side). Attention is torch's built-in varlen FlashAttention (no flash-attn
# dependency), registered through AttentionInterface like the perf tree's SDPA wrapper. Local layers then only compute
# their 128-token window instead of a masked L x L.
_HF_ATTN = "unsloth_decision_varlen"
_HF_SDPA = []


def _varlen_attention(module, query, key, value, attention_mask, scaling = None, sliding_window = None,
                      cu_seq_lens_q = None, cu_seq_lens_k = None, max_length_q = None, max_length_k = None, **kwargs):
    if cu_seq_lens_q is None:
        # Padded batches (eval, fp32) keep the perf tree's own SDPA wrapper.
        return _HF_SDPA[0](module, query, key, value, attention_mask, scaling = scaling, **kwargs)
    from torch.nn.attention.varlen import varlen_attn
    q, k, v = (x.squeeze(0).transpose(0, 1) for x in (query, key, value))
    window = (sliding_window - 1, sliding_window - 1) if sliding_window and max_length_k > sliding_window else (-1, -1)
    out = varlen_attn(q, k, v, cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k, scale = scaling, window_size = window)
    return out.unsqueeze(0), None


def _padding_free_forward(self, input_ids = None, attention_mask = None, **kwargs):
    from transformers.modeling_flash_attention_utils import _get_unpad_data, _pad_input
    if not self.training or input_ids is None or attention_mask is None or not torch.is_autocast_enabled():
        return self._laya_padded_forward(input_ids = input_ids, attention_mask = attention_mask, **kwargs)
    batch, width = input_ids.shape
    indices, cu, max_len = _get_unpad_data(attention_mask)
    # Only an upper bound for the kernel; a power of two keeps the compiled layers to a few int specializations.
    max_len = 1 << (int(max_len) - 1).bit_length()
    position_ids = torch.arange(width, device = input_ids.device).expand(batch, width).flatten()[indices].unsqueeze(0)
    out = self._laya_padded_forward(
        input_ids = input_ids.flatten()[indices].unsqueeze(0), position_ids = position_ids,
        attention_mask = {"full_attention": None, "sliding_attention": None},
        cu_seq_lens_q = cu, cu_seq_lens_k = cu, max_length_q = max_len, max_length_k = max_len, **kwargs,
    )
    out.last_hidden_state = _pad_input(out.last_hidden_state.squeeze(0), indices, batch, width)
    return out


def unpad_hf(model, D, args):
    from types import MethodType
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask
    _HF_SDPA[:] = [D._encoder_sdpa]
    AttentionInterface.register(_HF_ATTN, _varlen_attention)
    AttentionMaskInterface.register(_HF_ATTN, sdpa_mask)
    base = _base_encoder(model)
    base.config._attn_implementation = _HF_ATTN
    base._laya_padded_forward = base.forward
    base.forward = MethodType(_padding_free_forward, base)
    return model


def unpad_hf_pre(D, args):
    # The perf tree's compile step re-points the encoder at its SDPA wrapper; keep ours, and skip the pad-to-64
    # (packed shapes vary anyway, so this path compiles with dynamic shapes: see dyn_layers).
    orig = D._compile_encoder_layers

    def comp(model):
        impl = model.encoder.config._attn_implementation
        done = orig(model)
        model.encoder.config._attn_implementation = impl
        if done:
            model._unsloth_pad_multiple = 0
        return done

    D._compile_encoder_layers = comp


PRE["unpad_hf"] = unpad_hf_pre
POST["unpad_hf"] = unpad_hf
