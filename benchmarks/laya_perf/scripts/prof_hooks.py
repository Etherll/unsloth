# torch.profiler wiring for bench.py: ranges for encoder layers, head, scorer, backward, optimizer, collate.
import torch
from torch.autograd.profiler import record_function
from torch.profiler import ProfilerActivity, profile, schedule


def _ranges(module, name):
    stack = []

    def pre(*_):
        r = record_function(name)
        r.__enter__()
        stack.append(r)

    def post(*_):
        stack.pop().__exit__(None, None, None)

    module.register_forward_pre_hook(pre)
    module.register_forward_hook(post)


def _wrap(obj, attr, name):
    fn = getattr(obj, attr)

    def wrapped(*a, **k):
        with record_function(name):
            return fn(*a, **k)

    setattr(obj, attr, wrapped)


def _layer_ranges():
    # Hooks on compiled layers would be traced by dynamo (graph breaks): LP_LAYER_RANGES=0 for compiled runs.
    import os
    return os.environ.get("LP_LAYER_RANGES", "1") == "1"


def install(trainer, model, args, TrainerCallback):
    base = model.encoder
    base = getattr(base, "get_base_model", lambda: base)()
    _ranges(base.embeddings, "enc_embed")
    for layer in base.layers if _layer_ranges() else ():
        _ranges(layer, "enc_layer")
    _ranges(base.final_norm, "enc_final_norm")
    for layer in model.head.layers:
        _ranges(layer, "head_layer")
    _ranges(model.scorer, "scorer")
    _ranges(model.act_head, "act_head")
    _wrap(trainer.accelerator, "backward", "backward")
    collate = trainer.data_collator
    trainer.data_collator = lambda items: _collate(collate, items)
    prof = profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes = True, with_stack = True, with_flops = True,
        schedule = schedule(wait = 2, warmup = 2, active = 3),
        on_trace_ready = lambda p: _done(p, args.prof),
    )

    class Cb(TrainerCallback):
        def on_train_begin(self, *a, **k):
            prof.start()
            opt = trainer.optimizer
            if opt is not None and not getattr(opt, "_lp_wrapped", False):
                _wrap(opt, "step", "optimizer_step")
                opt._lp_wrapped = True

        def on_step_begin(self, *a, **k):
            opt = trainer.optimizer
            if opt is not None and not getattr(opt, "_lp_wrapped", False):
                _wrap(opt, "step", "optimizer_step")
                opt._lp_wrapped = True

        def on_step_end(self, *a, **k):
            prof.step()

        def on_train_end(self, *a, **k):
            prof.stop()

    trainer.add_callback(Cb())


def _collate(collate, items):
    with record_function("collate"):
        return collate(items)


def _done(p, out):
    import os, sys
    os.makedirs(out, exist_ok = True)
    p.export_chrome_trace(os.path.join(out, "trace.json"))
    # Stacks nest deep enough that key_averages' parent walk overflows the default limit.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 200000))
    try:
        ka = p.key_averages()
        with open(os.path.join(out, "top_cuda.txt"), "w") as f:
            f.write(ka.table(sort_by = "self_cuda_time_total", row_limit = 40))
        with open(os.path.join(out, "top_cpu.txt"), "w") as f:
            f.write(ka.table(sort_by = "self_cpu_time_total", row_limit = 40))
    except Exception as exc:
        print("key_averages failed", type(exc).__name__)


def install_nvtx(trainer, model):
    # NVTX ranges for nsys: step / forward / backward / optimizer, and enc_layer (inside backward = GC recompute).
    from transformers import TrainerCallback
    nvtx = torch.cuda.nvtx

    def wrap(obj, attr, name):
        fn = getattr(obj, attr)

        def wrapped(*a, **k):
            nvtx.range_push(name)
            try:
                return fn(*a, **k)
            finally:
                nvtx.range_pop()

        setattr(obj, attr, wrapped)

    wrap(trainer, "compute_loss", "forward")
    wrap(trainer.accelerator, "backward", "backward")
    if _layer_ranges():
        base = model.encoder
        base = getattr(base, "get_base_model", lambda: base)()
        for layer in base.layers:
            # Hooks must return None, or they replace the layer's inputs / outputs.
            layer.register_forward_pre_hook(lambda *_: (nvtx.range_push("enc_layer"), None)[1])
            layer.register_forward_hook(lambda *_: (nvtx.range_pop(), None)[1])

    class Cb(TrainerCallback):
        def on_step_begin(self, *a, **k):
            opt = trainer.optimizer
            if opt is not None and not getattr(opt, "_lp_nvtx", False):
                wrap(opt, "step", "optimizer")
                opt._lp_nvtx = True
            nvtx.range_push("step")

        def on_step_end(self, *a, **k):
            nvtx.range_pop()

    trainer.add_callback(Cb())
