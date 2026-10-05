# Real DecisionTrainer benchmark: one config per process, JSON out.
import argparse, json, os, statistics, subprocess, sys, time

p = argparse.ArgumentParser()
p.add_argument("--tree", required = True)
p.add_argument("--mode", default = "full", choices = ["full", "lora"])
p.add_argument("--mb", type = int, default = 2)
p.add_argument("--acc", type = int, default = 16)
p.add_argument("--steps", type = int, default = 40)
p.add_argument("--epochs", type = float, default = 0)
p.add_argument("--warmup", type = int, default = 4)
p.add_argument("--gc", default = "unsloth", choices = ["unsloth", "true", "off"])
p.add_argument("--opts", default = "")
p.add_argument("--optim", default = "adamw_torch")
p.add_argument("--items", default = "bench", choices = ["bench", "train"])
p.add_argument("--data", default = "items.pt", help = "data file under $LP_ROOT/data (items_long.pt: make_long.py)")
p.add_argument("--eval", action = "store_true")
p.add_argument("--dropout-off", action = "store_true")
p.add_argument("--save-curve", default = "")
p.add_argument("--out", required = True)
p.add_argument("--tag", default = "")
p.add_argument("--prof", default = "")
p.add_argument("--stop-at", type = int, default = 0)
p.add_argument("--memsnap", default = "")
p.add_argument("--seed", type = int, default = 3407)
p.add_argument("--eval-at", default = "", help = "comma-separated optimizer steps for a holdout eval")
p.add_argument("--nvtx", action = "store_true")
p.add_argument("--lr", type = float, default = 0)
p.add_argument("--warmup-steps", type = int, default = 0, help = "LR warmup (the PR recipe has none)")
p.add_argument("--cuprof", default = "", help = "START,STOP optimizer steps for cudaProfilerStart/Stop (nsys capture range)")
args = p.parse_args()

def _pin():
    # Windows: high priority on the V-cache CCD. Linux: the lane's taskset range (env-linux.sh `pin`).
    if os.name != "nt":
        return
    try:
        import psutil
        me = psutil.Process()
        me.nice(psutil.HIGH_PRIORITY_CLASS if os.name == "nt" else -5)
        if os.environ.get("LP_AFFINITY", "1") == "1":
            me.cpu_affinity(list(range(16)))
    except Exception as exc:
        print("pin failed", exc)

_pin()

sys.path.insert(0, args.tree)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import torch
import unsloth  # noqa: F401
from unsloth.models import decision as D
from transformers import TrainerCallback, TrainingArguments
import opts as O
if os.environ.get("LP_TORCH_LOGS"):
    # unsloth_zoo clears TORCH_LOGS on import; turn the useful ones back on.
    torch._logging.set_logs(graph_breaks = True, recompiles = True, perf_hints = True)

assert os.path.abspath(D.__file__).lower().startswith(os.path.abspath(args.tree).lower()), D.__file__
LP = os.environ["LP_ROOT"]
MODEL = f"{LP}/model"
names = [o for o in args.opts.split(",") if o]
O.pre(names, D, args)

data = torch.load(f"{LP}/data/{args.data}", weights_only = False)
items, holdout = data[args.items], data["holdout"]
gc = {"unsloth": "unsloth", "true": True, "off": False}[args.gc]
t_load = time.perf_counter()
load_kw = O.load_kwargs(names, args)
model, tok = D.FastDecisionModel.from_pretrained(
    MODEL, full_finetuning = args.mode == "full", use_gradient_checkpointing = gc, **load_kw
)
if args.mode == "lora":
    model = D.FastDecisionModel.get_peft_model(
        model, r = 64, lora_alpha = 64, lora_dropout = 0.0, use_gradient_checkpointing = gc, random_state = args.seed
    )
if args.dropout_off:
    O.dropout_off(model)
model = O.post(names, model, D, args)
t_load = time.perf_counter() - t_load


def smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output = True, text = True, timeout = 10,
        ).stdout.strip()
        return out
    except Exception as exc:
        return str(exc)


class Timer(TrainerCallback):
    def __init__(self):
        self.times, self.t = [], None

    def on_train_begin(self, *a, **k):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.t = time.perf_counter()

    def on_step_begin(self, *a, **k):
        self.p0 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    def on_pre_optimizer_step(self, *a, **k):
        self.fb = max(getattr(self, "fb", 0), torch.cuda.max_memory_allocated())
        self.p1 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    def on_optimizer_step(self, *a, **k):
        self.opt = max(getattr(self, "opt", 0), torch.cuda.max_memory_allocated())
        self.res = max(getattr(self, "res", 0), torch.cuda.max_memory_reserved())

    def on_step_end(self, a, state, control, **k):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.times.append(now - self.t)
        self.t = now
        if args.stop_at and state.global_step >= args.stop_at:
            control.should_training_stop = True


lr = args.lr or (2.5e-5 if args.mode == "full" else 8e-4)
targs = dict(
    output_dir = f"{os.environ.get('TMPDIR', LP + '/tmp')}/run-{os.getpid()}",
    per_device_train_batch_size = args.mb,
    per_device_eval_batch_size = 16,
    gradient_accumulation_steps = args.acc,
    learning_rate = lr,
    weight_decay = 0.01,
    lr_scheduler_type = "cosine",
    optim = args.optim,
    seed = args.seed,
    # Studio's own choice (Unsloth's is_bfloat16_supported): bf16 on Ampere+, fp16 on a T4. torch's own
    # is_bf16_supported() also counts emulation and says True on a T4.
    bf16 = D.is_bfloat16_supported(),
    fp16 = not D.is_bfloat16_supported(),
    logging_steps = 1,
    report_to = "none",
    save_strategy = "no",
    disable_tqdm = True,
    warmup_steps = args.warmup_steps,
)
if args.epochs:
    targs["num_train_epochs"] = args.epochs
else:
    targs["max_steps"] = args.steps
timer = Timer()
trainer = D.DecisionTrainer(
    model = model, args = TrainingArguments(**targs), train_dataset = items,
    processing_class = tok, callbacks = [timer],
)
counts = []
collate = trainer.data_collator


def counting(batch_items):
    batch = collate(batch_items)
    counts.append((int(batch["attention_mask"].sum()), batch["attention_mask"].numel()))
    return batch


trainer.data_collator = counting
O.trainer(names, trainer, D, args)
if args.prof:
    import prof_hooks
    prof_hooks.install(trainer, model, args, TrainerCallback)
if args.memsnap:
    class Snap(TrainerCallback):
        def on_train_begin(self, *a, **k):
            torch.cuda.memory._record_memory_history(max_entries = 200000)

        def on_step_end(self, a, state, control, **k):
            if state.global_step == 3:
                torch.cuda.memory._dump_snapshot(args.memsnap)
                torch.cuda.memory._record_memory_history(enabled = None)

    trainer.add_callback(Snap())
eval_at = sorted(int(x) for x in args.eval_at.split(",") if x)
checkpoints = []
if eval_at:
    class Ckpt(TrainerCallback):
        # Holdout eval through the normal evaluate/_logits path; its time is kept out of step timing.
        def on_log(self, a, state, control, logs = None, **k):
            step = state.global_step
            if step not in eval_at or any(c["step"] == step for c in checkpoints):
                return
            torch.cuda.synchronize()
            t = time.perf_counter()
            ho = D.FastDecisionModel.evaluate(model, tok, holdout)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            timer.t += dt
            losses = [h["loss"] for h in trainer.state.log_history if "loss" in h and "grad_norm" in h]
            times = sorted(timer.times[args.warmup:] or timer.times)
            checkpoints.append(dict(
                step = step, wall_s = time.perf_counter() - t0 - sum(c["eval_s"] for c in checkpoints) - dt,
                s_step_p25 = times[len(times) // 4], s_step_med = statistics.median(times),
                peak_alloc_mib = max(getattr(timer, "fb", 0), getattr(timer, "opt", 0)) / 2**20,
                peak_res_mib = getattr(timer, "res", 0) / 2**20,
                train_loss_win = statistics.mean(losses[-20:]), holdout = ho, eval_s = dt,
            ))
            print("ckpt", json.dumps(checkpoints[-1]), flush = True)
    trainer.add_callback(Ckpt())
if args.cuprof:
    c0, c1 = (int(x) for x in args.cuprof.split(","))

    class CuProf(TrainerCallback):
        def on_step_begin(self, a, state, *x, **k):
            if state.global_step == c0:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()

        def on_step_end(self, a, state, *x, **k):
            if state.global_step == c1:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()

    trainer.add_callback(CuProf())
if args.nvtx:
    import prof_hooks
    prof_hooks.install_nvtx(trainer, model)
import gpuwatch
watch = gpuwatch.Watch(every = 10.0)
util0 = smi()
t0 = time.perf_counter()
trainer.train()
torch.cuda.synchronize()
wall = time.perf_counter() - t0
peak_alloc = max(torch.cuda.max_memory_allocated(), getattr(timer, 'fb', 0), getattr(timer, 'opt', 0))
peak_res = max(torch.cuda.max_memory_reserved(), getattr(timer, 'res', 0))

logs = [h for h in trainer.state.log_history if "loss" in h and "grad_norm" in h]
steps = len(timer.times)
timed = timer.times[args.warmup:] if steps > args.warmup + 2 else timer.times
real = sum(c[0] for c in counts)
slots = sum(c[1] for c in counts)
tok_per_step = real / max(1, steps)
med = statistics.median(timed)
p25 = sorted(timer.times[args.warmup:-1] or timer.times)[len(timer.times[args.warmup:-1] or timer.times) // 4]
result = dict(
    tag = args.tag, tree = args.tree, mode = args.mode, mb = args.mb, acc = args.acc, gc = args.gc,
    opts = names, optim = args.optim, steps = steps, load_s = t_load, wall_s = wall,
    s_step = med, s_step_p25 = p25, s_step_mean = statistics.mean(timed), first_step_s = timer.times[0], seed = args.seed,
    tok_s = tok_per_step / med, pad_frac = 1 - real / max(1, slots),
    peak_alloc_mib = peak_alloc / 2**20, peak_res_mib = peak_res / 2**20,
    loss = [h["loss"] for h in logs], grad_norm = [h["grad_norm"] for h in logs],
    step_times = timer.times, peak_fwdbwd_mib = getattr(timer, 'fb', 0) / 2**20, peak_opt_mib = getattr(timer, 'opt', 0) / 2**20, smi_start = util0, torch = torch.__version__, file = D.__file__,
    env = watch.done(), checkpoints = checkpoints, half = "bf16" if D.is_bfloat16_supported() else "fp16", gpu = torch.cuda.get_device_name(), cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
)
if args.eval:
    t = time.perf_counter()
    result["holdout"] = D.FastDecisionModel.evaluate(model, tok, holdout)
    result["eval_s"] = time.perf_counter() - t
try:
    from torch._dynamo.utils import counters
    result["dynamo_counters"] = {k: dict(v) for k, v in counters.items() if v}
except Exception as exc:
    result["dynamo_counters"] = str(exc)
with open(args.out, "w") as f:
    json.dump(result, f, indent = 1)
print(json.dumps({k: v for k, v in result.items() if k not in ("loss", "grad_norm", "step_times")}))
