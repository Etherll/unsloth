# Run laya's official notebooks/laya_finetune_typed_decisions_mps.py as shipped (fp32, TF32 "high"), on CUDA,
# capped at N optimizer steps, to get this PC's official step time and peak VRAM.
import argparse, json, os, shutil, statistics, sys, time

p = argparse.ArgumentParser()
p.add_argument("--mb", type = int, default = 2)
p.add_argument("--acc", type = int, default = 16)
p.add_argument("--steps", type = int, default = 40)
p.add_argument("--warmup", type = int, default = 4)
p.add_argument("--no-checkpointing", action = "store_true")
p.add_argument("--out", required = True)
args = p.parse_args()
LP = os.environ["LP_ROOT"]
if os.name == "nt":
    try:
        import psutil
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
        psutil.Process().cpu_affinity(list(range(16)))
    except Exception:
        pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpuwatch
sys.path.insert(0, f"{LP}/laya-official")
sys.path.insert(0, f"{LP}/laya-official/notebooks")
import torch
import laya_finetune_typed_decisions_mps as M

model_dir = f"{LP}/model-official"
if not os.path.isdir(model_dir):
    shutil.copytree(f"{LP}/model", model_dir)
items_path = f"{LP}/data/official_items.pt"
if not os.path.exists(items_path):
    data = torch.load(f"{LP}/data/items.pt", weights_only = False)
    torch.save([{"ids": i["input_ids"], "markers": i["markers"], "qtype": i["qtype"], "target": i["target"],
                 "label": i["label"]} for i in data["train"]], items_path)

times, state = [], {"t": None}


class Stop(Exception):
    pass


class TimedAdamW(torch.optim.AdamW):
    def step(self, *a, **k):
        out = super().step(*a, **k)
        torch.cuda.synchronize()
        now = time.perf_counter()
        times.append(now - state["t"])
        state["t"] = now
        if len(times) >= args.steps:
            raise Stop
        return out


torch.optim.AdamW = TimedAdamW
M.choose_device = lambda requested: torch.device("cuda")
_collate = M.collate


def collate(items, pad_id):
    if state["t"] is None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        state["t"] = time.perf_counter()
    return _collate(items, pad_id)


M.collate = collate
sys.argv = ["x", "--model-dir", model_dir, "--output-dir", f"{os.environ.get('TMPDIR', LP + '/tmp')}/official-out", "--items", items_path,
            "--epochs", "1", "--micro-batch", str(args.mb), "--grad-accum", str(args.acc), "--device", "auto"]
if args.no_checkpointing:
    sys.argv.append("--no-checkpointing")
watch = gpuwatch.Watch(every = 10.0)
try:
    M.main()
except Stop:
    pass
env = watch.done()
timed = times[args.warmup:]
r = dict(mb = args.mb, acc = args.acc, steps = len(times), s_step = statistics.median(timed), first_step_s = times[0],
         peak_alloc_mib = torch.cuda.max_memory_allocated() / 2**20, peak_res_mib = torch.cuda.max_memory_reserved() / 2**20,
         step_times = times, torch = torch.__version__, gc = not args.no_checkpointing,
         s_step_p25 = sorted(timed)[len(timed) // 4], env = env)
json.dump(r, open(args.out, "w"), indent = 1)
print(json.dumps({k: v for k, v in r.items() if k != "step_times"}))
