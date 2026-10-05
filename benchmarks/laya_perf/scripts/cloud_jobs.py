# Runs on a rented Colab / Kaggle VM (one GPU): builds the base / perf / cand trees from the bench clone,
# installs the pinned stack around the VM's own torch, preps model + data, then runs JOBS in order.
# Every job's JSON comes back on a "LAYA_RESULT <name> <json>" line; progress on "LAYA_JOB" lines.
# Expects REPO (the bench-branch clone) and JOBS (list of {"name", "cmd": [script, args...], "env": {}}).
import json, os, shutil, subprocess, sys, threading, time

BASE_SHA = "c5cb7cba596bbfd58d1d2bd78edf6d4d5d5d4c57"  # placeholder, replaced by cloud_submit.py
PIN = ["transformers==5.5.0", "unsloth_zoo==2026.9.9", "peft==0.21.2", "accelerate==1.15.0", "trl==1.13.0",
       "datasets>=3.4.1,<5", "sentencepiece", "protobuf", "tyro", "hf_transfer", "psutil", "pandas", "pyarrow",
       "pytest", "typer", "rich", "pydantic", "pyyaml", "nest-asyncio", "structlog", "click", "diffusers",
       "sentence-transformers", "safetensors", "huggingface_hub"]
ROOT = "/kaggle/working/lp" if os.path.isdir("/kaggle") else "/content/lp"
S = f"{REPO}/benchmarks/laya_perf/scripts"


def sh(cmd, **kw):
    print("+", cmd, flush = True)
    r = subprocess.run(cmd, shell = True, text = True, capture_output = True, **kw)
    print(r.stdout[-3000:], r.stderr[-3000:], flush = True)
    if r.returncode:
        raise RuntimeError(f"failed: {cmd}")
    return r.stdout


def setup():
    import torch
    os.makedirs(ROOT, exist_ok = True)
    torch_pin = f"torch=={torch.__version__}"
    open(f"{ROOT}/constraints.txt", "w").write(torch_pin + "\n")
    for attempt in range(4):
        try:
            sh(f"{sys.executable} -m pip install -q --timeout 120 -c {ROOT}/constraints.txt " + " ".join(f"'{p}'" for p in PIN))
            break
        except RuntimeError:
            if attempt == 3:
                raise
            time.sleep(20)
    sh(f"git -C {REPO} fetch -q origin {BASE_SHA} laya-decision-perf")
    for name, ref in (("base", BASE_SHA), ("perf", "FETCH_HEAD")):
        if not os.path.isdir(f"{ROOT}/{name}"):
            if name == "perf":
                sh(f"git -C {REPO} fetch -q origin laya-decision-perf")
            sh(f"git -C {REPO} worktree add -q --detach {ROOT}/{name} {ref}")
    if not os.path.isdir(f"{ROOT}/cand"):
        sh(f"git -C {REPO} worktree add -q --detach {ROOT}/cand $(git -C {ROOT}/perf rev-parse HEAD)")
        sh(f"cd {ROOT}/cand && (git apply --check {REPO}/benchmarks/laya_perf/patches/d1_static.diff 2>/dev/null && git apply {REPO}/benchmarks/laya_perf/patches/d1_static.diff && echo applied || echo patch already on perf)")
    for t in ("base", "perf", "cand"):
        print("LAYA_TREE", t, sh(f"git -C {ROOT}/{t} log -1 --format='%H %s'").strip(), flush = True)
    env = os.environ
    env.update(LP_ROOT = ROOT, PY = sys.executable, BASE = f"{ROOT}/base", PERF = f"{ROOT}/perf", CAND = f"{ROOT}/cand", S = S,
               HF_HUB_ENABLE_HF_TRANSFER = "1", TOKENIZERS_PARALLELISM = "false", UNSLOTH_DISABLE_STATISTICS = "1",
               CUDA_VISIBLE_DEVICES = "0", CUDA_DEVICE_ORDER = "PCI_BUS_ID", LANE = "0", PYTHONWARNINGS = "ignore",
               TMPDIR = f"{ROOT}/tmp", AB_OUT = f"{ROOT}/ab")
    os.makedirs(f"{ROOT}/tmp", exist_ok = True)
    os.makedirs(f"{ROOT}/out", exist_ok = True)
    if not os.path.exists(f"{ROOT}/data/items.pt"):
        os.makedirs(f"{ROOT}/data", exist_ok = True)
        from huggingface_hub import snapshot_download
        snapshot_download("convaiinnovations/laya", revision = "55cf4c4ebb", local_dir = f"{ROOT}/model",
                          allow_patterns = ["rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"])
        sh(f"cd {ROOT} && PYTHONPATH={ROOT}/perf {sys.executable} {S}/prep.py")
    import psutil
    print("LAYA_ENV", json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0), "cc": torch.cuda.get_device_capability(0),
        "mem_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "bf16": torch.cuda.is_bf16_supported(), "cpus": os.cpu_count(), "ram_gib": round(psutil.virtual_memory().total / 2**30),
        "python": sys.version.split()[0], "pins": sh(f"{sys.executable} -m pip list 2>/dev/null | grep -iE '^(transformers|unsloth.zoo|peft|accelerate|triton|trl|xformers|flash.attn) '"),
        "cpu_model": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": ") if os.path.exists("/proc/cpuinfo") else "?",
    }), flush = True)


def heartbeat(name, stop):
    t0 = time.time()
    while not stop.wait(240):
        print(f"LAYA_JOB {name} running {int(time.time() - t0)} s", flush = True)


def run(job):
    name, cmd, env = job["name"], job["cmd"], {**os.environ, **{k: str(v) for k, v in job.get("env", {}).items()}}
    cmd = [x.replace("$ROOT", ROOT).replace("$S", S).replace("$PY", sys.executable) for x in cmd]
    if cmd[0].endswith(".py"):
        cmd = [sys.executable] + cmd
    out = job.get("out", "").replace("$ROOT", ROOT) or next((cmd[i + 1] for i, x in enumerate(cmd) if x == "--out"), None)
    log = f"{ROOT}/out/{name}.log"
    stop = threading.Event()
    threading.Thread(target = heartbeat, args = (name, stop), daemon = True).start()
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout = f, stderr = subprocess.STDOUT, env = env, cwd = job.get("cwd", ROOT).replace("$ROOT", ROOT)).returncode
    stop.set()
    print(f"LAYA_JOB {name} rc={rc} {time.time() - t0:.0f}s", flush = True)
    if job.get("print_log"):
        print(f"LAYA_LOG {name} " + json.dumps(open(log).read()[-job["print_log"]:]), flush = True)
    if rc:
        print(f"LAYA_FAIL {name} " + json.dumps(open(log).read()[-4000:]), flush = True)
    elif out and os.path.exists(out):
        print(f"LAYA_RESULT {name} " + open(out).read().replace("\n", " "), flush = True)
    return rc


setup()
failed = [j["name"] for j in JOBS if run(j)]
print("LAYA_DONE", json.dumps({"jobs": len(JOBS), "failed": failed}), flush = True)
# Kaggle ships the whole working dir back as output: keep only the logs.
for d in ("model", "data", "tmp", "base", "perf", "cand", "xf", "xfpkgs", "laya-official"):
    shutil.rmtree(f"{ROOT}/{d}", ignore_errors = True)
subprocess.run(f"git -C {REPO} worktree prune; find {ROOT}/out -name '*.json' -size +5M -delete; rm -rf {ROOT}/out/prof_*", shell = True)
