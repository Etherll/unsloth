# Local side: build a notebook that clones the bench branch and runs cloud_jobs.py with a job list, rent one
# Colab / Kaggle GPU through ~/my-project/scripts/scripts/notebook_cloud_run.py, then pull every LAYA_* line
# out of the executed notebook into results/cloud/<tag>/ (one JSON per job, plus env / log lines).
# usage: cloud_submit.py --backend colab --gpu G4 --tag g4_b1 --jobs jobs.json [--dry-run]
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NCR = Path.home() / "my-project/scripts/scripts/notebook_cloud_run.py"
REPO_URL = "https://github.com/Etherll/unsloth.git"
BASE_SHA = "c5cb7cba59f9522d8da2de0522baf77c527f579e"
ap = argparse.ArgumentParser()
ap.add_argument("--backend", default = "colab", choices = ("colab", "kaggle"))
ap.add_argument("--gpu", default = "G4")
ap.add_argument("--tag", required = True)
ap.add_argument("--jobs", required = True)
ap.add_argument("--ref", default = "laya-decision-perf-bench")
ap.add_argument("--wall", type = int, default = 4 * 3600)
ap.add_argument("--dry-run", action = "store_true")
a = ap.parse_args()
jobs = json.load(open(a.jobs))
out = HERE.parent / "results" / "cloud" / a.tag
out.mkdir(parents = True, exist_ok = True)
remote = "/kaggle/working/repo" if a.backend == "kaggle" else "/content/repo"
driver = (HERE / "cloud_jobs.py").read_text().replace('"c5cb7cba596bbfd58d1d2bd78edf6d4d5d5d4c57"', repr(BASE_SHA))
cells = [
    "!nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv; nproc; free -g | head -2; python -V",
    "%%bash\nset -eo pipefail\n"
    f"rm -rf {remote}; for i in 1 2 3; do git clone -q --filter=blob:none -b {a.ref} {REPO_URL} {remote} && break; sleep 10; done\n"
    f"git -C {remote} log -1 --format='LAYA_BENCH %H %s'\n",
    f"REPO = {remote!r}\nJOBS = {json.dumps(jobs)}\n" + driver,
]
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c, "id": f"c{i}"}
                for i, c in enumerate(cells)]}
nb_path = out / f"laya_{a.tag}.ipynb"
nb_path.write_text(json.dumps(nb, indent = 1))
cmd = [sys.executable, str(NCR), "--backend", a.backend, "--gpu", a.gpu, "--no-smoke-patch", "--non-interactive",
       "--per-cell-timeout", str(a.wall), "--wall-timeout", str(a.wall + 600), "--idle-timeout", "1800",
       "--outdir", str(out / "run"), str(nb_path)]
if a.backend == "colab":
    cmd[4:4] = ["--colab-auth", "oauth2"]
env = dict(os.environ, PATH = "/mnt/disks/unslothai/etherl/.local/bin:" + os.environ["PATH"],
           KAGGLE_API_TOKEN = "/mnt/disks/unslothai/etherl/.secrets/kaggle_api_token")
print(" ".join(cmd), flush = True)
if a.dry_run:
    print(f"notebook {nb_path}: {len(jobs)} jobs", [j["name"] for j in jobs])
    sys.exit(0)
t0 = time.time()
with open(out / "run.log", "w") as log:
    rc = subprocess.run(cmd + (["--dry-run"] if a.dry_run else []), stdout = log, stderr = subprocess.STDOUT, env = env).returncode
rep = out / "run" / "report.json"
res = (json.loads(rep.read_text()).get("results") or [{}])[0] if rep.exists() else {}
summary = {"tag": a.tag, "backend": a.backend, "gpu": a.gpu, "rc": rc, "status": res.get("status"), "minutes": round((time.time() - t0) / 60, 1),
           "summary": res.get("summary"), "failures": res.get("failures", [])[:5]}
nbx = res.get("executed_notebook")
if nbx and Path(nbx).exists():
    text = []
    for c in json.loads(Path(nbx).read_text()).get("cells", []):
        for o in c.get("outputs", []) or []:
            t = o.get("text", "")
            text += ("".join(t) if isinstance(t, list) else t).splitlines()
    for ln in text:
        if ln.strip().startswith("LAYA_ENV "):
            (out / "env.json").write_text(ln.strip()[9:])
            continue
        m = re.match(r"(LAYA_[A-Z]+) (\S+) ?(.*)", ln.strip())
        if not m:
            continue
        kind, name, rest = m.groups()
        if kind == "LAYA_RESULT":
            (out / f"{name}.json").write_text(rest)
        elif kind in ("LAYA_FAIL", "LAYA_LOG"):
            (out / f"{name}.{kind[5:].lower()}.txt").write_text(json.loads(rest) if rest.startswith('"') else rest)
        summary.setdefault("lines", []).append(ln.strip()[:300]) if kind in ("LAYA_JOB", "LAYA_DONE", "LAYA_TREE", "LAYA_BENCH") and "running" not in ln else None
(out / "summary.json").write_text(json.dumps(summary, indent = 1))
ledger = HERE.parent / "results" / "cloud" / "runs.jsonl"
with open(ledger, "a") as f:
    f.write(json.dumps({k: summary[k] for k in ("tag", "backend", "gpu", "rc", "status", "minutes")} | {"jobs": [j["name"] for j in jobs], "ended": time.strftime("%F %T")}) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "lines"}, indent = 1))
print("\n".join(summary.get("lines", [])[-60:]))
