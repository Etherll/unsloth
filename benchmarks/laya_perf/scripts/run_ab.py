# Paired A/B: fresh process per run, ABAB pairs; speed = paired median of p25 step-time ratios.
# usage: run_ab.py NAME "<common bench args>" "<arm A args>" "<arm B args>" [pairs] [max_pairs]
# Starts with `pairs` (3); adds pairs up to max_pairs (5) only if the pair spread is > 3% or the
# median is within 1% of a decision threshold. A run that saw a foreign GPU process is redone.
import json, os, shutil, statistics, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpufree

name, common, arm_a, arm_b = sys.argv[1:5]
pairs = int(sys.argv[5]) if len(sys.argv) > 5 else 3
max_pairs = int(sys.argv[6]) if len(sys.argv) > 6 else max(pairs, 5)
LP = os.environ["LP_ROOT"]
OUT = os.environ["AB_OUT"]
os.makedirs(f"{OUT}/{name}", exist_ok = True)
py = os.environ.get("PY", sys.executable)
bench = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")
lane = int(os.environ["LANE"])
THRESHOLDS = (0.05, 0.10, -0.01)
import holdlib
LOAD_MAX = holdlib.LOAD_MAX


def hold_others():
    holdlib.hold_others(log)


release_others = holdlib.release_others


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {name} {msg}"
    print(line, flush = True)
    open(f"{OUT}/{name}/log.txt", "a").write(line + "\n")


def run(arm, args, i):
    out = f"{OUT}/{name}/{arm}{i}.json"
    for attempt in range(3):
        if os.path.exists(out):
            r = json.load(open(out))
            env = r.get("env", {})
            load = max(env.get("start", {}).get("loadavg", [0])[0], env.get("end", {}).get("loadavg", [0])[0])
            if not env.get("foreign") and load <= LOAD_MAX:
                return r
            shutil.move(out, out.replace(".json", f".discard{attempt}.json"))
            log(f"{arm}{i} discarded: foreign GPU process {env.get('foreign')} / load {load}")
        hold_others()
        cmd = [py, bench] + common.split() + args.split() + ["--out", out, "--tag", f"{name}:{arm}{i}"]
        try:
            r = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace")
        finally:
            release_others()
        if r.returncode != 0 or not os.path.exists(out):
            open(f"{OUT}/{name}/{arm}{i}.err", "w", encoding = "utf-8").write(r.stdout[-20000:] + r.stderr[-20000:])
            raise SystemExit(f"{name} {arm}{i} failed, see {arm}{i}.err")
        j = json.load(open(out))
        log(f"{arm}{i} p25 {j['s_step_p25']:.4f} med {j['s_step']:.4f} res {j['peak_res_mib']:.0f} load {j['env']['start']['loadavg'][0]}->{j['env']['end']['loadavg'][0]}")
    raise SystemExit(f"{name} {arm}{i}: foreign GPU process on 3 attempts")


def summarize(res):
    sp = [a["s_step_p25"] / b["s_step_p25"] - 1 for a, b in res]
    sm = [a["s_step"] / b["s_step"] - 1 for a, b in res]
    tok = [(b["tok_s"] * b["s_step"] / b["s_step_p25"]) / (a["tok_s"] * a["s_step"] / a["s_step_p25"]) - 1 for a, b in res]
    vram = [b["peak_res_mib"] / a["peak_res_mib"] - 1 for a, b in res]
    alloc = [b["peak_alloc_mib"] / a["peak_alloc_mib"] - 1 for a, b in res]
    med = lambda k, arm: statistics.median(x[arm][k] for x in res)
    return dict(
        name = name, pairs = len(res), common = common, arm_a = arm_a, arm_b = arm_b,
        A_p25 = med("s_step_p25", 0), B_p25 = med("s_step_p25", 1), A_s_step = med("s_step", 0), B_s_step = med("s_step", 1),
        A_tok_s = med("tok_s", 0) * med("s_step", 0) / med("s_step_p25", 0), B_tok_s = med("tok_s", 1) * med("s_step", 1) / med("s_step_p25", 1),
        speedup = statistics.median(sp), speed_pairs = [round(x, 4) for x in sp], spread = max(sp) - min(sp),
        same_sign = all(s > 0 for s in sp) or all(s < 0 for s in sp), median_speedup = statistics.median(sm), tok_speedup = statistics.median(tok),
        A_res = med("peak_res_mib", 0), B_res = med("peak_res_mib", 1), vram = statistics.median(vram),
        A_alloc = med("peak_alloc_mib", 0), B_alloc = med("peak_alloc_mib", 1), alloc = statistics.median(alloc),
        first_step = (res[0][0]["first_step_s"], res[0][1]["first_step_s"]),
        loadavg = [(a["env"]["start"]["loadavg"][0], b["env"]["end"]["loadavg"][0]) for a, b in res],
        others = res[0][0]["env"]["start"]["gpus"],
    )


res = []
i = 0
while True:
    res.append((run("A", arm_a, i), run("B", arm_b, i)))
    i += 1
    if i < pairs:
        continue
    s = summarize(res)
    near = any(abs(s["speedup"] - t) < 0.01 for t in THRESHOLDS)
    if i >= max_pairs or not (s["spread"] > 0.03 or near):
        break
    log(f"extending: spread {s['spread']:.3f} near-threshold {near}")
json.dump(s, open(f"{OUT}/{name}/summary.json", "w"), indent = 1)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k not in ("others", "common")}))
