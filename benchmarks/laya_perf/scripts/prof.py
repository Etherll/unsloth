# Summarize a bench.py --prof trace: GPU busy %, kernel time by phase/region/category.
import bisect, collections, json, re, sys

path = sys.argv[1]
tr = json.load(open(path, encoding = "utf-8"))
ev = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
steps = [e for e in ev if e.get("name", "").startswith("ProfilerStep#") and e.get("cat") == "user_annotation"]
t0 = min(e["ts"] for e in steps); t1 = max(e["ts"] + e["dur"] for e in steps)
wall = t1 - t0
dev = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and t0 <= e["ts"] <= t1]
runtime = {e["args"].get("correlation"): e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
ranges = collections.defaultdict(list)
for e in ev:
    if e.get("cat") == "user_annotation" and not e["name"].startswith("ProfilerStep"):
        ranges[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
top = [r for rs in ranges.values() for r in rs if r[2] in ("backward", "optimizer_step", "collate")]


def enclosing(tid, ts):
    best = None
    for a, b, n in ranges.get(tid, ()):
        if a <= ts <= b and (best is None or a >= best[0]):
            best = (a, b, n)
    return best[2] if best else None


def phase(ts):
    for a, b, n in top:
        if a <= ts <= b:
            return n
    return "forward"


def cat(name):
    n = name.lower()
    if "memcpy" in n or n.startswith("memcpy"):
        return "memcpy"
    if "memset" in n:
        return "memset"
    if any(k in n for k in ("flash", "fmha", "attention", "cudnn", "sdpa", "efficient")):
        return "attention"
    if any(k in n for k in ("gemm", "cutlass", "sm90", "sm100", "sm120", "nvjet", "xmma", "ampere", "magma")):
        return "gemm"
    if "multi_tensor" in n or "foreach" in n or "adam" in n:
        return "optimizer"
    if "norm" in n:
        return "norm"
    if "reduce" in n:
        return "reduce"
    return "elementwise"


busy_iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in dev)
busy, cur = 0, None
for a, b in busy_iv:
    if cur is None or a > cur[1]:
        if cur:
            busy += cur[1] - cur[0]
        cur = [a, b]
    else:
        cur[1] = max(cur[1], b)
if cur:
    busy += cur[1] - cur[0]

by_phase_region = collections.Counter(); by_cat = collections.Counter(); by_name = collections.Counter()
memcpy_kind = collections.Counter()
for e in dev:
    launch = runtime.get(e["args"].get("correlation"))
    region = enclosing(launch["tid"], launch["ts"]) if launch else None
    ph = phase(launch["ts"]) if launch else "?"
    if ph == "backward" and region == "enc_layer":
        key = "backward/enc_recompute"
    elif ph == "backward" and region in ("head_layer", "scorer", "act_head", "enc_embed", "enc_final_norm"):
        key = f"backward/{region}_recompute"
    elif ph == "forward":
        key = f"forward/{region or 'other'}"
    else:
        key = ph
    c = cat(e["name"]) if e.get("cat") == "kernel" else e.get("cat").replace("gpu_", "")
    by_phase_region[key] += e["dur"]; by_cat[c] += e["dur"]; by_name[e["name"][:90]] += e["dur"]
    if c == "memcpy":
        memcpy_kind[e["name"][:40]] += e["dur"]

n = len(steps)
out = [f"trace {path}", f"steps {n}  wall/step {wall / n / 1e3:.1f} ms  GPU busy {busy / wall * 100:.1f}%  kernel-sum/step {sum(by_cat.values()) / n / 1e3:.1f} ms"]
out.append("\n## GPU time by phase/region (ms/step, % of wall)")
for k, v in by_phase_region.most_common():
    out.append(f"  {k:32s} {v / n / 1e3:8.1f}  {v / wall * 100:5.1f}%")
out.append("\n## GPU time by kernel category")
for k, v in by_cat.most_common():
    out.append(f"  {k:32s} {v / n / 1e3:8.1f}  {v / wall * 100:5.1f}%")
out.append("\n## memcpy kinds")
for k, v in memcpy_kind.most_common():
    out.append(f"  {k:40s} {v / n / 1e3:8.1f}")
out.append("\n## top kernels")
for k, v in by_name.most_common(25):
    out.append(f"  {v / n / 1e3:8.2f} ms  {k}")
cpu_ops = collections.Counter()
for e in ev:
    if e.get("cat") in ("cuda_runtime", "cuda_driver") and t0 <= e["ts"] <= t1:
        cpu_ops[e["name"]] += e["dur"]
out.append("\n## CUDA runtime API time (ms/step)")
for k, v in cpu_ops.most_common(10):
    out.append(f"  {k:40s} {v / n / 1e3:8.1f}")
text = "\n".join(out)
print(text)
open(path.replace("trace.json", "summary.txt"), "w").write(text)

# CPU wall split on the thread that owns ProfilerStep: backward / optimizer / collate / rest (forward + trainer).
main_tid = steps[0]["tid"]
split = collections.Counter()
for a, b, nme in ranges.get(main_tid, ()):
    if nme in ("backward", "optimizer_step", "collate"):
        split[nme] += b - a
fw = sum(b - a for a, b, nme in ranges.get(main_tid, ()) if nme == "enc_layer")
split["forward enc_layer (main thread)"] = fw
split["rest"] = wall - split["backward"] - split["optimizer_step"] - split["collate"]
extra = ["\n## CPU wall split (ms/step, % of wall)"] + [f"  {k:32s} {v / n / 1e3:8.1f}  {v / wall * 100:5.1f}%" for k, v in split.items()]
print("\n".join(extra))
open(path.replace("trace.json", "summary.txt"), "a").write("\n".join(extra))
