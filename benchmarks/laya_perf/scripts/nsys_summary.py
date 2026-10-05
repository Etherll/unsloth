# Summarize an nsys capture (bench.py --nvtx --cuprof): per step GPU busy %, kernel count, NVTX phase wall,
# GPU time by phase (forward / backward / recompute = enc_layer inside backward / optimizer), top kernels,
# and CUDA API launch cost. usage: nsys_summary.py <report.nsys-rep>
import collections, os, sqlite3, subprocess, sys

rep = sys.argv[1]
db = rep.replace(".nsys-rep", ".sqlite")
if not os.path.exists(db):
    subprocess.run(["/usr/local/cuda-13.1/bin/nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db, rep],
                   check = True, capture_output = True)
c = sqlite3.connect(db)
q = lambda s: c.execute(s).fetchall()
tables = {r[0] for r in q("select name from sqlite_master where type='table'")}
names = dict(q("select id, value from StringIds"))
nvtx = q("select start, end, coalesce(text, (select value from StringIds where id = textId)), globalTid from NVTX_EVENTS where eventType in (59, 60) and end is not null")
steps = sorted((s, e) for s, e, t, _ in nvtx if t == "step")
kern = q("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL")
mem = q("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY") if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []
api = q("select start, end, nameId, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME")
t0, t1 = steps[0][0], steps[-1][1]
wall = t1 - t0
n = len(steps)
dev = sorted([(s, e) for s, e, *_ in kern if t0 <= s <= t1] + [(s, e) for s, e, *_ in mem if t0 <= s <= t1])
busy, cur = 0, None
for a, b in dev:
    if cur is None or a > cur[1]:
        busy += (cur[1] - cur[0]) if cur else 0
        cur = [a, b]
    else:
        cur[1] = max(cur[1], b)
busy += (cur[1] - cur[0]) if cur else 0
# Map each kernel to the NVTX phase active on the launching thread at launch time.
launch = {corr: (s, tid) for s, e, nid, corr, tid in api}
ranges = collections.defaultdict(list)
for s, e, t, tid in nvtx:
    if t in ("forward", "backward", "optimizer", "enc_layer"):
        ranges[tid].append((s, e, t))


def phase(corr):
    if corr not in launch:
        return "?"
    ts, tid = launch[corr]
    inside = {t for s, e, t in ranges.get(tid, ()) if s <= ts <= e}
    if "backward" in inside:
        return "backward/recompute" if "enc_layer" in inside else "backward"
    for t in ("optimizer", "forward"):
        if t in inside:
            return t
    return "other"


by_phase, by_kernel, count = collections.Counter(), collections.Counter(), 0
for s, e, nid, corr in kern:
    if t0 <= s <= t1:
        by_phase[phase(corr)] += e - s
        by_kernel[names.get(nid, str(nid))[:80]] += e - s
        count += 1
memk = collections.Counter()
for s, e, kind, nbytes in mem:
    if t0 <= s <= t1:
        memk[{1: "HtoD", 2: "DtoH", 8: "DtoD"}.get(kind, str(kind))] += e - s
apis = collections.Counter()
for s, e, nid, corr, tid in api:
    if t0 <= s <= t1:
        apis[names.get(nid, str(nid))] += e - s
ph_wall = collections.Counter()
for s, e, t, tid in nvtx:
    if t in ("forward", "backward", "optimizer") and t0 <= s <= t1:
        ph_wall[t] += e - s
out = [f"{rep}", f"steps {n}  wall/step {wall / n / 1e6:.1f} ms  GPU busy {busy / wall * 100:.1f}%  kernels/step {count / n:.0f}  kernel-sum/step {sum(by_phase.values()) / n / 1e6:.1f} ms"]
out.append("## NVTX wall per step (ms, % of wall)")
out += [f"  {k:24s} {v / n / 1e6:8.1f}  {v / wall * 100:5.1f}%" for k, v in ph_wall.most_common()]
out.append("## GPU kernel time by phase (ms/step, % of wall)")
out += [f"  {k:24s} {v / n / 1e6:8.1f}  {v / wall * 100:5.1f}%" for k, v in by_phase.most_common()]
out.append("## memcpy (ms/step)")
out += [f"  {k:24s} {v / n / 1e6:8.2f}" for k, v in memk.most_common()]
out.append("## top kernels (ms/step, % of kernel time)")
tot = sum(by_kernel.values())
out += [f"  {v / n / 1e6:7.2f}  {v / tot * 100:5.1f}%  {k}" for k, v in by_kernel.most_common(25)]
out.append("## CUDA runtime API (ms/step)")
out += [f"  {k[:50]:50s} {v / n / 1e6:8.1f}" for k, v in apis.most_common(8)]
text = "\n".join(out)
print(text)
open(rep.replace(".nsys-rep", "_summary.txt"), "w").write(text)
