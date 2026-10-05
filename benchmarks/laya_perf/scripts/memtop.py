# Peak of a torch.cuda.memory snapshot: replay alloc/free trace, list live blocks at the peak by top frames.
import collections, pickle, sys
snap = pickle.load(open(sys.argv[1], "rb"))
events = [e for dev in snap["device_traces"] for e in dev]
live, cur, peak, peak_live, peak_i = {}, 0, 0, None, 0
for i, e in enumerate(events):
    if e["action"] == "alloc":
        live[e["addr"]] = e; cur += e["size"]
    elif e["action"] in ("free_completed",):
        if e["addr"] in live:
            cur -= live.pop(e["addr"])["size"]
    if cur > peak:
        peak, peak_live, peak_i = cur, dict(live), i
print(f"trace peak (only allocations recorded after history start) {peak/2**20:.0f} MiB at event {peak_i}/{len(events)}")
groups = collections.Counter()
for e in peak_live.values():
    frames = [f for f in e.get("frames", []) if not any(s in f["filename"] for s in ("torch/", "torch\\", "<frozen", "threading"))]
    key = " <- ".join(f"{f['name']}@{f['filename'].split('/')[-1].split(chr(92))[-1]}:{f['line']}" for f in frames[:3]) or "?"
    groups[key] += e["size"]
for k, v in groups.most_common(15):
    print(f"{v/2**20:9.1f} MiB  {k[:230]}")
seg = sum(s["total_size"] for s in snap["segments"]); alloc = sum(s["allocated_size"] for s in snap["segments"])
print(f"segments at dump: reserved {seg/2**20:.0f} MiB allocated {alloc/2**20:.0f} MiB")
