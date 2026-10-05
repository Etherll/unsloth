# Re-analyze run_ab dirs with a burst-robust step time: 25th percentile of post-warmup steps.
import glob, json, os, statistics, sys
def p25(r, warm = 4):
    t = sorted(r["step_times"][warm:-1] or r["step_times"])
    return t[len(t) // 4]
roots = sys.argv[1:] or ["E:/laya-perf/out/ab", "E:/laya-perf/out/stack", "E:/laya-perf/out/abw"]
for root in roots:
    for d in sorted(glob.glob(f"{root}/*/")):
        pairs = []
        i = 0
        while os.path.exists(f"{d}A{i}.json") and os.path.exists(f"{d}B{i}.json"):
            a, b = json.load(open(f"{d}A{i}.json")), json.load(open(f"{d}B{i}.json"))
            pairs.append((p25(a), p25(b), a["s_step"], b["s_step"]))
            i += 1
        if not pairs:
            continue
        sp = [x / y - 1 for x, y, _, _ in pairs]
        sm = [x / y - 1 for _, _, x, y in pairs]
        out = dict(name = os.path.basename(d.rstrip("/\\")), root = os.path.basename(root), pairs = len(pairs),
                   p25_speedup = statistics.median(sp), p25_pairs = [round(x * 100, 1) for x in sp],
                   median_speedup = statistics.median(sm), A_p25 = statistics.median(p[0] for p in pairs), B_p25 = statistics.median(p[1] for p in pairs))
        json.dump(out, open(f"{d}robust.json", "w"), indent = 1)
        print(f"{out['root']}/{out['name']:16s} p25 {out['p25_speedup']*100:+6.1f}% {out['p25_pairs']}  median {out['median_speedup']*100:+6.1f}%  A {out['A_p25']:.3f} B {out['B_p25']:.3f}")
