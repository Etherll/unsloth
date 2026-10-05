# Multi-seed G2 for runs whose single-seed PR-vs-PR control already exceeds the gate (fp16 GradScaler skips, the unstable
# LoRA recipe at mb2 x acc16). Per seed: max 10-step-window |dloss| between arms. Across seeds: the same gate on the
# seed-averaged loss curves, plus holdout CE / accuracy mean [range] per arm.
# usage: g2_seeds.py <results.json> <regex with named groups arm and seed> <ref arm> <arm> [<arm> ...]
import json, re, statistics as st, sys

R = json.load(open(sys.argv[1]))
pat, ref, arms = re.compile(sys.argv[2]), sys.argv[3], sys.argv[4:]
runs = {}
for name, r in R.items():
    m = pat.fullmatch(name)
    if m and isinstance(r, dict) and "loss" in r:
        runs.setdefault(m["arm"], {})[m["seed"]] = r


def windows(loss):
    return [sum(loss[i:i + 10]) / len(loss[i:i + 10]) for i in range(0, len(loss), 10)]


def gap(a, b):
    n = min(len(a), len(b))
    return max(abs(x - y) for x, y in zip(windows(a[:n]), windows(b[:n])))


def mean_curve(arm, seeds):
    n = min(len(runs[arm][s]["loss"]) for s in seeds)
    return [st.mean(runs[arm][s]["loss"][i] for s in seeds) for i in range(n)]


def holdout(arm, seeds):
    ce = [runs[arm][s]["holdout"]["loss"] for s in seeds]
    acc = [runs[arm][s]["holdout"]["accuracy"] for s in seeds]
    return f"CE {st.mean(ce):.3f} [{min(ce):.3f}..{max(ce):.3f}]  acc {st.mean(acc):.3f} [{min(acc):.3f}..{max(acc):.3f}]"


for arm in arms:
    seeds = sorted(set(runs.get(ref, {})) & set(runs.get(arm, {})))
    if not seeds:
        print(f"{arm}: no seeds shared with {ref}")
        continue
    per = {s: round(gap(runs[ref][s]["loss"], runs[arm][s]["loss"]), 4) for s in seeds}
    avg = gap(mean_curve(ref, seeds), mean_curve(arm, seeds))
    dce = abs(st.mean(runs[ref][s]["holdout"]["loss"] for s in seeds) - st.mean(runs[arm][s]["holdout"]["loss"] for s in seeds))
    dacc = abs(st.mean(runs[ref][s]["holdout"]["accuracy"] for s in seeds) - st.mean(runs[arm][s]["holdout"]["accuracy"] for s in seeds))
    print(f"{ref} vs {arm}  seeds {seeds}  per-seed window {per}")
    print(f"  seed-averaged window {avg:.4f} ({'PASS' if avg <= 0.03 else 'FAIL'})  holdout dCE {dce:.4f}  dacc {dacc:.4f}")
    print(f"  {ref:6s} {holdout(ref, seeds)}")
    print(f"  {arm:6s} {holdout(arm, seeds)}")
