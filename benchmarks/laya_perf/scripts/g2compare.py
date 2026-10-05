# G2: two bench.py runs (same seed/order, dropout on): mean |dloss| per 10-step window, holdout CE / accuracy.
import json, sys

a, b = (json.load(open(f)) for f in sys.argv[1:3])
la, lb = a["loss"], b["loss"]
n = min(len(la), len(lb))
win = [abs(sum(la[i:i + 10]) / len(la[i:i + 10]) - sum(lb[i:i + 10]) / len(lb[i:i + 10])) for i in range(0, n, 10)]
out = dict(steps = n, window_dloss = [round(w, 4) for w in win], max_window_dloss = max(win),
           max_step_dloss = max(abs(x - y) for x, y in zip(la, lb)))
ok = max(win) <= 0.03
if "holdout" in a and "holdout" in b:
    out["holdout_A"], out["holdout_B"] = a["holdout"], b["holdout"]
    out["d_ce"] = abs(a["holdout"]["loss"] - b["holdout"]["loss"])
    out["d_acc"] = abs(a["holdout"]["accuracy"] - b["holdout"]["accuracy"])
    ok = ok and out["d_ce"] <= 0.03 and out["d_acc"] <= 0.03
out["G2"] = "PASS" if ok else "FAIL"
print(json.dumps(out))
