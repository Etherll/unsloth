# Compare two parity.py outputs: per-batch |dloss|, grad-norm rel diff, batch-0 grad cosine / relL2.
import json, sys
import torch
from safetensors.torch import load_file

a, b = (json.load(open(f)) for f in sys.argv[1:3])
dl = [abs(x - y) for x, y in zip(a["loss"], b["loss"])]
dg = [abs(x - y) / max(abs(x), 1e-12) for x, y in zip(a["grad_norm"], b["grad_norm"])]
ga, gb = (load_file(f.replace(".json", "_g0.safetensors")) for f in sys.argv[1:3])
keys = sorted(set(ga) & set(gb))
va = torch.cat([ga[k].flatten() for k in keys]).double(); vb = torch.cat([gb[k].flatten() for k in keys]).double()
cos = float(torch.dot(va, vb) / (va.norm() * vb.norm()))
rel = float((va - vb).norm() / va.norm())
worst = min((float(torch.nn.functional.cosine_similarity(ga[k].flatten().double(), gb[k].flatten().double(), dim = 0)), k)
            for k in keys if ga[k].norm() > 0)
ok = max(dl) <= 0.03 and max(dg) <= 0.03
out = dict(max_dloss = max(dl), max_dgn_rel = max(dg), cos = cos, relL2 = rel, worst_param_cos = worst,
           missing = sorted(set(ga) ^ set(gb))[:5], G1 = "PASS" if ok else "FAIL")
print(json.dumps(out))
