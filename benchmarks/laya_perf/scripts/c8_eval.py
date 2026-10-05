# C8: time evaluate + calibrate on the 400 holdout decisions; dump logits to compare across runs.
import argparse, json, os, sys, time

p = argparse.ArgumentParser()
p.add_argument("--tree", required = True)
p.add_argument("--opts", default = "")
p.add_argument("--mode", default = "full")
p.add_argument("--out", required = True)
args = p.parse_args()
LP = os.environ["LP_ROOT"]
sys.path.insert(0, args.tree)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import torch
import unsloth  # noqa: F401
from unsloth.models import decision as D
import opts as O

names = [o for o in args.opts.split(",") if o]
O.pre(names, D, args)
data = torch.load(f"{LP}/data/items.pt", weights_only = False)
model, tok = D.FastDecisionModel.from_pretrained(f"{LP}/model", full_finetuning = args.mode == "full")
model = O.post(names, model, D, args)
hold = data["holdout"]
D.FastDecisionModel.evaluate(model, tok, hold[:32])  # warm up kernels
times = {}
for rep in range(3):
    torch.cuda.synchronize(); t = time.perf_counter()
    logits = D._logits(model, hold, tok.pad_token_id)
    torch.cuda.synchronize(); times.setdefault("logits", []).append(time.perf_counter() - t)
t = time.perf_counter(); ev = D.FastDecisionModel.evaluate(model, tok, hold); times["evaluate"] = time.perf_counter() - t
t = time.perf_counter(); cal = D.FastDecisionModel.calibrate(model, tok, hold); times["calibrate"] = time.perf_counter() - t
torch.save([z.clone() for z in logits], args.out.replace(".json", "_logits.pt"))
json.dump({"times": times, "evaluate": ev, "calibrate": cal, "opts": names}, open(args.out, "w"), indent = 1)
print(json.dumps({"times": times, "evaluate": ev, "calibrate": {k: v for k, v in cal.items()}}))
