# Long-context variant of the bench data: every item gets filler context (question tokens of other items) inserted
# right after [CLS], so its total length is log-uniform in [LO, HI]; option markers shift with it, so each item is still
# a valid decision. Writes $LP_ROOT/data/items_long.pt with the same keys bench.py / parity.py read.
import math, os, random, statistics, torch

LO, HI = int(os.environ.get("LONG_LO", 512)), int(os.environ.get("LONG_HI", 4096))
LP = os.environ["LP_ROOT"]
data = torch.load(f"{LP}/data/items.pt", weights_only = False)
rng = random.Random(0)
pool = [t for item in data["train"] for t in item["input_ids"][1:item["markers"][0]] if t < 50280]


def lengthen(item):
    ids = item["input_ids"]
    want = int(math.exp(rng.uniform(math.log(LO), math.log(HI))))
    extra = max(0, want - len(ids))
    start = rng.randrange(len(pool) - extra) if extra else 0
    filler = pool[start:start + extra]
    return {**item, "input_ids": ids[:1] + filler + ids[1:], "markers": [m + extra for m in item["markers"]]}


out = {k: [lengthen(x) for x in data[k]] for k in ("train", "bench", "holdout")}
out["info"] = {**data.get("info", {}), "long": [LO, HI]}
torch.save(out, f"{LP}/data/items_long.pt")
lengths = [len(x["input_ids"]) for x in out["train"]]
print("LONG_DATA", len(lengths), "min", min(lengths), "median", statistics.median(lengths), "max", max(lengths))
