# Build the decision items once with the PR's own build_dataset/split_holdout.
import json, os, random, time, torch, pandas as pd
from huggingface_hub import snapshot_download
from unsloth.models.decision import FastDecisionModel

LP = os.environ["LP_ROOT"]
# Same dataset revision as the Windows run (c76749e), from the HF cache.
snap = snapshot_download("LocalLLaMA/typed-decisions", repo_type = "dataset", revision = "c76749e", allow_patterns = ["all/*"])
rows = pd.read_parquet(os.path.join(snap, "all", "train-00000-of-00001.parquet")).to_dict("records")
t = time.time()
model, tok = FastDecisionModel.from_pretrained(f"{LP}/model", full_finetuning = True)
items, report = FastDecisionModel.build_dataset(rows, tok, model)
train, hold = FastDecisionModel.split_holdout(items, seed = 3407)
bench = random.Random(0).sample(train, 1280)
lens = sorted(len(i["input_ids"]) for i in train)
info = dict(rows = len(rows), items = len(items), report = report, train = len(train), holdout = len(hold),
            len_min = lens[0], len_med = lens[len(lens)//2], len_max = lens[-1], pad_id = tok.pad_token_id,
            decision_config = {k: v for k, v in model.decision_config.items() if k in ("max_len", "head_max_len")},
            seconds = time.time() - t)
torch.save({"train": train, "holdout": hold, "bench": bench, "info": info}, f"{LP}/data/items.pt")
print(json.dumps(info, indent = 1, default = str))
