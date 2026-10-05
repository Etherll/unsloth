# G1: fixed micro-batches, dropout off, same weights: loss + grad per batch through DecisionTrainer.compute_loss.
import argparse, json, os, random, sys

p = argparse.ArgumentParser()
p.add_argument("--tree", required = True)
p.add_argument("--mode", default = "full", choices = ["full", "lora"])
p.add_argument("--precision", default = "bf16", choices = ["bf16", "fp32"])
p.add_argument("--gc", default = "unsloth", choices = ["unsloth", "true", "off"])
p.add_argument("--opts", default = "")
p.add_argument("--batches", type = int, default = 8)
p.add_argument("--bs", type = int, default = 4)
p.add_argument("--out", required = True)
p.add_argument("--data", default = "items.pt")
args = p.parse_args()
# "bf16" means the half-precision autocast this GPU trains in: bf16, or fp16 where bf16 is unsupported (T4).
LP = os.environ["LP_ROOT"]
sys.path.insert(0, args.tree)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import torch
import unsloth  # noqa: F401
from unsloth.models import decision as D
from transformers import TrainingArguments
from safetensors.torch import save_file
import opts as O

names = [o for o in args.opts.split(",") if o]
O.pre(names, D, args)
data = torch.load(f"{LP}/data/{args.data}", weights_only = False)
gc = {"unsloth": "unsloth", "true": True, "off": False}[args.gc]
kw = O.load_kwargs(names, args)
if args.precision == "fp32" and args.mode == "lora":
    kw["dtype"] = torch.float32
model, tok = D.FastDecisionModel.from_pretrained(
    f"{LP}/model", full_finetuning = args.mode == "full", use_gradient_checkpointing = gc, **kw
)
if args.mode == "lora":
    model = D.FastDecisionModel.get_peft_model(
        model, r = 64, lora_alpha = 64, lora_dropout = 0.0, use_gradient_checkpointing = gc, random_state = 3407
    )
    # Non-zero LoRA B so encoder adapter grads are not trivially tied to B = 0.
    g = torch.Generator().manual_seed(0)
    for n, prm in model.named_parameters():
        if "lora_B" in n:
            prm.data.copy_(torch.randn(prm.shape, generator = g).to(prm) * 1e-3)
O.dropout_off(model)
model = O.post(names, model, D, args)
model.train()
trainer = D.DecisionTrainer(
    model = model,
    args = TrainingArguments(output_dir = f"{os.environ.get('TMPDIR', LP + '/tmp')}/parity-{os.getpid()}", per_device_train_batch_size = args.bs,
                             bf16 = args.precision == "bf16" and D.is_bfloat16_supported(),
                             fp16 = args.precision == "bf16" and not D.is_bfloat16_supported(), report_to = "none", seed = 3407),
    train_dataset = data["bench"], processing_class = tok,
)
O.trainer(names, trainer, D, args)
pool = random.Random(1).sample(data["bench"], args.batches * args.bs)
batches = [pool[i * args.bs:(i + 1) * args.bs] for i in range(args.batches)]
collate = D.DecisionDataCollator(tok.pad_token_id)
dev = next(model.parameters()).device
params = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
res = {"loss": [], "grad_norm": [], "tree": args.tree, "opts": names, "mode": args.mode, "precision": args.precision,
       "half": "bf16" if D.is_bfloat16_supported() else "fp16", "gpu": torch.cuda.get_device_name()}
for i, b in enumerate(batches):
    inputs = {k: v.to(dev) for k, v in collate(b).items()}
    half = torch.bfloat16 if D.is_bfloat16_supported() else torch.float16
    with torch.autocast("cuda", dtype = half, enabled = args.precision == "bf16"):
        loss = trainer.compute_loss(model, inputs)
    trainer.accelerator.backward(loss)  # the trainer's own backward (perf tree: cuDNN guard over the GC recompute)
    gn = torch.sqrt(sum(q.grad.float().pow(2).sum() for _, q in params if q.grad is not None))
    res["loss"].append(float(loss)); res["grad_norm"].append(float(gn))
    if i == 0:
        save_file({n: q.grad.detach().float().cpu().contiguous() for n, q in params if q.grad is not None},
                  args.out.replace(".json", "_g0.safetensors"))
    model.zero_grad(set_to_none = True)
json.dump(res, open(args.out, "w"), indent = 1)
print(json.dumps(res))
