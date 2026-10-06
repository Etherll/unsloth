# What forward does a Laya LoRA layer have on this tree, and does the lean forward match stock PEFT's
# Linear.forward bit for bit under bf16 autocast? usage: lean_probe.py <tree>
import os, sys

sys.path.insert(0, sys.argv[1])
import torch
import unsloth  # noqa: F401
from unsloth.models import decision as D
import peft.tuners.lora.layer as L

model, _ = D.FastDecisionModel.from_pretrained(f"{os.environ['LP_ROOT']}/model", dtype = torch.float32, use_gradient_checkpointing = False)
model = D.FastDecisionModel.get_peft_model(model, r = 4, lora_alpha = 8)
layer = next(m for m in model.encoder.modules() if isinstance(m, L.Linear))
saved = getattr(layer, "_unsloth_peft_forward", layer.forward)
f = getattr(saved, "__func__", saved)
print("LAYA_PROBE class forward:", L.Linear.forward.__module__, L.Linear.forward.__qualname__)
print("LAYA_PROBE saved forward:", f.__module__, f.__qualname__, "| stock:", f is L.Linear.forward)
print("LAYA_PROBE base_layer:", type(layer.base_layer).__module__, type(layer.base_layer).__qualname__, "| lora_A dtype", layer.lora_A["default"].weight.dtype)
torch.manual_seed(0)
with torch.no_grad():
    layer.lora_B["default"].weight.normal_()
x = torch.randn(2, 5, layer.in_features, device = layer.lora_A["default"].weight.device)
with torch.autocast("cuda", dtype = torch.bfloat16):
    outs = {"lean": layer.forward(x), "saved": saved(x), "stock": L.Linear.forward(layer, x)}
for a in ("saved", "stock"):
    print(f"LAYA_PROBE lean == {a}:", torch.equal(outs["lean"], outs[a]), outs[a].grad_fn.__class__.__name__ if outs[a].grad_fn else None,
          "max diff", (outs["lean"].float() - outs[a].float()).abs().max().item())
