# Which SDPA backend does PyTorch pick here for the decision model's attention (bf16/fp16, masked)?
import argparse, json, torch
from torch.nn.attention import SDPBackend
p = argparse.ArgumentParser(); p.add_argument("--out", required = True); a = p.parse_args()
half = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
names = {int(v): k for k, v in SDPBackend.__members__.items()}
res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "half": str(half)}
for tag, (B, H, L, D, masked) in {"encoder_A": (2, 16, 300, 64, True), "encoder_B": (8, 16, 300, 64, True),
                                    "head_A": (2, 16, 300, 64, True), "nomask": (2, 16, 300, 64, False)}.items():
    q = torch.randn(B, H, L, D, device = "cuda", dtype = half, requires_grad = True)
    m = torch.ones(B, 1, L, L, device = "cuda", dtype = torch.bool) if masked else None
    try:
        res[tag] = names.get(int(torch._fused_sdp_choice(q, q, q, m, 0.0, False)), "?")
    except Exception as exc:
        res[tag] = f"err {type(exc).__name__}"
json.dump(res, open(a.out, "w")); print(res)
