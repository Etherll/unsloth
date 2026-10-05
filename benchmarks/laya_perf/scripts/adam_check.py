# Same params + same gradient sequence: fused vs foreach AdamW, param drift after N steps.
import torch
torch.manual_seed(0)
shapes = [(64, 1024), (1024, 64), (3072, 64), (64, 3072), (1024,), (1024, 1024), (1, 1024)] * 6
params0 = [torch.randn(s, device = "cuda") * 0.02 for s in shapes]
grads = [[torch.randn(s, device = "cuda") * (0.01 * (1 + 10 * (k % 7 == 3))) for s in shapes] for k in range(40)]
res = {}
for mode in ("foreach", "fused"):
    ps = [torch.nn.Parameter(p.clone()) for p in params0]
    groups = [{"params": ps[: len(ps) // 2], "lr": 8e-4, "weight_decay": 0.01}, {"params": ps[len(ps) // 2:], "lr": 1e-4, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, betas = (0.9, 0.999), eps = 1e-8, **({"fused": True} if mode == "fused" else {"foreach": True}))
    for g in grads:
        for p, gg in zip(ps, g):
            p.grad = gg.clone()
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
    res[mode] = torch.cat([p.detach().flatten() for p in ps])
d = (res["fused"] - res["foreach"]).abs()
upd = (res["foreach"] - torch.cat([p.flatten() for p in params0])).abs()
print("max |dparam|", d.max().item(), "rel to update", (d.norm() / upd.norm()).item())
