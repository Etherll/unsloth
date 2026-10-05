# Attention microbench (triton do_bench, fwd+bwd) over the real micro-batch lengths of the bench subset,
# ModernBERT-large shapes (16 heads x 64, bf16). Global layers: key padding only; local layers: |i-j| <= 64.
# Variants that cannot run in this environment are reported as unavailable. Writes JSON.
import argparse, json, os, sys

p = argparse.ArgumentParser()
p.add_argument("--mb", type = int, default = 2)
p.add_argument("--batches", type = int, default = 32)
p.add_argument("--variants", default = "sdpa_eff,sdpa_cudnn,sdpa_math_free,varlen_fa2,xf_padded,xf_bd,fa4_varlen")
p.add_argument("--out", required = True)
p.add_argument("--data", default = "items.pt")
p.add_argument("--random", action = "store_true", help = "random batches (SentenceTransformer-style) instead of length-grouped")
args = p.parse_args()
sys.path.insert(0, os.environ["PERF"])
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton.testing import do_bench

H, D, W = 16, 64, 64
dev = "cuda"
dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
data = torch.load(f"{os.environ['LP_ROOT']}/data/{args.data}", weights_only = False)["bench"]
from transformers.trainer_pt_utils import get_length_grouped_indices

lens = [len(i["input_ids"]) for i in data]
order = get_length_grouped_indices(lens, args.mb, generator = torch.Generator().manual_seed(3407))
if args.random:
    order = torch.randperm(len(lens), generator = torch.Generator().manual_seed(3407)).tolist()
mbs = [[lens[j] for j in order[i:i + args.mb]] for i in range(0, args.mb * args.batches, args.mb)]
torch.manual_seed(0)


def padded(ls):
    B, L = len(ls), max(ls)
    q, k, v = (torch.randn(B, H, L, D, device = dev, dtype = dt, requires_grad = True) for _ in range(3))
    keep = torch.zeros(B, L, dtype = torch.bool, device = dev)
    for b, n in enumerate(ls):
        keep[b, :n] = True
    i = torch.arange(L, device = dev)
    glob = keep[:, None, None, :].expand(B, 1, L, L)
    loc = glob & ((i[:, None] - i[None, :]).abs() <= W)[None, None]
    return q, k, v, glob.contiguous(), loc.contiguous(), keep


def packed(ls):
    T = sum(ls)
    q, k, v = (torch.randn(T, H, D, device = dev, dtype = dt, requires_grad = True) for _ in range(3))
    cu = torch.tensor([0] + list(torch.tensor(ls).cumsum(0)), device = dev, dtype = torch.int32)
    return q, k, v, cu, max(ls)


def run_padded(fn):
    sets = [padded(ls) for ls in mbs]

    def step():
        for q, k, v, g, l, keep in sets:
            for mask in (g, l):
                o = fn(q, k, v, mask)
                o.backward(torch.ones_like(o))
    return step


def sdpa(backend):
    def fn(q, k, v, mask):
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(q, k, v, attn_mask = mask)
    return fn


def varlen_fa2():
    from torch.nn.attention.varlen import varlen_attn
    sets = [packed(ls) for ls in mbs]

    def step():
        for q, k, v, cu, mx in sets:
            for win in ((-1, -1), (W, W)):
                o = varlen_attn(q, k, v, cu, cu, mx, mx, window_size = win)
                o.backward(torch.ones_like(o))
    return step


def varlen_global():
    # Global layers only and no window argument, so it also runs on torch 2.10 (window_size arrived in 2.11).
    from torch.nn.attention.varlen import varlen_attn
    sets = [packed(ls) for ls in mbs]

    def step():
        for q, k, v, cu, mx in sets:
            o = varlen_attn(q, k, v, cu, cu, mx, mx)
            o.backward(torch.ones_like(o))
    return step


def sdpa_eff_global():
    sets = [padded(ls) for ls in mbs]

    def step():
        for q, k, v, g, l, keep in sets:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                o = F.scaled_dot_product_attention(q, k, v, attn_mask = g)
            o.backward(torch.ones_like(o))
    return step


def sdpa_eff_local():
    sets = [padded(ls) for ls in mbs]

    def step():
        for q, k, v, g, l, keep in sets:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                o = F.scaled_dot_product_attention(q, k, v, attn_mask = l)
            o.backward(torch.ones_like(o))
    return step


def _flex(local):
    # Packed tokens with a document (+ |i-j| <= W window) block mask: Triton, so it also runs on sm75 (T4).
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    fa = torch.compile(flex_attention, dynamic = False)
    sets = []
    for ls in mbs:
        T = sum(ls)
        q, k, v = (torch.randn(1, H, T, D, device = dev, dtype = dt, requires_grad = True) for _ in range(3))
        doc = torch.repeat_interleave(torch.arange(len(ls), device = dev), torch.tensor(ls, device = dev))
        pos = torch.cat([torch.arange(n, device = dev) for n in ls])
        if local:
            mod = lambda b, h, qi, ki: (doc[qi] == doc[ki]) & ((pos[qi] - pos[ki]).abs() <= W)
        else:
            mod = lambda b, h, qi, ki: doc[qi] == doc[ki]
        sets.append((q, k, v, create_block_mask(mod, None, None, T, T, device = dev)))

    def step():
        for q, k, v, bm in sets:
            o = fa(q, k, v, block_mask = bm)
            o.backward(torch.ones_like(o))
    return step


def xf_padded():
    import xformers.ops as xops
    sets = []
    for ls in mbs:
        q, k, v, g, l, keep = padded(ls)
        to = lambda x: x.detach().transpose(1, 2).contiguous().requires_grad_()
        bias = lambda m: torch.zeros(m.shape, device = dev, dtype = dt).masked_fill(~m, float("-inf")).expand(-1, H, -1, -1)
        sets.append((to(q), to(k), to(v), bias(g), bias(l)))

    def step():
        for q, k, v, g, l in sets:
            for b in (g, l):
                o = xops.memory_efficient_attention(q, k, v, attn_bias = b)
                o.backward(torch.ones_like(o))
    return step


def xf_bd():
    # Global layers only: xformers has no non-causal block-diagonal local mask with a backward.
    import xformers.ops as xops
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    sets = []
    for ls in mbs:
        q, k, v, cu, mx = packed(ls)
        sets.append((q[None], k[None], v[None], BlockDiagonalMask.from_seqlens(ls)))

    def step():
        for q, k, v, bias in sets:
            o = xops.memory_efficient_attention(q, k, v, attn_bias = bias)
            o.backward(torch.ones_like(o))
    return step


def fa4_varlen():
    from torch.nn.attention import activate_flash_attention_impl
    activate_flash_attention_impl("FA4")
    return varlen_fa2()


res = {"mb": args.mb, "batches": args.batches, "tokens": sum(map(sum, mbs)), "padded_tokens": sum(max(m) * len(m) for m in mbs),
       "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
builders = {
    "sdpa_eff": lambda: run_padded(sdpa(SDPBackend.EFFICIENT_ATTENTION)),
    "sdpa_cudnn": lambda: run_padded(sdpa(SDPBackend.CUDNN_ATTENTION)),
    "sdpa_math_free": lambda: run_padded(sdpa(SDPBackend.MATH)),
    "varlen_fa2": varlen_fa2, "varlen_global": varlen_global, "sdpa_eff_global": sdpa_eff_global, "sdpa_eff_local": sdpa_eff_local, "flex_global": lambda: _flex(False), "flex_local": lambda: _flex(True), "xf_padded": xf_padded, "xf_bd": xf_bd, "fa4_varlen": fa4_varlen,
}
for name in args.variants.split(","):
    try:
        step = builders[name]()
        step()
        torch.cuda.synchronize()
        ms = do_bench(step, warmup = 50, rep = 300)
        res[name] = {"ms": ms, "ms_per_mb": ms / args.batches}
        if name == "xf_bd":
            import xformers.ops as xops
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask
            q, k, v, cu, mx = packed(mbs[0])
            from xformers.ops.fmha.common import Inputs
            from xformers.ops.fmha.dispatch import _dispatch_bw, _dispatch_fw
            inp = Inputs(q[None], k[None], v[None], BlockDiagonalMask.from_seqlens(mbs[0]))
            res[name]["ops"] = [_dispatch_fw(inp, True).NAME, _dispatch_bw(inp, True).NAME]
    except Exception as exc:
        res[name] = {"unavailable": f"{type(exc).__name__}: {str(exc)[:300]}"}
    print(name, res[name], flush = True)
json.dump(res, open(args.out, "w"), indent = 1)
