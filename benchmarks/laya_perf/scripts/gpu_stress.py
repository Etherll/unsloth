# Keep the GPU busy with large bf16 matmuls (contention for race hunting), until killed or DURATION s.
import os, sys, time, torch
a = torch.randn(8192, 8192, device = "cuda", dtype = torch.bfloat16)
t0 = time.time()
while time.time() - t0 < float(sys.argv[1] if len(sys.argv) > 1 else 3600):
    for _ in range(20):
        a = (a @ a).clamp_(-1, 1)
    torch.cuda.synchronize()
