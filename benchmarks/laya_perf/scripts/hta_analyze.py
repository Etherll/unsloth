# HTA on a bench.py --prof trace dir: temporal breakdown, kernel-type breakdown, top kernels,
# CUDA launch delay (CPU launch -> GPU start) and queue length. Writes <dir>/hta.txt.
import io, json, os, shutil, sys, contextlib
import pandas as pd
from hta.trace_analysis import TraceAnalysis

src = sys.argv[1]
work = os.path.join(src, "hta")
os.makedirs(work, exist_ok = True)
tr = json.load(open(os.path.join(src, "trace.json")))
tr.setdefault("distributedInfo", {"rank": 0, "world_size": 1})
json.dump(tr, open(os.path.join(work, "rank-0.json"), "w"))
pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 90)
out = io.StringIO()
with contextlib.redirect_stderr(io.StringIO()):
    ta = TraceAnalysis(trace_dir = work)
    for title, fn in [
        ("temporal breakdown", lambda: ta.get_temporal_breakdown(visualize = False)),
        ("kernel type breakdown", lambda: ta.get_gpu_kernel_breakdown(visualize = False, num_kernels = 15)),
        ("idle time breakdown", lambda: ta.get_idle_time_breakdown(ranks = [0], visualize = False)),
        ("queue length summary", lambda: ta.get_queue_length_summary(ranks = [0])),
        ("cuda kernel launch stats", lambda: ta.get_cuda_kernel_launch_stats(ranks = [0], visualize = False)),
    ]:
        try:
            r = fn()
            out.write(f"\n## {title}\n")
            for part in (r if isinstance(r, tuple) else (r,)):
                if isinstance(part, dict):
                    for k, v in part.items():
                        out.write(f"[{k}]\n{v.describe() if title == 'cuda kernel launch stats' else v}\n")
                else:
                    out.write(f"{part}\n")
        except Exception as exc:
            out.write(f"\n## {title}: failed {type(exc).__name__}: {str(exc)[:200]}\n")
text = out.getvalue()
open(os.path.join(src, "hta.txt"), "w").write(text)
print(text)
shutil.rmtree(work, ignore_errors = True)
