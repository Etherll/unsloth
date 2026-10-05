
Continue the Laya decision-model perf work from the attached HANDOFF.md on this shared AWS box.
Read HANDOFF.md and benchmarks/laya_perf/PLAN.md first. Do not re-derive anything they already settle.

GOAL
Two things:
1. Confirm the existing 5 commits on Linux.
2. Find and fix what actually slows laya training down, using profiler evidence, not guesses.
Keep within the PLAN.md decision rules and parity gates.
Scope: Unsloth core only (unsloth/models/decision.py, new unsloth/models/_decision_*.py,
tests/test_decision_model.py).

GOAL AMENDMENT
The 2-epoch requirement in HANDOFF §10 (condition 4) and §7.4 is replaced by the LONG-RUN CHECKS
below. Wherever HANDOFF says "2-epoch," use LONG-RUN CHECKS instead.

GPU RULES (hard constraints)
- Use ONLY physical GPUs 5, 6 and 7.
  - Set CUDA_DEVICE_ORDER=PCI_BUS_ID.
  - Pin each lane with CUDA_VISIBLE_DEVICES=5, 6 or 7 (one GPU per lane).
  - Never touch GPUs 0–4, even if 5–7 are busy.
- A GPU counts as free only if it has no compute process at all:
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i <N>` returns nothing,
  AND memory used is under ~1 GB,
  AND utilization is 0% across 3 checks 10 s apart.
- Check before EVERY job, not just at startup; the machine is shared.
  - If a lane's GPU is busy, wait and re-check every 2 min. Do not borrow another GPU.
  - If it is still busy after 30 min, tell me and keep the free lanes going.
  - Never kill, renice or interfere with processes you didn't start.
- Log nvidia-smi state and system load average at the start and end of every timing job.
  If another user's process appeared on 5–7 mid-run, discard that measurement and redo it.
- Other users on GPUs 0–4 share the CPU and PCIe:
  - record their presence in timing logs;
  - re-run the interference control if load changes a lot;
  - prefer quiet periods for the final stacked A/B.

SETUP (time box ≤1h)
- Python 3.12 venv. Venvs, HF cache, model, dataset and outputs all go on the large volume.
- Record torch, triton, transformers, peft, unsloth_zoo, CUDA, driver, the GPU model and the CPU count.
  Check whether the zoo misc.py __name__ bug (handoff §5) still exists.
- Write env-linux.sh and fix the harness's hardcoded paths (handoff §7.1).
  Replace the Windows CPU affinity with per-lane core ranges.
- Give each lane its own TMPDIR (Unsloth pins the inductor cache there).
  Otherwise one lane warms the cache and another's cold-compile numbers are wrong.
- Dev tools in the venv only, never as unsloth dependencies:
  nsys, ncu, holistictraceanalysis (HTA), py-spy, tlparse.

LANES (run concurrently; each lane has its own GPU, taskset core range, TMPDIR and results dir
under benchmarks/laya_perf/results/linux/gpu<N>/)
- GPU5, TIMING (largest dedicated core range; geometry A is launch-bound, i.e. CPU-sensitive):
  1. Interference control: one PR full-A pair with GPU6/7 idle, then one with them busy.
     If they differ by more than 2%, run all timing work only while GPU6/7 are idle.
  2. Re-baseline the PR (full and LoRA × A and B) and the official script (VRAM ceiling).
  3. Stacked A/B for all 4 combos.
  4. Compile cost: cold (inductor and triton cache moved aside) vs warm, plus the eval-mode recompile.
  5. Full ABAB for anything the discovery lane promotes.
- GPU6, DISCOVERY: the discovery loop and the xformers tests below.
  Set TORCHINDUCTOR_COMPILE_THREADS low and stay in its own core range so it can't skew GPU5.
- GPU7, NUMERICS AND VARIANCE:
  - pytest, then G1 (fp32 and bf16) and G2 for the current stack and every promoted change;
  - LONG-RUN CHECKS (full FT PR/perf/control, LoRA seeds);
  - if GPU6 finishes discovery early, it can take extra LoRA seeds.
- PR and perf for any one comparison always run on the SAME GPU. Never compare across GPUs.

DISCOVERY LOOP (GPU6; repeat until stop condition)
1. Profile the current best build at A and B:
   - torch.profiler (record_shapes, with_stack) → analyze with HTA
     (temporal breakdown, kernel breakdown, CUDA launch delay / queue length);
   - nsys with NVTX ranges around forward / backward / GC recompute / optimizer
     (`nsys profile -t cuda,nvtx,osrt,cublas`);
   - ncu on the top 10 kernels by time (memory- vs compute-bound);
   - a memory snapshot (`torch.cuda.memory._record_memory_history` → memory_viz)
     to see what makes up the VRAM peak;
   - py-spy --native for CPU time at geometry A;
   - TORCH_LOGS="graph_breaks,recompiles,perf_hints" plus tlparse for any compiled run.
2. Keep a ranked bottleneck table in REPORT.md:
   item | % of step | evidence (trace file and location) | hypothesis | expected gain | exactness risk.
3. Take the top item. Prototype it with a microbench (triton.testing.do_bench) first.
   - Only if the microbench wins: run full ABAB on GPU5 and G1 on GPU7.
   - APPLY / REJECT by PLAN.md rules. Commit if applied.
   - Re-profile, because the bottleneck moves.
4. Stop when 2 consecutive top items are REJECTED, or the top remaining item is under 3% of step
   and not the VRAM peak.
- Rule for nominating items: nothing is tried unless a trace shows it costs ≥3% of step time
  or is the VRAM peak.
- Already-rejected candidates (C2, C3 dense/flex, C6, C7, C9, rope_once) stay rejected
  unless a Linux profile shows the situation changed. If so, explain why before retesting.

STARTING HYPOTHESES (from the Windows Phase-1 profile; verify on Linux before acting)
- Geometry A is launch-bound (~28k launches/step, GPU ~19–27% busy).
  - C12 CUDA graphs failed with "output overwritten" under GC recompute.
    Retry with torch.compiler.cudagraph_mark_step_begin() / output cloning,
    or with checkpointing inside the compiled region so cudagraph trees work.
- Autocast fp32→bf16 weight casts on every micro-batch (~25% of GPU time is elementwise).
  - LoRA: the frozen base Linear weights could be stored in bf16. Autocast casts them anyway,
    so this should be bit-exact for Linear layers; LayerNorm stays fp32.
    Verify with torch.equal.
  - Full FT: cache the bf16 weights once per optimizer step instead of per micro-batch.
    Costs ~0.8 GB; check it against the official script's VRAM ceiling.
- GC recompute (~21% of GPU time):
  - try selective activation checkpointing (save GEMM outputs, recompute cheap elementwise ops);
  - or use the compile activation memory budget;
  - must stay ≤ the official VRAM ceiling.
- GC offload memcpy (~6%): check for pinned memory, non_blocking copies, and stream overlap.
- GEMM (~55%): check shape alignment, fp32 GEMMs that slipped past autocast,
  bias+GELU epilogue fusion, and max-autotune on this GPU.
- LoRA: lots of tiny A/B GEMMs.
  - Retest compile for LoRA now that the lean forward and SDPA fix are in (handoff §6 Q6).
  - Consider a fused LoRA kernel.
- Custom Triton kernels (fused LN, GEGLU, RoPE) are in scope if they live in
  unsloth/models/_decision_*.py.
- Anything needing a new dependency (Liger, torchao FP8, Apex) is REPORT-ONLY:
  measure it if cheap, don't commit it.
  FP8 only matters on H100/B200; skip it on other GPUs.

XFORMERS (C3 retest + attention backend comparison, on GPU6)
- Install it in a SEPARATE venv (a clone of the main one).
  - xformers wheels pin a torch version and can silently downgrade torch.
  - Install the wheel built for the exact recorded torch, then verify that torch.__version__
    and the CUDA version are unchanged and that `python -m xformers.info` works.
  - If no matching wheel exists, try building from source once (≤30 min). Otherwise report it as unavailable.
- Test:
  (a) memory_efficient_attention as the encoder attention, padded batch;
  (b) unpadded/packed with BlockDiagonalMask (varlen). This is the C3 path that was untestable on Windows;
  (c) compare against the current SDPA path, stock FA2 and FA2 varlen.
- Padding is only ~0.8% (length-grouped sampler), so any win must come from the kernel itself.
  Use the ncu/nsys traces to show why it won or lost.
- Run G1 for any variant that gets promoted to full A/B.
- If it clears an APPLY rule, propose it as an optional, availability-guarded path in
  unsloth/models/_decision_*.py with an SDPA fallback, and ASK ME before committing.
  Otherwise, record it as REJECTED / REPORT-ONLY with the numbers.

STOP RULES (don't over-measure)
- A/B: start with 3 ABAB fresh-process pairs, using p25 step time.
  Add pairs (up to 5) only if pair-to-pair spread is over 3% or the result is within 1% of a rule threshold.
- Compile break-even = cold compile cost ÷ (seconds saved per decision). Compute it from measured
  numbers; don't run extra experiments for it.
  - If break-even is under ~3k decisions, keep always-compile.
  - Otherwise, report the numbers to me. Do not add a threshold yourself.
- If a job fails twice for the same reason, stop that job and log it. Don't loop.
- Any idea outside these hypotheses that lacks trace evidence: write it under "Open ideas"
  in REPORT.md and move on.

GATES
- G1 fp32: per-batch |Δloss| ≤ 0.03 and grad-norm relative diff ≤ 0.03.
- G1 bf16: loss ≤ 0.03, and grads no further from fp32 than the PR's own bf16 grads.
- G2: 10-step-window |Δloss| ≤ 0.03.
- Long run: holdout CE and accuracy within 0.03 of the PR at the final checkpoint (see LONG-RUN CHECKS).
- bf16 is nondeterministic: always include a PR-vs-PR control.
- Never run pytest or other GPU work on GPU5 during an A/B.

LONG-RUN CHECKS (replace the 2-epoch runs)
- Default length: 200 optimizer steps at geometry A (the official default), with the same recipe
  as the PR (dropout on, same lr/schedule, same seed handling).
- Run a holdout eval (CE and accuracy, via the normal evaluate/_logits path) at steps 60, 120 and 200.
- Log per checkpoint: wall time, s/step p25, peak alloc/reserved, train-loss window mean,
  holdout CE and accuracy.
- Pass: holdout CE and accuracy at the final checkpoint are within 0.03 of the PR,
  judged against the PR-vs-PR control spread.
- EXTEND to 400, then up to 600 steps, only if one of these holds:
  (a) the perf-vs-PR speed or VRAM advantage keeps growing between checkpoints
      (e.g. compile or caching amortizing), so the longer run shows the true steady-state gain; or
  (b) the CE/accuracy difference is still trending between checkpoints instead of flat,
      so 200 steps can't settle it.
  Stop extending as soon as two consecutive checkpoints are flat (Δ within noise).
- Full FT: one PR run, one perf run, and one PR-vs-PR control.
- LoRA: at least 3 PR and 3 perf runs (different seeds), compared as mean ± range per checkpoint.
  - If the perf range overlaps the PR range, that settles it.
  - If it doesn't overlap after 3+3 runs, add 2+2 runs, then report.
  - Watch for grad spikes (the PR recipe uses lr 8e-4 with no warmup).
    Log grad norm every step, and note any spike-and-recover events in both arms.
- Also run one long-run check at geometry B for the final stacked build (full and LoRA, PR vs perf).
- Run long-run checks on the final stacked build, and on any single change that touches numerics
  (attention backend, bf16 weight storage/caching, checkpointing policy).
  Don't run them for pure launch or scheduling changes that already pass G1 exactly.

OWNERSHIP AND GIT
- You are the only one who edits code, commits, or writes REPORT.md.
  If you use sub-agents, they only run jobs in their lane and write JSON/logs to their results dir.
- Code goes on laya-decision-perf (based on PR head c5cb7cba59).
  Keep the bench commit on top in laya-decision-perf-bench.
- Commit as Etherll <61019402+Etherll@users.noreply.github.com>.
  Never override user.email or edit git config.
- Push ONLY those two branches to Etherll/unsloth.
  No PRs, nothing to unslothai/unsloth, never touch the PR author's branch or any studio/ files.
- ASK ME before:
  - any new public argument or dependency;
  - committing the xformers path;
  - enabling LoRA compile;
  - changing the compile threshold;
  - adding a VRAM-costing cache;
  - fixing the unsloth_zoo bug upstream.

REPORT (benchmarks/laya_perf/REPORT_SKELETON.md → REPORT.md)
Include:
- both environments (Windows/WSL and AWS Linux);
- the Phase-1 profile breakdown (Windows + WSL, which exist already, plus Linux);
- the ranked bottleneck table and how it changed after each applied fix;
- one row per candidate C1–C12 plus extras (lean LoRA, rope_once, xformers variants, new discoveries),
  each with s/step, tok/s, peak alloc/reserved, G1, verdict (APPLIED / REJECTED / REPORT-ONLY /
  SKIPPED <3%) and the rule cited;
- the final stacked table (full and LoRA × A and B, speed % and VRAM %);
- compile cost and break-even;
- long-run results: a per-checkpoint table (60/120/200, plus extensions) of CE, accuracy,
  s/step and VRAM for PR vs perf, the PR-vs-PR control spread, LoRA mean ± range,
  grad-spike notes, and why any run was extended;
- the commit list;
- Studio-pass open items: C10 GC choice, adamw_8bit, LoRA lr 8e-4 / no-warmup instability,
  the zoo bug, report-only dependency options.
Write concisely; percentages for speed/VRAM.

DONE
The 5 goal conditions in HANDOFF §10, with E:\ paths mapped to the Linux worktrees and
condition 4's 2-epoch run replaced by LONG-RUN CHECKS.
In your final message, show:
- the pytest output (`python -m pytest tests/test_decision_model.py -q`);
- the final stacked table;
- a 5-line summary: headline speed/VRAM %, what changed in code, what needs my decision.
