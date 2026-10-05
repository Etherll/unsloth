# Laya decision-model training: performance pass (Windows RTX 5090 + WSL)

## Context
PR #12585 (NilayYadav, branch `fine-tune-laya-decision-models-v2`, here at `c5cb7cba59`) adds
`FastDecisionModel` + `DecisionTrainer` in `unsloth/models/decision.py`. On Linux/B200 the earlier session
showed it trains correctly (fp32 grad cosine 1.000000, CE-only matches official) and found one perf
cliff: bf16 cuDNN SDPA rebuilding plans for every sequence length in laya's `nn.TransformerEncoder` head.
**Goal now:** make laya training as fast and lean as possible with Unsloth-style optimizations, keeping
loss/grad within 0.03 of the PR. **Scope: Unsloth core only** (`decision.py` + new private helpers + tests).
Studio files are untouched (Studio is a later pass). Work on a new local branch; never push.

## Machine facts (probed 2026-10-04)
- RTX 5090 32 GB, sm_120, WDDM, drives the display. **Lively wallpaper + other GPU apps are running**:
  each benchmark batch first checks `nvidia-smi` util < 5%; if not, ask the user to pause them.
- Windows system py3.12: torch 2.10.0+cu128, cuDNN 9.10.02, transformers 5.5.0, triton-windows 3.4.0,
  xformers 0.0.35, no flash_attn. `import unsloth` resolves to the PR worktree (editable). Don't install
  into system Python; if extra packages are needed, make `E:\laya-perf\venv-win` with uv.
- WSL Ubuntu: `~/pr-4460-20260922/environments/python` has torch 2.11.0+cu130 + **flash_attn 2.8.1
  (works on sm_120)** + transformers 5.5.0 + triton 3.6.0. Reuse its FA2 build (same torch ABI), don't rebuild.
- Cached assets: `E:\pr-4460-20260922\cache\huggingface\hub\` → `models--convaiinnovations--laya`
  (rev 55cf4c4e; root = ModernBERT-large, 421M params) and `datasets--LocalLLaMA--typed-decisions`
  (rev c76749e; 1,200 rows → 6,000 decisions, 123–512 tokens, median 307). The whole dataset is small.
- Profilers: torch.profiler (primary), Nsight Compute `C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\ncu.bat`
  (kernel deep dives). No nsys on Windows; optional in WSL.

## What we already know (don't redo)
- 2026-09-25 on this PC (`E:\laya-unsloth-20260925\REPORT.md`, official objective, fp16, mb8×acc4, len 512):
  no-GC +31.9% speed / +4.24 GiB; fused AdamW +4.8% / −222 MiB; CUDA graphs no gain; WDDM OOMs with
  "17 GiB free". torch.compile (+14%) and encoder packing were rejected for gradient mismatch, **but both
  were compared with dropout and RL noise active**, so RNG streams differed. Retest with dropout off.
- Reusable from that work: `E:\laya-unsloth-20260925\prepare.py` (data prep), `analyze.py` (paired bootstrap CI),
  `packed_encoder.py` (packing reference, buggy), `env.ps1`/`env.sh`.
- Unsloth pieces to reuse:
  - `unsloth/models/_sentence_transformer_unpadding.py` `_encoder_forward`/`_sentence_attention`: pack valid tokens → varlen attention → scatter back.
  - `unsloth/utils/attention_dispatch.py` `select_attention_backend`, `run_attention`, `AttentionConfig/Context` (FLASH_VARLEN / XFORMERS / SDPA, has `sliding_window`).
  - `unsloth/utils/packing.py` `_XFormersBidirectionalMask`, `_get_cached_block_mask`.
  - cuDNN exclusion precedent `fix_cudnn_sdpa_d256_masked_backward` in `unsloth/import_fixes.py` (~6907–6961).
  - `FastSentenceTransformer` compiles ModernBERT natively (`sentence_transformer.py` ~2064–2200, `_estimate_compile_threshold`).
  - Triton kernels `unsloth/kernels/geglu.py` (exact GEGLU fwd/bwd), `rope_embedding.py`, `layernorm.py` (needs a bias).
- transformers 5.5 ModernBERT: `_supports_flash_attn`, dispatches via `ALL_ATTENTION_FUNCTIONS`, passes
  `sliding_window` (local layers, window 128 = ±64) and `**kwargs` to the attention fn; its FA2 path only
  unpads inside attention, so MLP/norms still run on pad tokens.

## Storage: everything on E: (user rule)
All work, outputs, venvs and caches live on E:. Leave E: only when there is no other way, and say so in REPORT.md.
C: has 25 GB free; the WSL Ubuntu disk is `E:\WSL\Ubuntu`, so WSL `~` counts as E:.
- `E:\laya-perf\env.ps1` (dot-source it before every Windows run) sets:
  - `TEMP`/`TMP` = `E:\laya-perf\tmp`
  - `HF_HOME` = `E:\pr-4460-20260922\cache\huggingface`, with `HF_HUB_OFFLINE=1`
  - `TORCHINDUCTOR_CACHE_DIR` = `E:\laya-perf\cache\inductor`
  - `TRITON_CACHE_DIR` = `E:\laya-perf\cache\triton`
  - `CUDA_CACHE_PATH` = `E:\laya-perf\cache\nv`
  - `UV_CACHE_DIR` = `E:\laya-perf\cache\uv`
  - `PIP_CACHE_DIR` = `E:\laya-perf\cache\pip`
  - `XDG_CACHE_HOME` = `E:\laya-perf\cache`
  - `PYTHONPYCACHEPREFIX` = `E:\laya-perf\cache\pyc`
- `env.sh` for WSL sets the same variables, pointing into `~/laya-perf/cache`.
- Scratch files go in `E:\laya-perf\tmp`, not the session scratchpad on C:.
- Write profiler traces, memory snapshots and ncu reports only under `E:\laya-perf\out`.

## Setup (Phase 0, ≤30 min)
1. `git worktree add -b laya-decision-perf E:\orca\workspaces\unsloth\laya-decision-perf c5cb7cba59`.
   The current worktree stays the untouched **baseline tree**. Choose the tree per run with
   `PYTHONPATH=<tree>`, which takes precedence over the editable install. Commit with the configured identity; never edit git config.
2. Bench root `E:\laya-perf\` (`scripts\`, `data\`, `out\`, `tmp\`, `cache\`, `env.ps1`, `env.sh`, `REPORT.md`). See "Storage" above.
3. `prep.py`: `FastDecisionModel.build_dataset` + `split_holdout(seed=3407)` once → `data\items.pt`
   (train ≈5,600 / holdout 400). Fixed bench subset = `random.Random(0).sample(train, 1280)`.
4. Official baseline: fresh clone of NandhaKishorM/laya at `2e4d9c8` into `E:\laya-perf\laya-official`
   (leave the old 970dc8c clone alone). Run `notebooks/laya_finetune_typed_decisions_mps.py` as shipped (fp32),
   only to get **this PC's official peak VRAM and speed** (the VRAM ceiling in the rules below), at geometry A,
   and at B too if its batch constants can be set without editing its logic. Cap it at 40 optimizer steps.
5. WSL venv `~/laya-perf/venv` (ext4): py3.12, torch 2.11.0+cu130. Copy `flash_attn*` (package +
   `flash_attn_2_cuda*.so`) from the pr-4460 env's site-packages. Pin transformers 5.5.0 and the same
   unsloth_zoo as Windows. Read code from `/mnt/e/...` via PYTHONPATH.

## Harness (E:\laya-perf\scripts)
- `bench.py --tree --mode full|lora --mb --acc --steps --warmup 4 --gc unsloth|true|off --opts a,b,c --platform`
  Real `DecisionTrainer` with bf16 (what Studio runs on this GPU), adamw_torch, cosine, lr 2.5e-5 for full / 8e-4 for LoRA, `logging_steps=1`.
  A callback times each step with `cuda.synchronize`. Output JSON: median s/step after warmup,
  non-pad tok/s, peak alloc/reserved (reset after load), per-step loss and grad_norm, nvidia-smi util at start.
  `--opts` turns candidates on via monkeypatches first; they are ported into `decision.py` only after they win.
- Geometries: **A** mb2×acc16 (official/handoff, launch-bound) 40 steps; **B** mb8×acc8 (Studio default) 20 steps.
  Both use 1,280 decisions, so a run is about 1 min including load.
- `run_ab.py`: fresh process per run, ABAB order × 3 pairs, then `analyze.py`-style paired median and CI.
- `parity.py` (**G1**, deterministic): dropout off (every `nn.Dropout.p=0`, head `self_attn.dropout=0`).
  8 fixed micro-batches, run in bf16 and fp32. Per batch: |Δloss| ≤ **0.03** abs, |Δgrad_norm|/grad_norm ≤ **0.03**.
  Also reports per-parameter cosine and global relL2, which should be ≈1e-5 in fp32 for exact rewrites like unpadding.
- `curve.py` (**G2**): 60 steps, dropout on, same seed/order. Mean |Δloss| over 10-step windows ≤ 0.03,
  and holdout CE |Δ| ≤ 0.03 via `FastDecisionModel.evaluate`.
- `prof.py`: torch.profiler (wait 2, warmup 2, active 3), with `record_function` around encoder, head, scorer+loss,
  backward, optimizer.step, the GC recompute/offload, and data/collate.
  Outputs: Perfetto trace, top-30 by self CUDA and by self CPU, GPU-busy % (kernel time / wall),
  and a `torch.cuda.memory._record_memory_history` snapshot at peak (view at pytorch.org/memory_viz).
  Run on Windows and WSL to measure the WDDM launch overhead. Use ncu only on the top 3 kernels if they look odd.

## Decision rules (apply per candidate, then re-check the stack)
Speed = paired median over 3 ABAB pairs, all 3 pairs the same sign. VRAM = peak reserved (alloc reported too).
- **APPLY** if any of these holds (and G1 passes; G2 too when numerics change beyond exact reordering):
  - speed ≥ +5% and VRAM ≤ +0.5%;
  - VRAM ≤ −2% and speed ≥ −1%;
  - speed ≥ +10%, VRAM higher, but VRAM still ≤ the official script's peak on this PC.
- Pathology fix (C1): apply if neutral (|Δ| < 1%) on the 5090, because it removes a measured ~100× cliff on B200.
- Report-only: anything that needs a new public argument, overrides a user setting (optimizer, batch, GC
  mode), or adds a required dependency. Optional backends must be auto-detected, with the current SDPA path as the fallback.
- After stacking, a winner whose stacked gain drops below half its solo gain is re-tested alone and dropped if it no longer qualifies.

## Phase 1: baseline and profile (≤1 h)
Run `bench.py` on the baseline tree (full and LoRA × A and B), the official script (A), and `prof.py` on Windows and WSL.
Write the breakdown table into REPORT.md: share of time in encoder fwd/bwd, head, GC recompute, offload copies,
optimizer, data, CPU gaps, plus memory at peak. **Use it to reorder the candidates below**:
skip any candidate whose target is under 3% of step time.

## Phase 2: candidates (screen each: bench A/B → G1 → G2 if needed)
| # | Candidate | Where | Notes |
|---|---|---|---|
| C1 | Exclude cuDNN SDPA: `sdpa_kernel([FLASH, EFFICIENT, MATH])` around the forward in `compute_loss` (decision.py:353) and `_logits` (:401) | core | handoff fix 1; check whether torch 2.10/sm_120 even picks cuDNN |
| C2 | Collator `pad_to_multiple_of` 8 / 64 (`DecisionDataCollator.__call__`, :247) | core | tensor-core alignment, cuDNN plan reuse, fewer compile shapes |
| C3 | **Encoder unpadding**: pack valid tokens before the encoder (`input_ids[mask]`, per-sequence `position_ids`) and register an attention fn through `AttentionInterface` that calls `run_attention` (FLASH_VARLEN on Linux, XFORMERS / flex / packed-SDPA on Windows) with ModernBERT's symmetric ±64 window on local layers; scatter `last_hidden_state` back to [B,L,d] for the head (pad rows are masked keys and never gathered, so the loss is exact) | core, new `_decision_unpadding.py` | must pass fp32 G1 at ≈1e-5. Variants: (a) stock `attn_implementation="flash_attention_2"` (WSL), (b) packed+FA2 varlen, (c) packed+xformers if it supports a bidirectional local window, (d) flex_attention block mask (doc ∧ \|i−j\|≤64), (e) packed dense SDPA mask. Gain = skipped pad tokens + sparse local attention (2/3 of layers) |
| C4 | Regional `torch.compile` of each ModernBERT layer (dynamic, with C2 buckets) | core | compare with and without GC (non-reentrant vs "unsloth"); report compile seconds and gate on a step threshold like `_estimate_compile_threshold`; Windows Inductor crashed before (0xC0000006): if it still does, don't ship on Windows |
| C5 | Unsloth Triton kernels on ModernBERT: exact GEGLU (MLP `act(input)*gate`), RoPE, bias-less LayerNorm | core | only if C4 is rejected and elementwise kernels are ≥15% of GPU time |
| C6 | Head varlen/SDPA path reusing the same `in_proj`/`out_proj` weights (attention dropout 0.1 kept) | core | only if the head is ≥10% of step time |
| C7 | LoRA: store the frozen encoder in the autocast dtype (bf16 on bf16 GPUs; today fp16 → a cast every matmul, decision.py:591) | core | LoRA arms only |
| C8 | `_logits` / evaluate / calibrate: length-sorted batches, larger batch, `inference_mode`, results back in input order | core | gate: same logits (bf16 ≤1e-2, fp32 ≤1e-5) and same metrics; time evaluate+calibrate on 400 items |
| C9 | Data path: pre-tensorize items / pinned memory | core | only if data+collate is ≥5% of step |
| C10 | GC variants: off / `True` / "unsloth" / partial (turn off `gradient_checkpointing` on some ModernBERT layers) | report | changes a user setting; report speed vs VRAM against the official ceiling for the Studio pass |
| C11 | Optimizers: `adamw_torch_fused`, `adamw_8bit` (if bitsandbytes works), TF32 for fp32 ops | report | for the Studio pass |
| C12 | CUDA graphs / `reduce-overhead` | core | only if GPU-busy is still < 70% at geometry A after C3/C4 |

## Phase 3: port winners into core
- Port each winner into `unsloth/models/decision.py` (or a private `unsloth/models/_decision_*.py`), in repo style
  (`kw = value` spacing, short comments, AGPL SPDX header on new files). No new public arguments.
  Auto-detect backends with an SDPA fallback, so Windows without xformers/FA2 and CPU keep today's behavior.
- Tests in `tests/test_decision_model.py`, in its existing fixture style (`checkpoint`, `lora` parametrize):
  - packed vs padded encoder: hidden states and gradients equal (tiny random ModernBERT config, CUDA-gated, skipped without a varlen backend);
  - the cuDNN guard is engaged during `compute_loss`/`_logits` (spy on `sdpa_kernel`);
  - `_logits` keeps input order with length sorting;
  - collator padding keeps markers and mask;
  - LoRA encoder dtype follows `_amp_dtype`.
- One commit per change on `laya-decision-perf`, each message stating the measured gain.

## Phase 4: verification
1. `python -m pytest tests/test_decision_model.py -q` in the perf worktree on Windows; also in WSL if FA2 code was added.
2. Stacked build vs baseline: `run_ab.py` for full and LoRA × A and B on Windows (and WSL if a backend is Linux-only), plus G1 and G2.
3. One full 2-epoch run each (5,600 items, geometry A, bf16, same seed/order), baseline vs final:
   holdout CE/accuracy |Δ| ≤ 0.03, curves PNG, total seconds, peak VRAM. About 7–10 min each.
4. `prof.py` on the final build: a before/after breakdown in REPORT.md.

## Phase 5: report
`E:\laya-perf\REPORT.md`:
- environment;
- the baseline and official numbers;
- the profile breakdown (Windows vs WSL);
- one row per candidate: s/step, tok/s, Δspeed, peak alloc/reserved, ΔVRAM, G1/G2 numbers, and a verdict (APPLIED / REJECTED / REPORT-ONLY) with the rule that decided it;
- the final stacked numbers;
- the commit list;
- open items for the Studio pass (C10/C11, defaults).

Tell the user the headline and that nothing was pushed.

## Goal command for the executing agent
Turn on auto mode first so tool calls inside a turn don't stall. Then:

```
/goal Execute the plan at C:\Users\pc\.claude\plans\i-want-to-continue-elegant-walrus.md (laya decision-model training performance, Unsloth core only). Done when ALL are true and shown in your own output: (1) E:\laya-perf\REPORT.md contains the Phase-1 profile breakdown (Windows and WSL) and one row for each candidate C1-C12 with measured s/step, tok/s, peak alloc/reserved, G1 parity (per-batch |dloss| <= 0.03 and grad-norm relative diff <= 0.03, dropout off) and a verdict APPLIED / REJECTED / REPORT-ONLY / SKIPPED(<3% of step) citing the plan's decision rule; (2) every APPLIED change is committed on local branch laya-decision-perf in worktree E:\orca\workspaces\unsloth\laya-decision-perf, touching only unsloth/models/decision.py, new unsloth/models/_decision_*.py files and tests/test_decision_model.py; (3) `python -m pytest tests/test_decision_model.py -q` passes in that worktree on Windows (and in WSL if Linux-only code paths were added), with the output shown; (4) the final stacked build vs the PR baseline is reported for full and LoRA at mb2xacc16 and mb8xacc8 (speed and VRAM), passes G2 (10-step-window |dloss| <= 0.03), and a full 2-epoch run shows holdout CE and accuracy within 0.03 of the baseline; (5) all outputs, venvs and caches are on E: (env.ps1/env.sh redirect TEMP, inductor, triton, CUDA, uv and pip caches), or any exception is justified in REPORT.md; nothing was pushed, the baseline worktree E:\orca\workspaces\unsloth\studio-fine-tune-laya-decision-models-and-serve and all studio/ files are unchanged, and git config was not edited. Ask the user instead of proceeding if a change needs a new public argument, a new required dependency, a Studio edit, or a push, or if the GPU is busy with other apps. Stop after 200 turns.
```
