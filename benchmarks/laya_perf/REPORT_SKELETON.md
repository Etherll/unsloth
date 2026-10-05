# Laya decision-model training performance (PR #12585, Unsloth core)

Date: 2026-10-04/05. Machine: Windows 11, RTX 5090 32 GB (sm_120, WDDM, also drives the desktop), Ryzen 9 9950X3D.
Branch `laya-decision-perf` in `E:\orca\workspaces\unsloth\laya-decision-perf`, five commits on top of the PR head
`c5cb7cba59`. Nothing was pushed. Studio files and the PR worktree are unchanged.

## Headline

<!--HEADLINE-->

## Environment

| | Windows (primary) | WSL Ubuntu |
|---|---|---|
| Python | 3.12.10 (system) | 3.12.3 (`~/pr-4460-20260922/environments/python`, used read-only) |
| torch | 2.10.0+cu128, cuDNN 9.10.02 | 2.11.0+cu130 |
| transformers | 5.5.0 | 5.5.0 |
| triton | triton-windows 3.4.0 | 3.6.0 |
| flash-attn | not installed | 2.8.1 |
| xformers | 0.0.35 dev build at `E:\xformers`: `attn_bias` import fails (missing `mslk`), unusable | not installed |
| unsloth_zoo | 2026.5.1 | 2026.9.6 |

- Model: `convaiinnovations/laya` root checkpoint (rev 55cf4c4e): ModernBERT-large encoder (28 layers, d=1024, local window ±64 on 2 of 3 layers) and laya's 2-layer `nn.TransformerEncoder` head, 421M parameters. Copied to `E:\laya-perf\model` because the cached snapshot has no refs.
- Data: `LocalLLaMA/typed-decisions` (rev c76749e), 1,200 rows → 6,000 decisions through the PR's own `build_dataset`. `split_holdout(seed=3407)` gives 5,600 train and 400 holdout. Lengths are 123–634 tokens, median 307. Benchmarks use a fixed 1,280-decision subset, `random.Random(0).sample(train, 1280)`.
- Geometries: **A** = mb2 × acc16, 24 steps (the official script's default); **B** = mb8 × acc8, 16 steps (Studio's default). bf16 autocast, `adamw_torch`, cosine schedule. lr is 2.5e-5 for full fine-tuning and 8e-4 for LoRA (r=64, alpha=64, all-linear). Gradient checkpointing is "unsloth", the Studio default.
- **Storage:** everything is on E:.
  - `env.ps1`/`env.sh` (Windows) and `env-wsl.sh` set TEMP, the HF, Triton, CUDA, uv, pip and pyc caches, and TORCH_HOME.
  - Unsloth itself pins the inductor cache to `%TEMP%\torchinductor_<user>`, which resolves to `E:\laya-perf\tmp\torchinductor_pc`.
  - The WSL disk is `E:\WSL\Ubuntu`.
  - No exceptions were needed.

## Method

- `scripts/bench.py` runs the real `DecisionTrainer`, one config per fresh process, pinned to the V-cache CCD at high priority.
  - Step time is measured with a `cuda.synchronize` callback.
  - It records peak allocated/reserved memory, split into forward/backward and optimizer peaks.
  - It records per-step loss and grad norm.
- Candidates were toggled as monkeypatches (`scripts/opts.py`) before porting.
- `scripts/run_ab.py`: ABAB fresh-process pairs, speed = paired median of step-time ratios.
  - The desktop (Edge, mpv, Hayase) causes bursts where steps run 30–40% slower in either arm.
  - So `scripts/reab.py` also reports a burst-robust speed from the 25th-percentile step time. Both are listed when they disagree.
- **G1** (`scripts/parity.py`, `pcompare.py`): dropout off, 8 fixed micro-batches of 4, the same weights, `DecisionTrainer.compute_loss`. It compares per-batch loss, grad norm, cosine and relative L2 of the batch-0 gradient.
- **G2** (`scripts/bench.py --items train --steps 60 --eval`, `g2compare.py`): dropout on, same seed and order, mean |Δloss| per 10-step window, and holdout CE / accuracy from `FastDecisionModel.evaluate`.

### What the 0.03 gate can and cannot mean in bf16

The PR's bf16 training is not deterministic. Efficient-attention backward and embedding-gradient atomics differ between two identical runs, so two PR runs drift apart by up to 0.035 in a 10-step window.

More importantly, the PR's own bf16 gradients are far from its fp32 gradients:

- the PR bf16 grad norm differs from PR fp32 by up to **69%** on batch 1 (62.5 vs 36.9);
- the bf16 batch-0 gradient has relative L2 **0.22** vs fp32.

A "grad norm within 0.03 of the PR in bf16" gate is therefore below bf16's own noise floor for any change that touches rounding. G1 is applied as:

1. **fp32:** every batch |Δloss| ≤ 0.03 and grad-norm relative difference ≤ 0.03. Exact rewrites must reach ≈1e-5.
2. **bf16:** |Δloss| ≤ 0.03 against the PR, and the gradient is no further from the fp32 reference than the PR's own bf16 gradient is. Exact rewrites must also meet grad-norm ≤ 0.03 against the PR directly.

<!--PHASE1-->

<!--CANDIDATES-->

<!--FINAL-->

<!--COMMITS-->

<!--STUDIO-->
