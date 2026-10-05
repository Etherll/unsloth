# Laya decision-model training performance (PR #12585, Unsloth core)

Branch `laya-decision-perf` (Etherll/unsloth fork) on top of the PR head `c5cb7cba59`; tooling on
`laya-decision-perf-bench`. Scope: `unsloth/models/decision.py` and `tests/test_decision_model.py` only. Studio untouched.
Dates: 2026-10-04/05. Speed = paired median of p25 step time over interleaved fresh-process pairs; VRAM = peak reserved.

## Headline

Final build vs the PR, measured in one Colab VM (RTX PRO 6000 Blackwell, torch 2.11), 3 pairs each:

| | mb2 x acc16 (A) | mb8 x acc8 (B) |
|---|---|---|
| Full fine-tune | **+54.7% speed, -14.0% VRAM** | **+25.9%, -9.4%** |
| LoRA r=64 | **+7.9%, 0.0%** | **+6.8%, -0.6%** |

All four pass G2; long runs match the PR within its own seed-to-seed spread. The LoRA collapse seen on the B200 is the
PR recipe's own instability (lr 8e-4, no warmup): it collapses equally often in the PR (3/26 seeds) and the perf build
(3/26), and a 10-step warmup removes it in both (needs your decision, below).

## Environments

| | Windows (Phase 0-4) | WSL | AWS B200 (partial) | Colab G4 | Colab L4 | Kaggle T4x2 |
|---|---|---|---|---|---|---|
| GPU | RTX 5090 32 GB, sm_120 | same | B200 180 GB, sm_100 | RTX PRO 6000 Blackwell Server 96 GB, sm_120 | L4 22 GB, sm_89 | Tesla T4 15 GB, sm_75 (one used) |
| CPU | Ryzen 9 9950X3D | same | Xeon 8559C, 24 pinned threads per lane | AMD EPYC 9B45, 48 vCPU | Xeon @ 2.20 GHz, 12 vCPU | 4 vCPU |
| Python / torch | 3.12 / 2.10.0+cu128 | 3.12 / 2.11.0+cu130 | 3.12 / 2.13.0+cu130 | 3.13 / 2.11.0+cu130 | 3.13 / 2.11.0+cu130 | 3.13 / 2.11.0+cu128 |
| triton / cuDNN | win 3.4 / 9.10 | 3.6 | 3.7.1 / 9.20 | 3.6.0 / 9.27 | 3.6.0 / 9.27 | 3.6.0 / 9.19 |
| transformers / peft / zoo | 5.5.0 / - / 2026.5.1 | 5.5.0 / - / 2026.9.6 | 5.5.0 / 0.21.2 / 2026.9.9 | same pins | same pins | same pins |
| half precision | bf16 | bf16 | bf16 | bf16 | bf16 | fp16 (T4 has no bf16) |
| SDPA backend picked (encoder) | cuDNN (PR) / mem-eff (C1) | same | cuDNN (PR) / mem-eff (C1) | mem-eff | mem-eff | mem-eff |
| Status | complete | complete | **partial**: shared box; GPUs 5-7 held by other users for hours and CPU load 200-600; stopped by the user | complete | complete | see T4 section |

- Model `convaiinnovations/laya` rev 55cf4c4e (ModernBERT-large + 2-layer head, 421M); data `LocalLLaMA/typed-decisions` rev c76749e:
  6,000 decisions, 5,600 train / 400 holdout, 123-634 tokens (median 307). Identical on every machine.
- unsloth_zoo `misc.py` `__name__` bug (handoff section 5): **fixed** in 2026.9.9 (`used = [x for x in items if not x.startswith("__") ...]`).
- ncu was not usable on AWS (`RmProfilingAdminOnly=1`, no sudo); kernel classification uses shapes and kernel times instead.
- Cloud runs go through `scripts/cloud_submit.py` (local) and `scripts/cloud_jobs.py` (VM): one fresh VM per batch, PR (`c5cb7cba59`)
  and perf trees built from the same clone in the same VM, so every PR/perf comparison shares one machine.

## Gates and how they were read

- **G1** (`parity.py`, dropout off, 8 fixed batches): fp32 per-batch |dloss| <= 0.03 and grad-norm rel. diff <= 0.03.
  bf16: |dloss| <= 0.03 against the PR, and grads no further from fp32 than the PR's own bf16 grads. bf16 grad norms
  spike on single batches for every path (PR worst batch 0.17 from fp32, PR+C1 0.69), so the bf16 grad check uses the
  **median** per-batch distance (PR 0.054, current perf 0.057, final 0.025 on B200); worst batch is reported too.
- **G2** (60 steps, dropout on): max 10-step-window |dloss| <= 0.03, with a PR-vs-PR control every time. Where the
  control itself fails the gate (identical code and seed: L4 LoRA A, T4 fp16), G2 is read over >= 3 seeds per arm on the
  seed-averaged loss curves, with the same 0.03 limit (`scripts/g2_seeds.py`).
- **Long runs**: 200 steps at A (official default), holdout CE / accuracy at 60/120/200, PR-vs-PR control,
  LoRA as mean [range] over >= 3 seeds per arm.

## Phase-1 profile breakdown (PR baseline)

torch.profiler, 3 active steps (`scripts/prof.py`). "GPU busy" = kernel time / profiled wall.

| | GPU busy A | GPU busy B | Largest GPU categories (B) | Host |
|---|---|---|---|---|
| Windows 5090 | 19% | 68% | GEMM 38%, elementwise 17%, attention 6%, GC memcpy 4% | backward 66% of wall at A; ~28k launches/step |
| WSL 5090 | 27% | 87% | same split | lower launch cost than WDDM |
| AWS B200 (perf build) | 19.8% (nsys) | n/a | A: attention (sm80 mem-eff kernels) 19%, fp32 grad-accumulation adds 10%, casts | 18,100 kernels / 60,700 host ops per step; ~60% of host time in Inductor's Python wrapper |
| Colab G4 | 25.4% | 57.1% | GEMM 22%, elementwise 21%, attention 6%, GC memcpy 4% | backward 65% / rest 35% of host wall |
| Colab L4 | 25.9% | 72.8% | elementwise 30%, GEMM 26%, attention 6%, GC memcpy 5%, optimizer 5% | A host-bound on a 2.2 GHz Xeon |
| Kaggle T4 (fp16) | 54.6% | n/m | A: elementwise 29%, GEMM 16%, attention 4.5%, optimizer 3.4% | GPU-bound already at A (slow GPU) |

Head <= 1% and data <= 0.5% of the step everywhere, so C6 and C9 stay skipped.

## Ranked bottleneck table (final build, Colab G4)

Real step: A 0.476 s, B 0.421 s. Kernel time from `results/cloud/g4_prof_cand_full_{A,B}.summary.txt`.

| # | Item | % of step | Evidence | Hypothesis / action | Expected gain | Exactness risk | Outcome |
|---|---|---|---|---|---|---|---|
| 1 | Host / launch overhead at A | ~37% (A) | kernel sum 302 ms of 476 ms; B200 nsys GPU busy 19.8% | fewer, cheaper launches: static shapes (D1), C++ wrapper, CUDA graphs, one encoder graph | 10-50% at A | none (same kernels) | D1 **APPLIED** (+11%); C12 per-layer and whole-encoder CUDA graphs **REJECTED**; whole-encoder graph REPORT-ONLY; cpp_wrapper REPORT-ONLY |
| 2 | GEMM | ~50% of kernels at B | cuBLAS nvjet sm100/sm120 kernels | already tensor-core bf16, aligned shapes | < 3% | - | no action |
| 3 | Elementwise (casts, fp32 grad accumulation) | ~20% of kernels at B, ~10% at A | `vectorized_elementwise add<float>` 23 ms/step, `to_copy` kernels | bf16 weight cache (VRAM-costing) | <= 5% | rounding of cached copy | not built (needs your OK; below 5% expected) |
| 4 | Attention | ~13% of kernels at B | `fmha_cutlass*_sm80` (mem-eff) | varlen FA2 / xformers / FA4 | microbench: varlen -11% at mb2, +45% at mb8; e2e -14% A, +1% B | none | C3 variants **REJECTED** |
| 5 | GC offload memcpy | ~10% of kernels at B | DtoH / HtoD on side stream | already pinned, non_blocking, double-buffered | small | - | no action |

Stop rule: items 1 (CUDA graphs) and 4 (attention) were the last two tried and both rejected; the rest are < 3% of
step or need a VRAM-costing cache.

## Candidates

s/step and tok/s are the arm's p25 median; VRAM = peak alloc / reserved (MiB); speed and VRAM % are vs the arm it replaces.
Env: W = Windows 5090, L = WSL, B2 = AWS B200, G4 = Colab RTX PRO 6000. Rules (PLAN.md): **R1** speed >= +5% and VRAM <= +0.5%;
**R2** VRAM <= -2% and speed >= -1%; **R3** speed >= +10% and VRAM <= the official script's peak; **RP** pathology fix applied if
neutral; **RO** report-only (new public arg, overrides a user setting, or new dependency); **S3** skipped, target < 3% of step.

| # | Candidate | Env | s/step base -> cand | tok/s cand | alloc / reserved cand | Speed / VRAM | G1 (fp32; bf16) | Verdict |
|---|---|---|---|---|---|---|---|---|
| C1 | cuDNN SDPA guard, forward **and** backward (fixed on Linux: the GC recompute ran outside the guard and crashed `gradient_checkpointing=True` on the B200) | W full B | 0.754 -> 0.753 | 24,944 | 7,980 / 8,608 | +0.2% / 0.0% | exact; dloss 0, dgn 1e-4 | **APPLIED** (RP; G4 already picks mem-eff, so a no-op there) |
| C2 | pad to 8 / 64 | W full B | 0.750 -> 0.751 / 0.757 | 25,000 / 24,793 | 7,981 / 8,634; 7,993 / 8,596 | -0.1% / +0.3%; -0.9% / -0.1% | exact (padding is masked) | **REJECTED** (R1-R3 not met); pad64 reused inside D1 as the static-shape bucket |
| C3a | stock FA2 (padded) | L full B | 0.663 -> 0.660 | 28,470 | 7,984 / 8,626 | +0.8% (p25 +3.5%) / 0.0% | n/m (no-gain) | **REJECTED** (< 5%) |
| C3b | packed FA2 varlen (flash-attn) | L full A / B | 0.865 -> 0.880; 0.634 -> 0.626 | 10,495; 30,023 | 7,982 / 8,610 | -1.6%; +1.4% / -0.2% | n/m | **REJECTED** |
| C3c | packed, torch-native FA2 varlen (`torch.nn.attention.varlen`, +/-64 window) | G4 full A / B (eager) | 0.727 -> 0.847; 0.537 -> 0.531 | 10,886; 35,031 | 7,953 / 8,108; 7,966 / 8,134 | -14%; +1.1% / 0% | n/m | **REJECTED** |
| C3d | packed flex_attention | W full B | 0.748 -> 0.992 | 18,916 | 7,981 / 8,546 | -24.7% / -0.7% | n/m | **REJECTED** |
| C3e | packed dense SDPA mask | W full B | 0.755 -> 1.077 | 17,437 | 7,992 / 8,604 | -29.8% / 0% | n/m | **REJECTED** |
| C3x | xformers `memory_efficient_attention` padded / BlockDiagonal varlen | G4 microbench | - | - | - | **unavailable**: 0.0.35 is built for torch 2.10, ops fail to load on 2.11; its Blackwell FMHA is sm_100-only; other paths call the same torch kernels as SDPA. W: dev build missing `mslk` | - | **REPORT-ONLY** (unavailable; a source build cannot add an sm_120 kernel) |
| C3m | attention microbench, fwd+bwd per micro-batch (global+local) | G4 | SDPA mem-eff 0.251 / 0.515 ms (mb2 / mb8); cuDNN 0.245 / 0.522; varlen FA2 0.283 / 0.280 | - | - | - | - | evidence for C3c / C3x |
| C4 | regional compile of the ModernBERT layers (full FT) | G4 full A / B | no-compile 0.720 -> 0.536; 0.517 -> 0.429 | 17,224; 43,721 | 6,730 / 6,980; 7,096 / 7,368 | **+34.4%; +20.3%** / -7.3%; -5.9% | dloss 1.9e-6, dgn 1.0e-5; bf16 median 0.043 (PR 0.041) | **APPLIED** (R1, R2) |
| C5 | Triton GEGLU / RoPE / LN kernels | - | - | - | - | - | - | **SKIPPED** (plan: only if C4 rejected) |
| C6 | head varlen / SDPA | profile | head <= 1% of step | - | - | - | - | **SKIPPED** (S3) |
| C7 | LoRA frozen encoder stored in bf16 | W LoRA A / B | 1.674 -> 1.643; 0.924 -> 0.907 | 5,614; 20,698 | 1,971 / 2,164; 2,310 / 2,616 | +1.9%; +2.1% / -0.2%; +0.4% | fp32 n/a; bf16 dloss 0.013, dgn 0.069 (rounds the weights) | **REJECTED** (< 5%; would also round the merged checkpoint) |
| C8 | `_logits` length-sorted batches | W eval 400 items | 1.18 s -> 0.98 s per pass | - | - | +16.5% | identical logits and metrics | **APPLIED** (R1) |
| C9 | data path | profile | collate 0.2-0.7% | - | - | - | - | **SKIPPED** (S3) |
| C10 | GC off / `True` | W full B | 0.721 -> 0.499; 0.622 -> 0.625 | 37,598; 30,049 | 12,830 / 13,777; 7,604 / 8,398 | off +44.5% / +74.8% (above the 8,448 official peak); True -0.4% / +6.5% | exact | **REPORT-ONLY** (user setting) |
| C11a | fused AdamW for `adamw_torch` (all-fp32 params) | W full A | 1.134 -> 1.125 | 8,205 | 7,394 / 7,934 | +0.9% / **-8.3%** | fused vs foreach 1 ulp | **APPLIED** (R2) |
| C11b | `adamw_8bit` | W full B | 0.623 -> 0.625 | 30,037 | 4,719 / 5,358 | -0.3% / **-32.1%** | n/m | **REPORT-ONLY** (user setting) |
| C12a | CUDA graphs, per layer (`reduce-overhead`, trees, pad64, stable `.grad`) | B2 full A / B | 0.912 -> 10.39; 0.509 -> 5.03 | 771 | 6,825 / 6,874 | **-91%** (re-records every call: 28 layers share one graph with different parameter inputs) | n/m | **REJECTED** |
| C12b | CUDA graphs, whole encoder (`enc_cg`) | G4 full A | - | - | - | **SIGSEGV (rc -11) twice** during graph capture | n/m | **REJECTED** (crash) |
| X1 | lean LoRA forward | W LoRA A / B | 1.751 -> 1.553; 0.931 -> 0.864 | 5,942; 21,724 | 1,968 / 2,178; 2,304 / 2,596 | +7.9%; +7.6% / ~0 | bit-identical (fp32 and bf16 dloss 0) | **APPLIED** (R1) |
| X2 | rope_once (rotary computed once per type) | W full A | 1.110 -> 1.075 | 8,587 | 7,964 / 8,650 | +3.3% / 0% | exact (dgn 1e-7) | **REJECTED** (< 5%) |
| X3 | **D1 static shapes** (`dynamic=False`, device-side pad to 64, eval eager) | G4 full A (5 pairs) / B | 0.536 -> 0.487; 0.429 -> 0.423 | 18,943; 44,326 | 6,735 / 6,988; 7,132 / 7,368 | **+11.0%** (9.0..14.1); +1.4% / +0.1%; 0.0% | dloss 2.4e-6, dgn 2.3e-5; bf16 median 0.036; 6/6 reruns bitwise equal, 3 under GPU stress | **APPLIED** (R1) |
| X4 | D1 + Inductor `cpp_wrapper` | B2 full A screen | static 0.767 -> 0.682 | 13,536 | 6,781 / 7,028 | +12.5% over D1 | fp32 exact on torch 2.13; **crashes on torch 2.11** (`CudaKernelParamCache not populated`, G4 and pytest) | **REPORT-ONLY** (needs a torch-version gate: your decision) |
| X5 | whole encoder as one compiled graph, torch checkpointing inside (`enc_compile`) | G4 full A | final 0.476 -> 0.404 | 22,864 | 8,080 / 8,246 | +18% over final / **+18% VRAM** (under the 8,604 official peak) | dloss 1.6e-6, dgn 4.8e-5; bf16 dloss 0.010 | **REPORT-ONLY** (replaces Unsloth's offloaded GC: a user setting) |
| X6 | LoRA compile (D1 static path also for LoRA) | G4 LoRA A / B, 3 pairs each (g4_b8) | 1.025 -> 0.662; 0.607 -> 0.469 | 13,930; 40,057 | 1,905 / 2,032; 2,311 / 2,474 | **+54.6% (49.5..55.0); +29.5% (28.9..31.6)** / -2.4%; -0.6% | dloss 1.6e-6, dgn 7.5e-6; bf16 dloss 0.020; G2 A 0.016, B 0.020 PASS; 200-step 3 seeds: 0/3 collapse | **REPORT-ONLY** (plan: ask before enabling LoRA compile; meets R1) |
| X6 | LoRA compile, second GPU | L4 LoRA A / B, 3 pairs each (l6_l4lc) | 3.271 -> 2.256; 2.555 -> 2.045 | 4,090; 9,289 | 1,905 / 2,032; 2,311 / 2,416 | **+45.0% (41.7..48.0); +24.9% (24.0..25.5)** / -2.4%; +0.25% | G2 over 5 seeds/arm: A 0.021 PASS; B 0.050 as measured (PR seed 11 collapsed), 0.019 PASS without it | REPORT-ONLY (as above); cold first step 75 s on L4 |
| X6 | LoRA compile, third GPU (fp16) | T4 LoRA A / B, 2 pairs each (k5_t4) | 3.457 -> 2.689; 5.111 -> 4.101 | 3,446; 4,631 | 1,905 / 2,012; 2,291 / 2,374 | **+23.6 / +34.3%; +24.8 / +24.4%** / -2.4%; -1.4% | G2 3 seeds: A 0.028 PASS; B 0.092 (**LoRA-compile seed 3407 collapsed**), 0.025 without it | REPORT-ONLY (as above); cold first step 70-83 s on T4 |
| X7 | LoRA compile, dynamic shapes | G4 LoRA A / B screen | 1.028 -> 0.766; 0.598 -> 0.482 | 12,051; 39,425 | 1,896 / 2,006; 2,274 / 2,438 | +34%; +24% | n/m | superseded by X6 |
| X8 | official laya script (fp32, as shipped) | G4 A / B | 0.850 / 1.053 s/step | - | 7,964 / **8,604**; 7,969 / **8,814** | VRAM ceiling for R3 | - | reference |

n/m = not measured (no speed case to promote). C3 variants were never promoted, so they have no G1.

## Final stacked build vs the PR

| Env | Combo | PR s/step | final s/step | Speed | Reserved VRAM | G2 (window; PR-vs-PR control) |
|---|---|---|---|---|---|---|
| **G4** | full A | 0.736 | 0.476 | **+54.7%** (53.2..55.8) | 8,128 -> 6,988 (**-14.0%**) | 0.017 (0.010) PASS |
| G4 | full B | 0.530 | 0.421 | **+25.9%** (25.6..26.6) | 8,128 -> 7,368 (**-9.4%**) | 0.006 (0.004) PASS |
| G4 | LoRA A | 1.113 | 1.028 | **+7.9%** (7.3..8.4) | 2,082 -> 2,082 (0.0%) | 0.022 (0.174, control collapsed) PASS |
| G4 | LoRA B | 0.645 | 0.604 | **+6.8%** (6.3..7.0) | 2,506 -> 2,490 (-0.6%) | 0.007 (0.011) PASS |
| B2 (before D1) | full A / B | 1.925 / 1.338 | 0.846 / 0.499 | +126% / +168% (PR runs cuDNN attention and is host-bound on this slower CPU; C1 alone was not measured here) | -13.8% / -9.8% | 0.013 / 0.014 PASS |
| W (before D1, compile pre-fix) | full A / B; LoRA A / B | 1.111 / 0.752; 1.675 / 0.919 | 1.048 / 0.655; 1.699 / 0.862 | +5.6% / +14.6%; +7.8% / +5.9% (p25) | -14.0% / -6.6%; +0.3% / -1.3% | PASS (all four) |
| **L4** | full A | 2.484 | 1.865 | **+33.2%** (32.6..36.0) | 8,128 -> 6,988 (**-14.0%**) | 0.010 (0.006) PASS |
| L4 | full B | 2.660 | 2.154 | **+23.4%** (23.3..23.6) | 8,128 -> 7,268 (**-10.6%**) | 0.009 (0.013) PASS |
| L4 | LoRA A | 3.768 | 3.399 | **+11.3%** (4.0..11.9) | 2,082 -> 2,082 (0.0%) | 3-seed: **0.019** (0.011) PASS; single seed 3407 0.147 (control 0.039 fails itself) |
| L4 | LoRA B | 2.870 | 2.603 | **+10.3%** (9.1..10.4) | 2,426 -> 2,410 (-0.7%) | 0.0298 (0.017) PASS, marginal |
| T4 (fp16) | full A / full B / LoRA A / LoRA B | see Kaggle T4 section | | +23.1% / +27.2% / +9.6% / +10.3% | -14.0% / -10.8% / -0.9% / -0.7% | multi-seed 0.018 / 0.020 / 0.016 / 0.011, all PASS |

G4 G2 holdout: full A dCE 0.002 / dacc 0.008, full B 0.0003 / 0.010, LoRA A 0.005 / 0.003, LoRA B 0.009 / 0.030 (control 0.007 / 0.040).
L4 G2 holdout: full A 0.007 / 0.005, full B 0.009 / 0.015, LoRA B 0.003 / 0.040 (control 0.001 / 0.003). L4 at A is still host-bound
(GPU busy 26% in the PR, 27% in the final, on a 2.2 GHz Xeon), and the gain is smaller than on G4 (+33% vs +55%); the VRAM saving is identical.
L4 LoRA A, seeds 11-13 (l5_g2s): PR-vs-PR per-seed windows 0.015 / 0.040 / 0.039 (identical code), PR-vs-final 0.008 / 0.056 / 0.072;
seed-averaged **control 0.011, final 0.019, PASS**. Holdout PR 0.997 [0.988..1.009] / 0.650 [0.615..0.675], PR repeat 0.995 / 0.627,
final 1.004 [0.984..1.035] / 0.617 [0.555..0.647]. No collapse (loss 40-60 <= 1.05, grad norm >= 0.6).

## Compile cost and break-even (G4)

Excess seconds over steady state for a whole 24/16-step run, eval at steps 12 and 20.

| | cold A | warm A | cold B | warm B | first eval after training |
|---|---|---|---|---|---|
| D1 static (final) | 37.5 s | 10.5 s | 33.4 s | 9.8 s | 0.71 s (eager, no recompile) |
| previous dynamic compile | 46.0 s | 16.9 s | 22.0 s | 9.7 s | 5.9 s cold / 2.3 s warm (eval-mode recompile) |

Break-even vs not compiling (no-compile 0.720 / 0.517 s/step vs final 0.476 / 0.421): saved 7.6 ms per decision at A,
1.5 ms at B. **Cold: about 4,900 decisions at A and 22,000 at B; warm cache: about 1,400 and 6,500.** Both cold numbers are
above the plan's ~3k, so they go to you (below); no threshold was added.

## Long-run checks (200 steps, holdout at 60 / 120 / 200)

**Full fine-tune, A, G4**, 4 seeds per arm (3407-3410), mean [range]:

| step | PR CE | final CE | PR acc | final acc |
|---|---|---|---|---|
| 60 | 1.003 [0.976..1.074] | 0.993 [0.970..1.035] | 0.639 [0.595..0.683] | 0.657 [0.593..0.725] |
| 120 | 0.964 [0.916..1.052] | 0.932 [0.904..0.981] | 0.701 [0.610..0.753] | 0.734 [0.658..0.780] |
| 200 | 0.946 [0.899..1.044] | 0.913 [0.894..0.952] | 0.722 [0.620..0.773] | 0.747 [0.693..0.773] |

Paired per-seed CE at 200: -0.039, -0.138, 0.000, +0.044 (mixed sign; PR seed 3408 is the outlier). Same-seed PR-vs-PR
control (3407): 0.933 / 0.725 vs 0.926 / 0.750. **Extended** (plan rule b, the difference moved between 60 and 120) to 400
steps, seed 3407: dCE 0.011 / 0.0002 / 0.0016 and dacc 0.003 / 0.013 / 0.003 at 200 / 300 / 400, flat, so stopped.
Wall time 147 s (PR) vs 120 s (final) for 200 steps; peak reserved 8,128 vs 6,988 MiB.

**Full fine-tune, B, G4** (seed 3407): PR 0.880 / 0.810, final 0.881 / 0.785 (dCE 0.001, dacc 0.025); 113 s vs 98 s; 8,150 vs 7,390 MiB.

**LoRA, A (the collapse case), G4**, 5 seeds per arm (3407-3411):

| step | PR CE | perf CE | PR acc | perf acc |
|---|---|---|---|---|
| 60 | 1.063 [0.989..1.187] | 1.044 [0.981..1.213] | 0.559 [0.425..0.668] | 0.573 [0.228..0.690] |
| 120 | 1.019 [0.954..1.203] | 1.003 [0.931..1.213] | 0.627 [0.430..0.713] | 0.610 [0.260..0.720] |
| 200 | 0.981 [0.912..1.162] | 0.977 [0.894..1.213] | 0.668 [0.478..0.753] | 0.642 [0.250..0.763] |

Ranges overlap completely (plan: that settles it); means differ by 0.005 CE / 0.026 accuracy. One seed per arm
degenerates (PR 3410 to 0.48, perf 3409 to 0.25). Grad norms logged every step: no spike above 50 in any run; the
collapse is a slide to a constant predictor (loss ~1.21, grad norm ~0.3) within ~20 steps, not a spike-and-recover.
**LoRA, B, G4**: PR 0.884 / 0.793, perf 0.874 / 0.805.

**LoRA compile (X6, not committed), A, G4**, 3 seeds per arm (3407-3409), same VM as its PR arm (g4_b8):

| step | PR CE | LoRA-compile CE | PR acc | LoRA-compile acc |
|---|---|---|---|---|
| 60 | 1.069 [0.975..1.213] | 1.004 [0.997..1.015] | 0.506 [0.240..0.670] | 0.641 [0.628..0.665] |
| 120 | 1.032 [0.935..1.213] | 0.954 [0.947..0.962] | 0.566 [0.313..0.698] | 0.680 [0.670..0.695] |
| 200 | 1.010 [0.907..1.210] | 0.915 [0.895..0.945] | 0.596 [0.288..0.753] | 0.748 [0.695..0.785] |

PR seed 3408 collapsed (loss 40-60 1.230, grad norm 0.31); without it PR is 0.910 / 0.750 at 200, the same as LoRA compile.
0/3 LoRA-compile runs collapsed. Step time 1.08 s (PR) vs 0.657 s.

B200 reference (partial): full A PR 0.909 / 0.733, PR control 0.954 / 0.705, perf 0.901 / 0.760; LoRA A 3+3 seeds:
PR 0.915 [0.902..0.936] / 0.743 [0.715..0.768], perf 1.089 [0.924..1.213] / 0.518 [0.298..0.733] (2 of 3 perf seeds collapsed);
LoRA B PR 0.884 / 0.805, perf 0.886 / 0.775; full B PR 0.880 / 0.803.

### LoRA collapse: root cause

60-step collapse test at A (collapsed = loss of steps 40-60 > 1.15 with grad norm < 0.4):

| | PR | perf |
|---|---|---|
| G4, seeds 11-34 | 2 / 24 | 3 / 24 |
| G4, seeds 3408-3409 | 1 / 2 (3408) | 0 / 2 |
| B200 long runs, seeds 3407-3409 | 0 / 3 | 2 / 3 |
| G4 200-step, seeds 3407-3409 (g4_b8, against LoRA compile) | 1 / 3 (3408) | LoRA compile 0 / 3 |
| L4 60-step G2 runs at A (l4_final, l5, l6, l7), seeds 3407, 11-14 | 0 / 13 | 0 / 4 final, 0 / 5 LoRA compile |
| **L4 at B (mb8 x acc8)**, seeds 11-14, 3407 (l6, l7) | **1 / 5 (seed 11: loss 40-60 1.216, grad norm 0.21)** | LoRA compile 0 / 5 |
| T4 fp16 60-step runs (k5), seeds 3407, 11, 12 at A and B | 0 / 6 | **LoRA compile 1 / 6 (B, seed 3407)** |
| **10-step LR warmup**, seeds 3408, 3409, 11, 18 (incl. the collapsed ones) | **0 / 4** | **0 / 4** |

Ablation on the collapsing seeds (G4): removing C1, fused AdamW or the lean LoRA forward from perf, or adding C1 / fused
AdamW to the PR, changes which seed stalls but never stops it (seed 3408 stalls in PR, perf-with-PEFT-forward and PR+C1
alike). On G4 the perf LoRA forward and backward are bit-identical to the PR in bf16 (G1 dloss 0). **Cause: the PR's
LoRA recipe (lr 8e-4, no warmup, effective batch 32) sits at the edge of stability; small numeric differences pick
which seeds fall off it. No perf change is responsible, so nothing is reverted; a short warmup fixes it in both arms
(needs your decision: it changes a training setting).**

## Commits on `laya-decision-perf` (c5cb7cba59..43f60472b)

| SHA | Change |
|---|---|
| 379790a1c | Keep cuDNN attention out of the decision forward (C1) |
| cde313354 | Batch similar lengths when scoring decisions (C8) |
| 02e915716 | Run AdamW as one fused kernel (C11a) |
| 60eacc32a | Lean forward for plain LoRA layers on the decision encoder (X1) |
| f25ee6d96 | Compile the ModernBERT layers for full fine-tunes (C4) |
| 35a480e9e | Keep cuDNN attention out of the decision backward too (C1 fix found on the B200) |
| ca7303431 | Compile the decision encoder layers for static shapes (D1) |
| 43f60472b | Test the lean LoRA forward in the GPU's own autocast dtype (T4 fix) |

## Kaggle T4 (fp16)

Tesla T4 (sm_75, 15 GB, one of the two used), 4 vCPU, torch 2.11.0+cu128, cuDNN 9.19. The T4 has no native bf16, so Unsloth's
`is_bfloat16_supported()` picks fp16 with a GradScaler, as Studio does (torch's own check reports emulated bf16; k1 trained in
that by mistake and is discarded). Every T4 run below logged `half=fp16`. SDPA picks mem-eff for every call (probe). The full
model fits (8.1 GB reserved at most).

pytest on the perf tip, Kaggle T4 (k2b): `33 passed in 28.28s`.

| Combo | PR s/step | final s/step | Speed (3 pairs, k2b) | Reserved VRAM | tok/s final |
|---|---|---|---|---|---|
| full A | 3.419 | 2.779 | **+23.1%** (20.8..23.4) | 8,128 -> 6,988 (**-14.0%**) | 3,320 |
| full B | 5.225 | 4.107 | **+27.2%** (27.2..27.5) | 8,128 -> 7,248 (**-10.8%**) | 4,570 |
| LoRA A | 3.642 | 3.323 | **+9.6%** (8.2..9.9) | 2,080 -> 2,062 (-0.9%) | 2,777 |
| LoRA B | 5.594 | 5.070 | **+10.3%** (10.1..10.6) | 2,426 -> 2,408 (-0.7%) | 3,703 |

A second VM (k3) gave the same direction: full A 3.561 -> 3.051, full B 5.593 -> 4.616, LoRA A 3.731 -> 3.394, LoRA B 5.696 -> 5.279 s/step.

Profile, full A (3 steps, k2b): PR GPU busy 54.6%, kernel sum 2,801 ms/step (elementwise 1,488, GEMM 838, attention 229,
optimizer 176); final 53.2%, 2,221 ms/step (elementwise 919, GEMM 877, attention 249, optimizer 87). The T4 is GPU-bound
already at A, so the gain comes from fewer elementwise kernels (compile fusion) and the fused optimizer, not from launch cost.

**G2 in fp16.** Single-seed G2 cannot separate the arms on the T4: two runs of the identical PR code and seed differ by
0.035 (full A) and 0.062 (LoRA B) in k3, above the 0.03 gate. fp16 adds a second effect: the GradScaler starts at 2^16 and
skips the first step whose gradients overflow. In k2b and k3 the PR overflows at step 0 and the compiled final at step 1, so
each arm drops one different batch and the curves split from step 1 (full A 0.045, full B 0.044-0.045 against the PR).

Multi-seed G2 on the T4 (k2b, k3, k4; `scripts/g2_seeds.py`, seed-averaged 10-step windows, gate 0.03):

| Combo | Runs per arm | Per-run window PR vs final | Seed-averaged window | Holdout CE PR / final | Holdout acc PR / final |
|---|---|---|---|---|---|
| full A | 4 (seeds 11, 12, 3407 x 2 VMs) | 0.037 / 0.025 / 0.045 / 0.036 | **0.018 PASS** | 1.001 [0.989..1.012] / 1.004 [0.985..1.028] | 0.651 [0.625..0.670] / 0.634 [0.608..0.655] |
| LoRA B | 5 (seeds 11-13, 3407 x 2 VMs) | 0.038 / 0.073 / 0.014 / 0.109 / 0.055 | **0.011 PASS** | 0.990 [0.963..1.021] / 0.990 [0.971..1.052] | 0.664 [0.608..0.715] / 0.662 [0.593..0.698] |
| full B | 4 (seeds 11, 12, 3407 x 2 VMs) | 0.010 / 0.007 / 0.045 / 0.044 | **0.020 PASS** | 0.991 [0.974..1.017] / 0.994 [0.975..1.018] | 0.655 [0.627..0.672] / 0.656 [0.640..0.682] |
| LoRA A | 4 (seeds 11, 12, 3407 x 2 VMs) | 0.013 / 0.046 / 0.017 / 0.049 | **0.016 PASS** | 0.995 [0.992..1.000] / 0.995 [0.987..1.015] | 0.634 [0.615..0.660] / 0.637 [0.613..0.650] |

Noise floor (same code, same seed 3407, two Kaggle VMs): PR full A 0.024, final full A 0.044, PR LoRA B 0.055, final LoRA B 0.112.
**Why the compiled build skips a different first step:** with compile off (`UNSLOTH_COMPILE_DISABLE=1`), the perf build overflows
at step 0 exactly like the PR (k4, seed 3407); compiled, it does not. The likely mechanism (not isolated further) is that Inductor computes fused fp16 elementwise
chains in fp32 and rounds once at the store, so fewer intermediates overflow. After step 0 the scaler's skips land at scattered steps in every arm (PR 26 / 53 / 58, no-compile
33 / 37, final 19), so this is ordinary fp16 loss-scaling, not a defect. No run collapsed (loss 40-60 <= 1.05).



## Cloud runs

All Linux GPU work after the move ran on Colab or Kaggle; every Colab VM was released by `notebook_cloud_run.py`
("releasing Colab session ... (notebook finished)"), Kaggle kernels completed or were deleted.

| Tag | Backend / GPU | Purpose | Result |
|---|---|---|---|
| g4_b1 | Colab G4 | pytest + LoRA ablation | ABORTED during setup (see discarded) |
| g4_b1r | Colab G4 | pytest perf/cand, SDPA probe, LoRA ablation (14 runs) | PASS (cand pytest failed: cpp_wrapper on torch 2.11; probe script bug) |
| g4_b2 | Colab G4 | G1 PR/perf/cand, cand race, collapse rate seeds 11-18 | PASS (cand-with-cpp_wrapper jobs failed on torch 2.11, as found) |
| g4_b3 | Colab G4 | timing A/B PR/perf/cand x 4 combos, official script, compile cost, profiles | PASS (cpp_wrapper arm failed; static arm measured) |
| g4_b4a | Colab G4 | static cand G1 + race, LoRA long runs 5+5 seeds, LoRA B, LoRA G2 | PASS |
| g4_b4b | Colab G4 | D1 extension pairs, compile vs no-compile, full G2 + long runs, profiles, screens | PASS (enc_cg segfault; xformers venv failed) |
| g4_b5 | Colab G4 | collapse rate seeds 19-34, warmup test | PASS |
| g4_final | Colab G4 | pytest on ca7303431, PR vs final 4 combos x 3 pairs | PASS (33 passed) |
| g4_b7 | Colab G4 | full long-run seeds, 400-step extension, G1 LoRA compile / enc_compile, enc_cg retry, attention microbench, C3 screens | PASS (enc_cg segfault again; xformers ops unavailable) |
| k1_t4 | Kaggle T4x2 | pytest, G1, timing | COMPLETE; timings discarded (see below); found the T4 test bug |
| k2_t4 | Kaggle T4x2 | final A/B + G2 | CANCELLED (same harness bug) |
| k2b_t4 | Kaggle T4x2 | pytest, PR vs final 4 combos x 3 pairs (fp16), G2, SDPA probe, profiles (40 jobs) | COMPLETE (33 passed; G2 fails in fp16, see T4 section) |
| g4_b8 | Colab G4 | LoRA compile: 3+3 pairs A/B, G2, 200-step 3+3 seeds (22 jobs) | PASS |
| l4_final | Colab L4 (2 VMs: first lost its kernel after 11 s, retried) | pytest, PR vs final 4 combos x 3 pairs, G2, SDPA probe, profiles (42 jobs) | PASS (33 passed; LoRA A G2 control fails, see l5) |
| l5_g2s | Colab L4 (2 VMs: first lost its connection, retried) | LoRA A G2 seeds 11-13 x {PR, PR repeat, final} (9 jobs) | PASS (seed-averaged 0.019, control 0.011) |
| k3_t4 | Kaggle T4x2 | G2 PR / PR repeat / final x 4 combos (fp16), SDPA probe, profiles (15 jobs) | COMPLETE (showed the fp16 noise floor above the gate) |
| k4_t4 | Kaggle T4x2 | fp16 G2 seeds 11-13, compile-off arm, perf with foreach AdamW (14 jobs) | PASS (seed-averaged full A 0.018, LoRA B 0.011) |
| l6_l4lc | Colab L4 | LoRA compile 3+3 pairs A/B, G2 seeds 3407 / 11 (20 jobs) | timing PASS (+45.0% / +24.9%); G2 A 0.030 PASS, B 0.092 (PR seed 11 collapsed) -> more seeds in l7 |
| k6_t4 | Kaggle T4x2 | fp16 G2 seeds 11-12 for full B and LoRA A (8 jobs) | PASS (seed-averaged 0.020 / 0.016) |
| l7_l4lcs | Colab L4 | LoRA compile G2 seeds 12-14 at A and B (12 jobs) | 5-seed G2: A PASS 0.021; B 0.050 from a PR collapse, 0.019 without it |
| k5_t4 | Kaggle T4x2 | LoRA compile 2+2 pairs A/B, G2 seeds 3407 / 11 / 12 (20 jobs) | timing PASS (+24-34%); G2 A PASS, B 0.092 (LoRA-compile seed 3407 collapsed) |
<!--CLOUDROWS-->

### Discarded measurements

- AWS: `stack1_full_B` pairs 3-4 (load 161 -> 448 from another user's ~127 cores); `scr_cand_full_A` (load 98 -> 221, foreign process);
  `scr_static_cpp_full_B`, `scr_cand_full_B` (load 98-361); `ic_idle` B0 first try (foreign llama-server appeared);
  `long_s200_perf_full_A` wall time (foreign processes on GPU7; holdout numbers kept); two jobs killed while the perf tree was
  briefly edited (`logs/discarded.txt`). All other AWS timing work was gated on a free GPU and load < 40 and never ran.
- g4_b1: stopped during setup and the VM released at once, because its Colab session token had been pushed to the public
  fork inside a bench-commit amend; the commit was rewritten, the session ended, and `run/` folders are git-ignored since.
- k1_t4: T4 timings and G1 trained in emulated bf16 (torch's `is_bf16_supported()` counts emulation on sm_75; Studio uses
  Unsloth's check, which picks fp16). Harness fixed; k2 cancelled and rerun as k2b.

## Needs your decision

1. **Compile threshold** (you removed it): break-even of the static compile vs eager is ~4,900 decisions (A) / ~22,000 (B)
   cold and ~1,400 / ~6,500 with a warm cache (G4). Keep always-compile, or bring back a threshold?
2. **LoRA compile** (X6): +54.6% at A, +29.5% at B over the final LoRA path (3 pairs each, G4); +45.0% / +24.9% on L4.
   VRAM -2.4% / -0.6% (G4), -2.4% / +0.25% (L4); fp32 G1 exact, G2 PASS at A (0.016) and B (0.020) on G4, 200-step runs
   0.915 / 0.748 over 3 seeds with no collapse; first step +30 s cold on G4, +75 s on L4. L4 G2 over 5 seeds per arm: A 0.021
   PASS; B 0.050 as measured, because the **PR** arm collapsed on seed 11 (LoRA compile 0/10 collapses, PR 1/10), 0.019 without
   that run; LoRA compile holdout at B 0.990 CE vs PR 1.025. On the T4 (fp16): +24-34% at A and B, VRAM -2.4% / -1.4%, G2 A 0.028 PASS over 3 seeds; at B **LoRA compile itself collapsed
   on seed 3407** (loss 40-60 1.225, grad norm 0.22), 0.025 without it. Across every LoRA-compile training run (G4 b8 + g2lc, L4 l6 + l7, T4 k5): 1 collapse in 21;
   the PR arm of the same batches: 2 in 21. The collapse belongs to the no-warmup recipe (decision 5), not to compile, but if you enable
   LoRA compile, decision 5 matters as much.
   Meets R1 on every check. Enable it? (Not committed: the plan says ask first.)
3. **cpp_wrapper on top of D1** (X4): +12.5% more at A on torch 2.13 (B200), crashes on torch 2.11. Gate on torch >= 2.13, or leave off?
4. **Whole-encoder compiled graph with in-graph checkpointing** (X5): +18% over the final at A, +18% VRAM (8,246 MiB, under the
   8,604 official peak), but it replaces Unsloth's offloaded checkpointing that the user picked. Allow as an opt-in?
5. **LoRA warmup**: a 10-step warmup removed the collapse in 8/8 runs (PR collapse rate 3/26, perf 3/26 without it).
   Change the Studio / decision default from 0 warmup steps?
6. **VRAM-costing caches**: GC off is +44.5% speed but +74.8% VRAM (13.8 GB, above the official ceiling); a cached bf16 copy of the
   fp32 weights per optimizer step would cost ~0.8 GB for an estimated <= 5% (casts are ~10% of kernel time at A, ~20% at B,
   of which only part is weight casts). Not built.
7. **New dependencies (report-only, not committed)**: xformers is unusable on torch 2.11 / sm_120 (and only adds a kernel on
   sm_100); FA4 (`flash-attn-4`), Liger and torchao FP8 were not measured: FA4 / FP8 target sm_90/sm_100, and attention is
   ~13% of kernel time at B, so even a 2x attention kernel is worth <= 6%.
8. **adamw_8bit**: -32% VRAM at the same speed (W); a user optimizer choice.
9. **unsloth_zoo `misc.py` fix**: already fixed in unsloth_zoo 2026.9.9; nothing to send upstream.

## Studio-pass open items

- C10 GC choice (`unsloth` offload vs `True`; GC off only fits under the ceiling for small micro-batches).
- Micro-batch default: at mb2 every kernel is tiny and the CPU sets the pace (GPU busy 19-27% on G4 / L4 / B200). On G4 full FT,
  mb8 x acc8 takes 64 samples in 0.421 s against 32 in 0.476 s at mb2 x acc16 (about 2.3x the throughput at a larger effective
  batch); picking the largest micro-batch that fits (and lowering accumulation to keep the effective batch) is the biggest
  remaining lever and needs no code.
- `adamw_8bit` as an option for large full fine-tunes.
- LoRA recipe: lr 8e-4 with no warmup collapses ~12% of seeds at mb2 x acc16; add a short warmup.
- Compile threshold and LoRA compile (decisions 1-2).
- Report-only dependency options (xformers / FA4 / Liger / torchao) stay out.
