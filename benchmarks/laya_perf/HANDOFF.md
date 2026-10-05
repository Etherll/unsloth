# Handoff: Laya decision-model training performance (continue on Linux/AWS)

Written 2026-10-05 from the Windows RTX 5090 session. The next session runs on the user's **AWS Linux GPU
machine** and should finish the work there.

## 1. What this is

- **PR under study:** unslothai/unsloth#12585 "Studio: fine-tune Laya decision models and serve them".
  - Author: NilayYadav. Head `c5cb7cba59`.
  - Adds `FastDecisionModel` and `DecisionTrainer` in `unsloth/models/decision.py`.
  - The model is ModernBERT-large plus laya's 2-layer `nn.TransformerEncoder` head (421M params).
- **User's goal:** make laya training as fast and as lean on VRAM as possible with Unsloth-style optimizations.
  - Keep loss and gradients within 0.03 of the PR.
  - Scope is **Unsloth core only**: `unsloth/models/decision.py`, new `unsloth/models/_decision_*.py` files, and `tests/test_decision_model.py`. Studio is a later pass.
  - Do **not** change the PR itself. It is another author's PR.
- **The plan** with the decision rules, candidates C1–C12, G1/G2 gates and phases is `benchmarks/laya_perf/PLAN.md` on the bench branch. Read it first. Rules in short:
  - **APPLY** if any of these holds:
    - speed ≥ +5% and VRAM ≤ +0.5%;
    - VRAM ≤ −2% and speed ≥ −1%;
    - speed ≥ +10% and VRAM ≤ the official laya script's peak.
  - **Report-only:** anything needing a new public argument, a user-setting override, or a new dependency.
- **The `/goal` condition** the user set lists five done-criteria. It is quoted at the end of this doc.

## 2. Where everything is

| What | Where |
|---|---|
| Code commits (5, the deliverable) | GitHub `Etherll/unsloth` branch **`laya-decision-perf`** (tip `f25ee6d96a`, based on PR head `c5cb7cba59`) |
| Bench harness, plan, results summaries, this doc | GitHub `Etherll/unsloth` branch **`laya-decision-perf-bench`** = `laya-decision-perf` + one commit adding `benchmarks/laya_perf/`. **Never merge or PR this branch**; it's tooling only. |
| Raw traces, grads, logs (31 GB) | Windows box `E:\laya-perf\out` (not uploaded). Small JSON/summary copies are in `benchmarks/laya_perf/results/`. |
| Report draft | `benchmarks/laya_perf/REPORT_SKELETON.md` (method + environment written; result sections are `<!--…-->` placeholders) |

On AWS:
```bash
git clone https://github.com/Etherll/unsloth.git && cd unsloth
git fetch origin laya-decision-perf laya-decision-perf-bench
git worktree add ../laya-base c5cb7cba59   # PR baseline tree (BASE)
git worktree add ../laya-perf laya-decision-perf-bench   # perf tree + harness (PERF)
```
Note: the bench branch includes the perf commits, so it works as the PERF tree. Keep code edits on
`laya-decision-perf` and move the bench folder along.

## 3. The five commits (`git log c5cb7cba59..laya-decision-perf`)

1. **Keep cuDNN attention out of the decision forward.**
   - `compute_loss` and `_logits` run inside `sdpa_kernel([FLASH, EFFICIENT, MATH])`.
   - Pathology fix: cuDNN bf16 SDPA rebuilt its plan for every length, ~100× slower on a B200.
   - Neutral on the 5090. fp32 G1 exact.
2. **Batch similar lengths in `_logits`** (evaluate/calibrate), with order restored.
   - 400 holdout decisions: 1.18 → 0.98 s (+16.5%), identical logits.
3. **Fused AdamW** for `optim="adamw_torch"` when all trained params are fp32 on CUDA, unless the caller set foreach/fused.
   - Removes foreach's full-size temporaries, which were the VRAM peak: −8.3% reserved at mb2×acc16.
   - fused vs foreach verified to 1 ulp (`scripts/adam_check.py`).
4. **Lean LoRA forward.**
   - For plain LoRA Linears (1 adapter, no dropout/DoRA): `base(x) + B(A(x)) * s`, skipping PEFT's checks, its fp32 round-trip of x, and the multiply when s == 1.
   - **Bit-identical to PEFT** (test asserts `torch.equal`). +7.9% / +7.6%.
   - An earlier variant that shared one x cast was NOT exact and was dropped.
5. **Compile ModernBERT layers for full fine-tunes.** See section 6.

Windows: `python -m pytest tests/test_decision_model.py -q` → **30 passed** on the tip.
**WSL/Linux pytest has not been run yet.**

## 4. Results so far (Windows RTX 5090, torch 2.10, bf16, GC="unsloth")

Geometries:
- **A** = mb2×acc16 (the official script's default; launch-bound);
- **B** = mb8×acc8 (Studio's default; GPU-bound).

A/B runs use 3 ABAB fresh-process pairs. Use the 25th-percentile step time (`scripts/reab.py`), because desktop apps cause 30–40% slow bursts.

| Stacked build vs PR | Speed | Reserved VRAM | Note |
|---|---|---|---|
| Full A | +5.6% | −14.0% (8650→7442 MiB) | measured **before** the compile SDPA fix |
| Full B | +14.6% | −6.6% | before the SDPA fix |
| LoRA A | +7.8% | ~0 | final |
| LoRA B | +5.9–6.7% | −1.3% | final |

After the SDPA fix (commit 5), the C10 runs showed perf full A at ~0.73 s/step vs the PR's ~1.11 (≈ +50%), and B at ~0.62 vs 0.75. **Not confirmed.**
- The confirming re-measure (`results/stack2/full_A`, 2 pairs) ran during desktop interference: the PR arm was 1.5–1.76 s instead of ~1.1.
- Treat it as unreliable and **redo it on AWS**.

Other numbers:
- **Official laya script** (fp32, as shipped): peak reserved 8616 MiB at A, 8448 at B. This is the VRAM ceiling for rule 3.
- **G2** (60 steps, dropout on): all four geometry/mode combos PASS (window |Δloss| ≤ 0.02, holdout CE |Δ| ≤ 0.011).
  - LoRA B holdout accuracy differed by 0.04, inside the noise; see section 5.
- **2-epoch full fine-tune** (5,600 decisions, A, compile forced, old SDPA path): holdout CE |Δ| 0.015, accuracy |Δ| 0.010 → PASS.
  - Wall time 406 → 386 s, VRAM 8676 → 7516 MiB.
- **2-epoch LoRA:**
  - Exact-forward perf run 1: CE |Δ| 0.006, accuracy |Δ| 0.028 → PASS.
  - Perf run 2: accuracy 0.715 vs the PR's 0.80 → FAIL.
  - An older non-exact variant collapsed to 0.39.
  - Two PR runs: 0.8025 and 0.7925.
  - With the forward now bit-identical and AdamW equal to 1 ulp, the remaining difference is GPU nondeterminism plus the PR recipe's instability (lr 8e-4, no warmup, grad spikes). **This needs more seeds to settle** (section 7).

Phase-1 profile (`results/prof/*/summary.txt`, torch.profiler, Windows and WSL):

| | PR mb2×acc16 | PR mb8×acc8 |
|---|---|---|
| GPU busy | ~19% Windows / 27% WSL (launch-bound; ~28k launches/step) | ~87% (GPU-bound) |
| Time split | backward 66% of wall | GEMM ~55% of GPU time, elementwise ~25% (largely autocast fp32→bf16 weight casts per micro-batch), attention ~9%, GC offload memcpy ~6%, GC recompute ~21% |

- Padding is only 0.8% thanks to the length-grouped sampler, so unpadding can't win much.
- Head ≤1% and data ≤0.5% → those candidates were skipped.
- VRAM peak = foreach AdamW step (6.4 GB static + ~1.6 GB temporaries).
- WSL is faster than Windows for the PR (A 0.89 vs ~1.1 s/step): lower launch overhead.

Candidate verdicts so far (raw numbers in `results/ab/*/summary.json` and `robust.json`, `results/abw` for WSL):

| # | Candidate | Verdict |
|---|---|---|
| C1 | cuDNN guard | APPLIED (pathology rule; neutral here, exact) |
| C2 | pad8 / pad64 | REJECTED (−0.1% / −0.9%) |
| C3 | unpadding | REJECTED: packed dense SDPA −30%; flex −25%; packed FA2 varlen (WSL) +1% and noisy; stock FA2 +0.8%; xformers unusable on Windows (dev build missing `mslk`), not installed in WSL |
| C4 | regional compile | APPLIED for full fine-tune; LoRA REJECTED (−3% at A) |
| C5 | Triton GEGLU/RoPE/LN | SKIPPED (plan: only if C4 rejected) |
| C6 | head varlen | SKIPPED (head <3%) |
| C7 | LoRA bf16 storage | REJECTED (+2%; would also round the saved merged checkpoint) |
| C8 | sorted `_logits` | APPLIED |
| C9 | data path | SKIPPED (<3%) |
| C10 | GC modes, report-only | GC off: +44% at B but 13.8 GB (> official ceiling); GC `True`: same speed, +6.5% VRAM |
| C11 | optimizers | fused AdamW APPLIED as an implementation of `adamw_torch` (flag this to the user); `adamw_8bit` report-only: −32% VRAM, same speed |
| C12 | CUDA graphs | REJECTED: crashes with GC recompute (CUDAGraphs output overwritten) |
| extra | rope_once (dedup rotary) | REJECTED (+3.25% < 5%), exact |
| extra | lean LoRA | APPLIED (commit 4) |

**G1 note (important for the report):** the PR's own bf16 grads differ from its fp32 grads by up to 69% in grad norm (relative L2 0.22), and two identical runs are not bit-equal.

So G1 is judged as:
- **fp32 exactness:** per-batch |Δloss| ≤ 0.03 and grad-norm relative difference ≤ 0.03. The stacked build reaches relative L2 2e-5 (full) and 5e-7 (LoRA).
- **bf16 loss:** ≤ 0.03.
- **bf16 grads:** no further from fp32 than the PR's own bf16 grads.

This reasoning is written into `REPORT_SKELETON.md`.

## 5. Gotchas learned (don't rediscover)

- **unsloth_zoo bug (2026.9.x):** `unsloth_zoo/temporary_patches/misc.py` runs `exec("from transformers.quantizers.auto import (...)", globals())` with names from `dir(module)`, so it can overwrite misc's own `__name__`.
  - Every function defined after that, including its SDPA wrapper installed into `ALL_ATTENTION_FUNCTIONS["sdpa"]`, reports `transformers.quantizers.auto` as its module.
  - torch 2.11's dynamo then fails a guard (`InternalTorchDynamoError: module 'transformers.quantizers.auto' has no attribute 'torch'`) and silently runs eager.
  - Commit 5 works around it. The zoo bug itself is unfixed and worth reporting or fixing in unsloth_zoo, with the user's OK.
- Unsloth pins the inductor cache to `$TMPDIR/torchinductor_<user>`, so `TORCHINDUCTOR_CACHE_DIR` is overridden. To measure a cold compile, move that dir aside.
- "unsloth" offloaded GC is configured for the autocast dtype, so fp32 parity runs must use `--gc true` on both arms.
- `scripts/parity.py` doesn't call `trainer.train()`, so it needs `--opts compile` to exercise compile on the perf tree.
- The bench must force compile with `--opts compile_always`. That opt now does nothing (the threshold was removed) but is harmless. `no_compile` likewise.
- Never run pytest or other GPU work while an A/B is running; it skews the pair.
- bf16 training is nondeterministic (attention-backward and embedding atomics), so the G2 0.03 gates sit near the noise floor. Always run a PR-vs-PR control.

## 6. Revisit: is the compile worth it? (user asked for this explicitly)

**Current code** (`_compile_encoder_layers`, called from `DecisionTrainer.train`) compiles every ModernBERT layer with `layer.compile(dynamic=True)` when all of these hold:
- full fine-tune;
- CUDA;
- Triton available;
- `model_type == "modernbert"` (covers laya-english, laya-typed-decisions, and the mmBERT laya-multilingual);
- `UNSLOTH_COMPILE_DISABLE` not set.

Before compiling, it switches the encoder to a private plain-SDPA attention, `"unsloth_decision_sdpa"` (`_encoder_sdpa`).

The user had me **remove the 16,000-decision minimum** (it was based on pre-fix numbers: ~15 s cold compile, ~1.5 ms saved per decision). Now even short runs compile.

**Questions to answer on Linux:**
1. **Real speedup after the SDPA fix.** Run 3–5 ABAB pairs on an idle GPU for full A and full B, PR vs perf with `compile_always`. Windows hinted A ≈ +50% and B ≈ +20%, but that is unconfirmed. On Linux (torch ≥ 2.11, new zoo), compile had *never* worked before the fix (−2 to −5%), so this is the first real Linux measurement.
2. **Compile cost.** First-step time cold (empty inductor + Triton cache) vs warm, plus the eval-mode recompile when `evaluate`/`_logits` runs. Is it seconds or tens of seconds on the AWS GPU?
3. **Break-even.** Decisions needed to repay a cold compile: cost ÷ saved per decision, where saved per decision = (s/step_PR − s/step_perf) ÷ (mb × acc). If break-even is small (< a few thousand decisions), keep "always compile". If not, propose a threshold back to the user. Don't silently add one; they removed it.
4. **Numerics.** fp32 G1 relative L2 ≈ 2e-5 vs eager was already shown. On Linux re-run G1 (fp32) and G2, plus one 2-epoch full fine-tune with a PR-vs-PR control.
5. **VRAM.** Compile lowered the fwd/bwd peak by ~400 MiB on Windows. Confirm on Linux with fused AdamW.
6. **Should LoRA compile now?** The LoRA "no gain" verdict (−3% A, +4% B) predates both the lean forward and the SDPA fix. Re-test `--mode lora` with `compile` on Linux. If it is ≥ +5% under the rules, it's a candidate to enable.
7. **Scope.** Should `model_type != "modernbert"` encoders compile? Currently no: the private SDPA assumes ModernBERT's bidirectional, mask-driven attention and its `encoder.layers` layout. Only revisit if the user asks.

## 7. Remaining work (to satisfy the user's goal)

1. **Linux environment.**
   - Python 3.12; torch per the machine (record the version); transformers 5.5.0; triton; flash-attn if available; peft; accelerate.
   - unsloth_zoo: the version pip resolves. Record it, and check whether it still has the misc.py `__name__` bug.
   - Keep caches and venvs on the big data disk.
   - The harness hardcodes some Windows paths:
     - `scripts/prep.py` reads the dataset parquet from an E: blob path → point it at the HF cache, or `datasets.load_dataset("LocalLLaMA/typed-decisions", "all", split="train")`;
     - the model goes in `$LP_ROOT/model` (`snapshot_download("convaiinnovations/laya")`, root files only);
     - `bench.py`, `parity.py`, `official_bench.py` and `c8_eval.py` honour `LP_ROOT`;
     - bench pins CPU affinity to 0–15 and nice −5. On Linux, unprivileged nice fails harmlessly; drop or change the affinity for the AWS CPU count;
     - adapt `env-wsl.sh` to an `env-linux.sh`.
2. **Re-baseline on the AWS GPU:** PR full and LoRA × A and B, the official script (A/B) for the VRAM ceiling, and `prof.py` profiles for the PR and the final build.
3. **Stacked A/B for all four combos** (3+ pairs, or 5 if noisy), plus G1 (fp32 + bf16) and G2.
4. **2-epoch runs:**
   - full fine-tune: PR vs perf, plus one PR-vs-PR control;
   - LoRA: at least 3 PR runs and 3 perf runs, to show the spread is the PR recipe's run-to-run variance. Compare mean ± range of holdout CE/accuracy, not single runs.
5. **pytest on Linux:** `python -m pytest tests/test_decision_model.py -q` in the perf tree.
6. **Fill in REPORT.md** (`benchmarks/laya_perf/REPORT_SKELETON.md` → `REPORT.md`):
   - environments for both machines;
   - Phase-1 breakdown (Windows numbers are in `results/prof`; add Linux);
   - one row per C1–C12 (+ extras) with s/step, tok/s, peak alloc/reserved, G1, verdict and rule;
   - final stacked table;
   - commit list;
   - Studio-pass open items: C10 GC choice, adamw_8bit, the LoRA lr 8e-4 / no-warmup instability, the zoo bug.
7. **Report to the user:** the headline and what was pushed. The only pushes are the two `Etherll/unsloth` branches above; do **not** open PRs or push to `unslothai/unsloth` without asking.

## 8. Constraints and preferences

- Commit as `Etherll <61019402+Etherll@users.noreply.github.com>`; never override user.email; never edit git config.
- Ask before:
  - any new public argument;
  - a new required dependency;
  - any Studio edit;
  - pushing anywhere other than the two branches above;
  - touching the PR author's branch.
- The user wants percentages for speed/VRAM, concise answers, and real profiler evidence for "what is slow / what uses VRAM".
- On the Windows box everything had to live on E:. On AWS, keep data, caches and venvs on the large volume.

## 9. Suggested skills

- `diagnosing-bugs`: if compile or LoRA stability misbehaves on Linux (the "slow / regression" loop).
- `verification-before-completion`: before claiming any of the five goal conditions are met.
- `fork-pr-ci` / `pr-repro-ci`: only if the user later wants fork CI on `laya-decision-perf`.
- `unsloth-pr-mirror-review`: if the user decides to offer these commits to PR #12585's author (ask first).
- `humanizer`: for the final REPORT.md prose and any write-up.
- `caveman-commit`: optional, for further commit messages (repo style is short imperative sentences).

## 10. The user's goal condition (verbatim)

> Execute the plan at C:\Users\pc\.claude\plans\i-want-to-continue-elegant-walrus.md (laya decision-model training performance, Unsloth core only). Done when ALL are true and shown in your own output: (1) E:\laya-perf\REPORT.md contains the Phase-1 profile breakdown (Windows and WSL) and one row for each candidate C1-C12 with measured s/step, tok/s, peak alloc/reserved, G1 parity (per-batch |dloss| <= 0.03 and grad-norm relative diff <= 0.03, dropout off) and a verdict APPLIED / REJECTED / REPORT-ONLY / SKIPPED(<3% of step) citing the plan's decision rule; (2) every APPLIED change is committed on local branch laya-decision-perf in worktree E:\orca\workspaces\unsloth\laya-decision-perf, touching only unsloth/models/decision.py, new unsloth/models/_decision_*.py files and tests/test_decision_model.py; (3) `python -m pytest tests/test_decision_model.py -q` passes in that worktree on Windows (and in WSL if Linux-only code paths were added), with the output shown; (4) the final stacked build vs the PR baseline is reported for full and LoRA at mb2xacc16 and mb8xacc8 (speed and VRAM), passes G2 (10-step-window |dloss| <= 0.03), and a full 2-epoch run shows holdout CE and accuracy within 0.03 of the baseline; (5) all outputs, venvs and caches are on E: (…), or any exception is justified in REPORT.md; nothing was pushed, the baseline worktree … and all studio/ files are unchanged, and git config was not edited. …

On AWS, the paths map to the Linux worktrees. The user has since explicitly asked for the two branches to be pushed, so "nothing was pushed" now means nothing beyond those two branches.
