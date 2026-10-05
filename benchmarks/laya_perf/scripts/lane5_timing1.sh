# GPU5 timing, phase 1: interference control, then PR re-baseline + stacked A/B (current stack) for
# full/LoRA x A/B, the official script (VRAM ceiling), C1 alone, and compile cost (cold vs warm, eval recompile).
source $(dirname $0)/../env-linux.sh; lane 5; source $S/lanelib.sh; cd $LP_ROOT
AB="$PY $S/run_ab.py"
FA="--mode full --mb 2 --acc 16 --steps 24"; FB="--mode full --mb 8 --acc 8 --steps 16"
LA="--mode lora --mb 2 --acc 16 --steps 24"; LB="--mode lora --mb 8 --acc 8 --steps 16"
# Timing jobs hold lanes 6/7 when the interference control showed they matter (logs/timing_hold).
tjob() { if [ -f $LP_ROOT/logs/timing_hold ]; then $PY $S/gpufree.py 5; hold 6 7; job "$@"; local rc=$?; release 6 7; return $rc; else job "$@"; fi; }

# 1. Interference control: one PR-vs-perf full-A pair with lanes 6/7 idle, then one with them busy.
if [ ! -f $AB_OUT/ic_busy/summary.json ]; then
    $PY $S/gpufree.py 5; hold 6 7
    job ic_idle $AB ic_idle "$FA" "--tree $BASE" "--tree $PERF" 1 1
    release 6 7
    for i in $(seq 120); do [ -f $LP_ROOT/logs/running_gpu6 ] && [ -f $LP_ROOT/logs/running_gpu7 ] && break; sleep 30; done
    jlog "ic_busy with lanes: $(ls $LP_ROOT/logs/running_gpu* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
    job ic_busy $AB ic_busy "$FA" "--tree $BASE" "--tree $PERF" 1 1
    $PY - <<'PY' | tee -a $RES/jobs.log
import json, os
o = os.environ["AB_OUT"]
i, b = (json.load(open(f"{o}/{n}/summary.json")) for n in ("ic_idle", "ic_busy"))
d = {k: b[k] / i[k] - 1 for k in ("A_p25", "B_p25")}
d["speedup_idle"], d["speedup_busy"] = i["speedup"], b["speedup"]
worst = max(abs(d["A_p25"]), abs(d["B_p25"]))
print("IC", json.dumps({k: round(v, 4) for k, v in d.items()}), "-> hold 6/7 for timing" if worst > 0.02 else "-> no hold")
if worst > 0.02:
    open(os.environ["LP_ROOT"] + "/logs/timing_hold", "w").write(json.dumps(d))
PY
fi
# 2+3. Stacked A/B, PR vs current stack (its PR arms are the re-baseline).
tjob stack1_full_A $AB stack1_full_A "$FA" "--tree $BASE" "--tree $PERF"
tjob stack1_full_B $AB stack1_full_B "$FB" "--tree $BASE" "--tree $PERF"
tjob stack1_lora_A $AB stack1_lora_A "$LA" "--tree $BASE" "--tree $PERF"
tjob stack1_lora_B $AB stack1_lora_B "$LB" "--tree $BASE" "--tree $PERF"
# Official laya script as shipped (fp32): the VRAM ceiling.
mkdir -p $RES/official
tjob official_A $PY $S/official_bench.py --mb 2 --acc 16 --steps 24 --out $RES/official/official_A.json
tjob official_B $PY $S/official_bench.py --mb 8 --acc 8 --steps 16 --out $RES/official/official_B.json
# C1 alone on the B200 (the cliff it was written for): PR vs PR + guard (forward and backward).
tjob c1_full_A $AB c1_full_A "$FA --tree $BASE" "" "--opts cudnn_guard"
tjob c1_full_B $AB c1_full_B "$FB --tree $BASE" "" "--opts cudnn_guard"
# 4. Compile cost: perf full A with the lane's inductor + Triton caches moved aside (cold), then warm.
# Holdout evals at steps 12 and 20: the first pays the eval-mode recompile, the second is warm.
mkdir -p $RES/compile
aside() {
    local ts=$(date +%s)
    [ -d $TMPDIR/torchinductor_$USER ] && mv $TMPDIR/torchinductor_$USER $LP_ROOT/cache/gpu5/inductor_aside_$ts
    [ -d $TRITON_CACHE_DIR ] && mv $TRITON_CACHE_DIR $LP_ROOT/cache/gpu5/triton_aside_$ts
    mkdir -p $TRITON_CACHE_DIR
}
CC="$PY $S/bench.py --tree $PERF $FA --eval-at 12,20"
aside; tjob compile_cold_full_A $CC --out $RES/compile/cold_full_A.json
tjob compile_warm_full_A $CC --out $RES/compile/warm_full_A.json
tjob compile_pr_full_A $PY $S/bench.py --tree $BASE $FA --eval-at 12,20 --out $RES/compile/pr_full_A.json
aside; tjob compile_cold_full_B $PY $S/bench.py --tree $PERF $FB --eval-at 8,12 --out $RES/compile/cold_full_B.json
tjob compile_warm_full_B $PY $S/bench.py --tree $PERF $FB --eval-at 8,12 --out $RES/compile/warm_full_B.json
tjob compile_pr_full_B $PY $S/bench.py --tree $BASE $FB --eval-at 8,12 --out $RES/compile/pr_full_B.json
echo LANE5_PHASE1_DONE
