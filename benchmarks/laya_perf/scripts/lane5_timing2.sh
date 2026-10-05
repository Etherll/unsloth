# GPU5 timing, phase 1 (continued): load-gated A/Bs that hold lanes 6/7 per run only (holdlib), then the
# official script, C1 alone and compile cost. Re-running an existing A/B re-reads its clean pairs and only
# redoes discarded runs or extends to 5 pairs when the spread is > 3%.
source $(dirname $0)/../env-linux.sh; lane 5; source $S/lanelib.sh; cd $LP_ROOT
[ -f $LP_ROOT/logs/timing_hold ] && export LP_HOLD="6 7"
AB="$PY $S/run_ab.py"; W="$PY $S/waitrun.py"
FA="--mode full --mb 2 --acc 16 --steps 24"; FB="--mode full --mb 8 --acc 8 --steps 16"
LA="--mode lora --mb 2 --acc 16 --steps 24"; LB="--mode lora --mb 8 --acc 8 --steps 16"
for c in "full_A $FA" "full_B $FB" "lora_A $LA" "lora_B $LB"; do set -- $c; n=$1; shift
    job stack1_$n $AB stack1_$n "$*" "--tree $BASE" "--tree $PERF"
done
mkdir -p $RES/official $RES/compile
job official_A $W $PY $S/official_bench.py --mb 2 --acc 16 --steps 24 --out $RES/official/official_A.json
job official_B $W $PY $S/official_bench.py --mb 8 --acc 8 --steps 16 --out $RES/official/official_B.json
job c1_full_A $AB c1_full_A "$FA --tree $BASE" "" "--opts cudnn_guard"
job c1_full_B $AB c1_full_B "$FB --tree $BASE" "" "--opts cudnn_guard"
aside() {
    local ts=$(date +%s)
    [ -d $TMPDIR/torchinductor_$USER ] && mv $TMPDIR/torchinductor_$USER $LP_ROOT/cache/gpu5/inductor_aside_$ts
    [ -d $TRITON_CACHE_DIR ] && mv $TRITON_CACHE_DIR $LP_ROOT/cache/gpu5/triton_aside_$ts
    mkdir -p $TRITON_CACHE_DIR
}
CC="$PY $S/bench.py --tree $PERF $FA --eval-at 12,20"
aside; job compile_cold_full_A $W $CC --out $RES/compile/cold_full_A.json
job compile_warm_full_A $W $CC --out $RES/compile/warm_full_A.json
job compile_pr_full_A $W $PY $S/bench.py --tree $BASE $FA --eval-at 12,20 --out $RES/compile/pr_full_A.json
aside; job compile_cold_full_B $W $PY $S/bench.py --tree $PERF $FB --eval-at 8,12 --out $RES/compile/cold_full_B.json
job compile_warm_full_B $W $PY $S/bench.py --tree $PERF $FB --eval-at 8,12 --out $RES/compile/warm_full_B.json
job compile_pr_full_B $W $PY $S/bench.py --tree $BASE $FB --eval-at 8,12 --out $RES/compile/pr_full_B.json
echo LANE5_PHASE1_DONE
