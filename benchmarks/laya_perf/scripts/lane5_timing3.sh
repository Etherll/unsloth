# GPU5 timing, phase 2: candidate D1 (static shapes + C++ wrapper, tree $LP_ROOT/cand) vs the current stack,
# the final full build vs the PR, and the candidate's compile cost (cold vs warm).
source $(dirname $0)/../env-linux.sh; lane 5; source $S/lanelib.sh; cd $LP_ROOT
[ -f $LP_ROOT/logs/timing_hold ] && export LP_HOLD="6 7"
AB="$PY $S/run_ab.py"; W="$PY $S/waitrun.py"; C=$LP_ROOT/cand
FA="--mode full --mb 2 --acc 16 --steps 24"; FB="--mode full --mb 8 --acc 8 --steps 16"
job d1_full_A $AB d1_full_A "$FA" "--tree $PERF" "--tree $C"
job d1_full_B $AB d1_full_B "$FB" "--tree $PERF" "--tree $C"
job final_full_A $AB final_full_A "$FA" "--tree $BASE" "--tree $C"
job final_full_B $AB final_full_B "$FB" "--tree $BASE" "--tree $C"
aside() {
    local ts=$(date +%s)
    [ -d $TMPDIR/torchinductor_$USER ] && mv $TMPDIR/torchinductor_$USER $LP_ROOT/cache/gpu5/inductor_aside_$ts
    [ -d $TRITON_CACHE_DIR ] && mv $TRITON_CACHE_DIR $LP_ROOT/cache/gpu5/triton_aside_$ts
    mkdir -p $TRITON_CACHE_DIR
}
mkdir -p $RES/compile
aside; job compile_cold_cand_A $W $PY $S/bench.py --tree $C $FA --eval-at 12,20 --out $RES/compile/cold_cand_A.json
job compile_warm_cand_A $W $PY $S/bench.py --tree $C $FA --eval-at 12,20 --out $RES/compile/warm_cand_A.json
aside; job compile_cold_cand_B $W $PY $S/bench.py --tree $C $FB --eval-at 8,12 --out $RES/compile/cold_cand_B.json
job compile_warm_cand_B $W $PY $S/bench.py --tree $C $FB --eval-at 8,12 --out $RES/compile/warm_cand_B.json
UNSLOTH_COMPILE_DISABLE=1 job nocompile_full_A $W $PY $S/bench.py --tree $PERF $FA --out $RES/compile/nocompile_full_A.json
UNSLOTH_COMPILE_DISABLE=1 job nocompile_full_B $W $PY $S/bench.py --tree $PERF $FB --out $RES/compile/nocompile_full_B.json
echo LANE5_PHASE2_DONE
