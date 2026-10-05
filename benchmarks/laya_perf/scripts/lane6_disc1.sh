# GPU6 discovery round 1: screen the launch-overhead candidates at A (single runs), then the profiles.
source $(dirname $0)/../env-linux.sh; lane 6; source $S/lanelib.sh; cd $LP_ROOT
SC=$RES/screen; mkdir -p $SC $OUT/nsys $OUT/mem $OUT/pyspy $OUT/trace $RES/prof
A="--mb 2 --acc 16"; B="--mb 8 --acc 8"
SB="$PY $S/bench.py --steps 24 --warmup 6"
job scr_perf_full_A    $SB --tree $PERF --mode full $A --out $SC/perf_full_A.json
job scr_cg_full_A      $SB --tree $PERF --mode full $A --opts cg --out $SC/cg_full_A.json
job scr_cpp_full_A     $SB --tree $PERF --mode full $A --opts cpp_wrapper --out $SC/cpp_full_A.json
job scr_static_full_A  $SB --tree $PERF --mode full $A --opts static64 --out $SC/static_full_A.json
$PY $S/show.py $SC/*.json | tee $SC/summary.txt
# nsys: NVTX step/forward/backward/optimizer (+ enc_layer = GC recompute when eager), steps 7-9.
NSYS=/usr/local/cuda-13.1/bin/nsys
NS="$NSYS profile -t cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true"
NB="$PY $S/bench.py --steps 10 --warmup 2 --nvtx --cuprof 6,9"
for g in A B; do G=${!g}
    LP_LAYER_RANGES=0 job nsys_perf_full_$g $NS -o $OUT/nsys/perf_full_$g $NB --tree $PERF --mode full $G --out $OUT/nsys/perf_full_$g.json
    job nsys_perf_lora_$g $NS -o $OUT/nsys/perf_lora_$g $NB --tree $PERF --mode lora $G --out $OUT/nsys/perf_lora_$g.json
    job nsys_pr_full_$g $NS -o $OUT/nsys/pr_full_$g $NB --tree $BASE --mode full $G --out $OUT/nsys/pr_full_$g.json
done
for f in $OUT/nsys/*.nsys-rep; do $PY $S/nsys_summary.py $f > /dev/null 2>&1; cp ${f%.nsys-rep}_summary.txt $RES/prof/ 2>/dev/null; done
# Memory snapshot at step 3 (what makes up the VRAM peak).
for c in "full A" "full B" "lora B"; do set -- $c; G=${!2}
    job mem_perf_$1_$2 $PY $S/bench.py --steps 5 --warmup 1 --tree $PERF --mode $1 $G --memsnap $OUT/mem/perf_$1_$2.pickle --out $OUT/mem/perf_$1_$2.json
    $PY $S/memtop.py $OUT/mem/perf_$1_$2.pickle > $RES/prof/mem_perf_$1_$2.txt 2>&1
done
# py-spy --native at A.
for m in full lora; do
    job pyspy_perf_${m}_A $LP_ROOT/venv/bin/py-spy record --native --rate 200 -f raw -o $OUT/pyspy/perf_${m}_A.txt -- $PY $S/bench.py --steps 16 --warmup 4 --tree $PERF --mode $m $A --out $OUT/pyspy/perf_${m}_A.json
done
# Dynamo graph breaks / recompiles / perf hints + tlparse, eval in the middle (eval-mode recompile).
LP_TORCH_LOGS=1 TORCH_TRACE=$OUT/trace/perf_full_A job dynamo_perf_full_A $PY $S/bench.py --steps 16 --warmup 4 --tree $PERF --mode full $A --eval-at 8 --out $OUT/trace/perf_full_A.json
$LP_ROOT/venv/bin/tlparse $OUT/trace/perf_full_A -o $OUT/trace/perf_full_A_tlparse --overwrite --no-browser > /dev/null 2>&1
grep -iE "graph break|recompil|hint" $OUT/logs/dynamo_perf_full_A.log | sed 's/[0-9a-f]\{8,\}//g' | sort | uniq -c | sort -rn | head -40 > $RES/prof/dynamo_perf_full_A.txt
# torch.profiler (shapes + stacks) for the rest of the Phase-1 table.
P=$OUT/prof; BP="$PY $S/bench.py --steps 9 --warmup 2"
LP_LAYER_RANGES=0 job prof_perf_full_B $BP --tree $PERF --mode full $B --prof $P/perf_full_B --out $P/perf_full_B.json
for g in A B; do G=${!g}
    job prof_perf_lora_$g $BP --tree $PERF --mode lora $G --prof $P/perf_lora_$g --out $P/perf_lora_$g.json
    job prof_pr_full_$g   $BP --tree $BASE --mode full $G --prof $P/pr_full_$g --out $P/pr_full_$g.json
done
for d in $P/*/; do n=$(basename $d); [ -f $d/trace.json ] && [ ! -f $RES/prof/$n.summary.txt ] && $PY $S/prof.py $d/trace.json > /dev/null 2>&1 && cp $d/summary.txt $RES/prof/$n.summary.txt; done
echo LANE6_DISC1_DONE
