# GPU6 discovery, batch 1: profile the current build (and the PR for the Linux Phase-1 table) at A and B.
source $(dirname $0)/../env-linux.sh; lane 6; source $S/lanelib.sh; cd $LP_ROOT
NSYS=/usr/local/cuda-13.1/bin/nsys
P=$OUT/prof; mkdir -p $P $OUT/nsys $OUT/mem $OUT/pyspy $OUT/trace $RES/prof
A="--mb 2 --acc 16"; B="--mb 8 --acc 8"
BP="$PY $S/bench.py --steps 9 --warmup 2"
# torch.profiler (wait 2, warmup 2, active 3). Compiled layers get no per-layer hooks (LP_LAYER_RANGES=0).
for g in A B; do G=${!g}
    LP_LAYER_RANGES=0 job prof_perf_full_$g $BP --tree $PERF --mode full $G --prof $P/perf_full_$g --out $P/perf_full_$g.json
    job prof_perf_lora_$g $BP --tree $PERF --mode lora $G --prof $P/perf_lora_$g --out $P/perf_lora_$g.json
    job prof_pr_full_$g   $BP --tree $BASE --mode full $G --prof $P/pr_full_$g --out $P/pr_full_$g.json
    job prof_pr_lora_$g   $BP --tree $BASE --mode lora $G --prof $P/pr_lora_$g --out $P/pr_lora_$g.json
done
for d in $P/*/; do [ -f $d/trace.json ] && $PY $S/prof.py $d/trace.json > /dev/null 2>&1; n=$(basename $d); cp $d/summary.txt $RES/prof/$n.summary.txt 2>/dev/null; done
# nsys: NVTX step/forward/backward/optimizer (+ enc_layer = GC recompute inside backward when eager), steps 7-9.
NS="$NSYS profile -t cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true"
NB="$PY $S/bench.py --steps 10 --warmup 2 --nvtx --cuprof 6,9"
for g in A B; do G=${!g}
    LP_LAYER_RANGES=0 job nsys_perf_full_$g $NS -o $OUT/nsys/perf_full_$g $NB --tree $PERF --mode full $G --out $OUT/nsys/perf_full_$g.json
    job nsys_perf_lora_$g $NS -o $OUT/nsys/perf_lora_$g $NB --tree $PERF --mode lora $G --out $OUT/nsys/perf_lora_$g.json
done
job nsys_pr_full_A $NS -o $OUT/nsys/pr_full_A $NB --tree $BASE --mode full $A --out $OUT/nsys/pr_full_A.json
# Memory snapshot at step 3 (what makes up the VRAM peak).
for c in "full A" "full B" "lora B" "lora A"; do set -- $c; G=${!2}
    job mem_perf_$1_$2 $PY $S/bench.py --steps 5 --warmup 1 --tree $PERF --mode $1 $G --memsnap $OUT/mem/perf_$1_$2.pickle --out $OUT/mem/perf_$1_$2.json
    $PY $S/memtop.py $OUT/mem/perf_$1_$2.pickle > $RES/prof/mem_perf_$1_$2.txt 2>&1
done
job mem_pr_full_A $PY $S/bench.py --steps 5 --warmup 1 --tree $BASE --mode full $A --memsnap $OUT/mem/pr_full_A.pickle --out $OUT/mem/pr_full_A.json
$PY $S/memtop.py $OUT/mem/pr_full_A.pickle > $RES/prof/mem_pr_full_A.txt 2>&1
# py-spy --native at geometry A (CPU time; A is launch-bound).
for m in full lora; do
    job pyspy_perf_${m}_A $LP_ROOT/venv/bin/py-spy record --native --rate 250 -f raw -o $OUT/pyspy/perf_${m}_A.txt -- $PY $S/bench.py --steps 16 --warmup 4 --tree $PERF --mode $m $A --out $OUT/pyspy/perf_${m}_A.json
done
# Dynamo graph breaks / recompiles / perf hints + tlparse, with an eval in the middle (eval-mode recompile).
LP_TORCH_LOGS=1 TORCH_TRACE=$OUT/trace/perf_full_A job dynamo_perf_full_A $PY $S/bench.py --steps 16 --warmup 4 --tree $PERF --mode full $A --eval-at 8 --out $OUT/trace/perf_full_A.json
$LP_ROOT/venv/bin/tlparse $OUT/trace/perf_full_A -o $OUT/trace/perf_full_A_tlparse --overwrite --no-browser > /dev/null 2>&1
grep -iE "graph break|recompil|perf_hint|hint" $OUT/logs/dynamo_perf_full_A.log | sort | uniq -c | sort -rn | head -40 > $RES/prof/dynamo_perf_full_A.txt
echo LANE6_PROF_DONE
