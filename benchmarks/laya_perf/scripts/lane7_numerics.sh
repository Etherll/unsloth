# GPU7: G1 (fp32 + bf16, PR-vs-PR control) and G2 for the current stack, then the long-run checks.
source $(dirname $0)/../env-linux.sh; lane 7; source $S/lanelib.sh; cd $LP_ROOT
G=$RES/g1; mkdir -p $G $RES/g2 $RES/long
P="$PY $S/parity.py"
for mode in full lora; do
    job g1_pr_${mode}_fp32  $P --tree $BASE --mode $mode --precision fp32 --gc true --out $G/pr_${mode}_fp32.json
    job g1_perf_${mode}_fp32 $P --tree $PERF --mode $mode --precision fp32 --gc true --opts trainer_compile --out $G/perf_${mode}_fp32.json
    job g1_pr_${mode}_bf16  $P --tree $BASE --mode $mode --precision bf16 --out $G/pr_${mode}_bf16.json
    job g1_pr2_${mode}_bf16 $P --tree $BASE --mode $mode --precision bf16 --out $G/pr2_${mode}_bf16.json
    job g1_perf_${mode}_bf16 $P --tree $PERF --mode $mode --precision bf16 --opts trainer_compile --out $G/perf_${mode}_bf16.json
    for cmp in "pr_${mode}_fp32 perf_${mode}_fp32" "pr_${mode}_bf16 perf_${mode}_bf16" "pr_${mode}_bf16 pr2_${mode}_bf16" \
               "pr_${mode}_fp32 pr_${mode}_bf16" "pr_${mode}_fp32 perf_${mode}_bf16"; do
        set -- $cmp
        echo "G1 $1 vs $2 $($PY $S/pcompare.py $G/$1.json $G/$2.json 2>/dev/null | tail -1)" | tee -a $G/summary.txt
    done
done
# G2: 60 steps, dropout on, same seed/order; PR-vs-PR control for each combo.
B="$PY $S/bench.py --items train --steps 60 --eval"
for g in "A --mb 2 --acc 16" "B --mb 8 --acc 8"; do set -- $g; n=$1; shift
    for mode in full lora; do
        job g2_pr_${mode}_$n   $B --tree $BASE --mode $mode "$@" --out $RES/g2/pr_${mode}_$n.json
        job g2_pr2_${mode}_$n  $B --tree $BASE --mode $mode "$@" --out $RES/g2/pr2_${mode}_$n.json
        job g2_perf_${mode}_$n $B --tree $PERF --mode $mode "$@" --out $RES/g2/perf_${mode}_$n.json
        for cmp in "pr pr2" "pr perf"; do set -- $cmp
            echo "G2 $mode $n $1 vs $2 $($PY $S/g2compare.py $RES/g2/$1_${mode}_$n.json $RES/g2/$2_${mode}_$n.json)" | tee -a $RES/g2/summary.txt
        done
        set -- $g; shift
    done
done
echo LANE7_G_DONE
