# GPU7: G1 (fp32 + bf16) for a candidate tree (TREE, TAG), through the trainer's own compile path.
source $(dirname $0)/../env-linux.sh; lane 7; source $S/lanelib.sh; cd $LP_ROOT
G=$RES/g1; P="$PY $S/parity.py"; T=${TREE:-$LP_ROOT/cand}; t=${TAG:-cand}
for m in ${MODES:-full}; do
    job g1_${t}_${m}_fp32 $P --tree $T --mode $m --precision fp32 --gc true --opts trainer_compile --out $G/${t}_${m}_fp32.json
    job g1_${t}_${m}_bf16 $P --tree $T --mode $m --precision bf16 --opts trainer_compile --out $G/${t}_${m}_bf16.json
    for c in "pr_${m}_fp32 ${t}_${m}_fp32" "perf_${m}_fp32 ${t}_${m}_fp32" "pr_${m}_bf16 ${t}_${m}_bf16" "pr_${m}_fp32 ${t}_${m}_bf16"; do set -- $c
        echo "G1 $1 vs $2 $($PY $S/pcompare.py $G/$1.json $G/$2.json 2>/dev/null | tail -1)" | tee -a $G/summary.txt
    done
done
