# GPU7: G1 (fp32 + bf16) for discovery candidates, queued beside the long runs (flock keeps one job per GPU).
source $(dirname $0)/../env-linux.sh; lane 7; source $S/lanelib.sh; cd $LP_ROOT
G=$RES/g1; P="$PY $S/parity.py"
for o in ${CANDS:-static64,cpp_wrapper}; do t=$(echo $o | tr , _)
    job g1_${t}_full_fp32 $P --tree $PERF --mode full --precision fp32 --gc true --opts $o,trainer_compile --out $G/${t}_full_fp32.json
    job g1_${t}_full_bf16 $P --tree $PERF --mode full --precision bf16 --opts $o,trainer_compile --out $G/${t}_full_bf16.json
    for c in "pr_full_fp32 ${t}_full_fp32" "perf_full_fp32 ${t}_full_fp32" "pr_full_bf16 ${t}_full_bf16" "pr_full_fp32 ${t}_full_bf16"; do set -- $c
        echo "G1 $1 vs $2 $($PY $S/pcompare.py $G/$1.json $G/$2.json 2>/dev/null | tail -1)" | tee -a $G/summary.txt
    done
done
