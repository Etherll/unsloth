# GPU7: LONG-RUN CHECKS on the current stack: 200 optimizer steps, holdout eval at 60/120/200.
# Full FT: PR, PR control, perf. LoRA: 3 seeds per arm. Geometry B: full + LoRA, PR vs perf.
source $(dirname $0)/../env-linux.sh; lane 7; source $S/lanelib.sh; cd $LP_ROOT
L=$RES/long; mkdir -p $L
# G1 bf16 split for full FT: which part moves perf's bf16 grads (attention backend vs compile)?
G=$RES/g1
if [ ! -f $G/c1_full_bf16.json ]; then
    job g1_c1_full_bf16 $PY $S/parity.py --tree $BASE --mode full --precision bf16 --opts cudnn_guard --out $G/c1_full_bf16.json
    UNSLOTH_COMPILE_DISABLE=1 job g1_perfnc_full_bf16 $PY $S/parity.py --tree $PERF --mode full --precision bf16 --out $G/perfnc_full_bf16.json
    job g1_perf2_full_bf16 $PY $S/parity.py --tree $PERF --mode full --precision bf16 --opts trainer_compile --out $G/perf2_full_bf16.json
    for cmp in "pr_full_fp32 c1_full_bf16" "pr_full_fp32 perfnc_full_bf16" "pr_full_fp32 perf2_full_bf16" "perf_full_bf16 perf2_full_bf16" "c1_full_bf16 perf_full_bf16"; do
        set -- $cmp; echo "G1 $1 vs $2 $($PY $S/pcompare.py $G/$1.json $G/$2.json 2>/dev/null | tail -1)" | tee -a $G/summary.txt
    done
fi
STEPS=${STEPS:-200}; EV=${EV:-60,120,200}; TAG=${TAG:-s$STEPS}
B="$PY $S/bench.py --items train --steps $STEPS --eval-at $EV --warmup 4"
A="--mb 2 --acc 16"
job long_${TAG}_pr_full_A   $B --tree $BASE --mode full $A --out $L/${TAG}_pr_full_A.json
job long_${TAG}_perf_full_A $B --tree $PERF --mode full $A --out $L/${TAG}_perf_full_A.json
job long_${TAG}_pr2_full_A  $B --tree $BASE --mode full $A --out $L/${TAG}_pr2_full_A.json
for seed in 3407 3408 3409; do
    job long_${TAG}_pr_lora_A_$seed   $B --tree $BASE --mode lora $A --seed $seed --out $L/${TAG}_pr_lora_A_$seed.json
    job long_${TAG}_perf_lora_A_$seed $B --tree $PERF --mode lora $A --seed $seed --out $L/${TAG}_perf_lora_A_$seed.json
done
if [ -z "$SKIP_B" ]; then
    for mode in full lora; do
        job long_${TAG}_pr_${mode}_B   $B --tree $BASE --mode $mode --mb 8 --acc 8 --out $L/${TAG}_pr_${mode}_B.json
        job long_${TAG}_perf_${mode}_B $B --tree $PERF --mode $mode --mb 8 --acc 8 --out $L/${TAG}_perf_${mode}_B.json
    done
fi
echo LANE7_LONG_DONE
