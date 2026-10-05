source /mnt/e/laya-perf/env-wsl.sh; cd $LP_ROOT; export AB_OUT=$OUT/stackw
AB="$PY $S/run_ab.py"
$AB full_B "--mode full --mb 8 --acc 8 --steps 16" "--tree $BASE" "--tree $PERF --opts compile_always" 3 | cut -c1-300
$AB full_A "--mode full --mb 2 --acc 16 --steps 24" "--tree $BASE" "--tree $PERF --opts compile_always" 3 | cut -c1-300
$AB lora_B "--mode lora --mb 8 --acc 8 --steps 16" "--tree $BASE" "--tree $PERF" 3 | cut -c1-300
$AB lora_A "--mode lora --mb 2 --acc 16 --steps 24" "--tree $BASE" "--tree $PERF" 3 | cut -c1-300
P="$PY $S/parity.py"
$P --tree $BASE --mode full --precision bf16 --out $OUT/g1/wsl_base_full_bf16.json > /dev/null 2>&1
$P --tree $BASE --mode full --precision bf16 --opts unpad_flash --out $OUT/g1/wsl_unpad_flash_bf16.json > /dev/null 2>&1
$PY $S/pcompare.py $OUT/g1/wsl_base_full_bf16.json $OUT/g1/wsl_unpad_flash_bf16.json | sed 's/^/G1 wsl unpad_flash bf16 /'
$P --tree $PERF --mode full --precision bf16 --opts compile --out $OUT/g1/wsl_perf_full_bf16.json > /dev/null 2>&1
$PY $S/pcompare.py $OUT/g1/wsl_base_full_bf16.json $OUT/g1/wsl_perf_full_bf16.json | sed 's/^/G1 wsl perf full bf16 /'
$PY $S/bench.py --tree $PERF --mode full --mb 8 --acc 8 --steps 9 --warmup 2 --opts compile_always --prof $OUT/prof/wsl_final_full_B --out $OUT/prof/wsl_final_full_B.json > /dev/null 2>&1
$PY $S/bench.py --tree $PERF --mode full --mb 2 --acc 16 --steps 9 --warmup 2 --opts compile_always --prof $OUT/prof/wsl_final_full_A --out $OUT/prof/wsl_final_full_A.json > /dev/null 2>&1
for d in wsl_final_full_B wsl_final_full_A; do $PY $S/prof.py $OUT/prof/$d/trace.json > /dev/null; done
cd $PERF && $PY -m pytest tests/test_decision_model.py -q -p no:cacheprovider 2>&1 | tail -3
echo B4LDONE
