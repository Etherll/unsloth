source /mnt/e/laya-perf/env-wsl.sh
cd $LP_ROOT
mkdir -p $OUT/p1w $OUT/prof
B="$PY $S/bench.py --tree $BASE"
$B --mode full --mb 2 --acc 16 --steps 24 --out $OUT/p1w/base_full_A.json 2>&1 | tail -3
$B --mode full --mb 8 --acc 8 --steps 20 --out $OUT/p1w/base_full_B.json 2>&1 | tail -1
$B --mode lora --mb 2 --acc 16 --steps 24 --out $OUT/p1w/base_lora_A.json 2>&1 | tail -1
$B --mode lora --mb 8 --acc 8 --steps 20 --out $OUT/p1w/base_lora_B.json 2>&1 | tail -1
$B --mode full --mb 2 --acc 16 --steps 9 --warmup 2 --prof $OUT/prof/wsl_base_full_A --out $OUT/prof/wsl_base_full_A.json 2>&1 | tail -1
$B --mode full --mb 8 --acc 8 --steps 9 --warmup 2 --prof $OUT/prof/wsl_base_full_B --out $OUT/prof/wsl_base_full_B.json 2>&1 | tail -1
for d in wsl_base_full_A wsl_base_full_B; do $PY $S/prof.py $OUT/prof/$d/trace.json > /dev/null; done
echo P1WDONE
