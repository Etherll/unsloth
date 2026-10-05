source /e/laya-perf/env.sh; cd /e/laya-perf; export AB_OUT=E:/laya-perf/out/stack
AB="python scripts/run_ab.py"
$AB full_A "--mode full --mb 2 --acc 16 --steps 24" "--tree $BASE" "--tree $PERF --opts compile_always" 3
$AB full_B "--mode full --mb 8 --acc 8 --steps 16" "--tree $BASE" "--tree $PERF --opts compile_always" 3
$AB lora_A "--mode lora --mb 2 --acc 16 --steps 24" "--tree $BASE" "--tree $PERF" 3
$AB lora_B "--mode lora --mb 8 --acc 8 --steps 16" "--tree $BASE" "--tree $PERF" 3
echo STACKDONE
