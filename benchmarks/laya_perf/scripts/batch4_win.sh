source /e/laya-perf/env.sh; cd /e/laya-perf; export AB_OUT=E:/laya-perf/out/stack2
python scripts/run_ab.py full_A "--mode full --mb 2 --acc 16 --steps 24" "--tree $BASE" "--tree $PERF --opts compile_always" 3 | cut -c1-300
python scripts/run_ab.py full_B "--mode full --mb 8 --acc 8 --steps 16" "--tree $BASE" "--tree $PERF --opts compile_always" 3 | cut -c1-300
echo B4WDONE
