source /e/laya-perf/env.sh; cd /e/laya-perf
E="python scripts/bench.py --items train --epochs 2 --eval --mb 2 --acc 16 --warmup 4 --mode lora"
for i in 3 4 5; do $E --tree $BASE --out out/epoch2/base${i}_lora.json > out/epoch2/base${i}_lora.log 2>&1; done
$E --tree $PERF --out out/epoch2/perf5_lora.json > out/epoch2/perf5_lora.log 2>&1
echo B5DONE
