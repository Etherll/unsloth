source /e/laya-perf/env.sh; cd /e/laya-perf; mkdir -p out/epoch2
B="python scripts/bench.py --items train --epochs 2 --eval --mb 2 --acc 16 --warmup 4"
$B --tree $BASE --mode full --out out/epoch2/base_full.json > out/epoch2/base_full.log 2>&1
$B --tree $PERF --mode full --opts compile_always --out out/epoch2/perf_full.json > out/epoch2/perf_full.log 2>&1
python scripts/g2compare.py out/epoch2/base_full.json out/epoch2/perf_full.json | sed "s/^/epoch2 full /"
$B --tree $BASE --mode lora --out out/epoch2/base_lora.json > out/epoch2/base_lora.log 2>&1
$B --tree $PERF --mode lora --out out/epoch2/perf_lora.json > out/epoch2/perf_lora.log 2>&1
python scripts/g2compare.py out/epoch2/base_lora.json out/epoch2/perf_lora.json | sed "s/^/epoch2 lora /"
echo E2DONE
