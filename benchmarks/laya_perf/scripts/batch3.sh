source /e/laya-perf/env.sh; cd /e/laya-perf
AB_OUT=E:/laya-perf/out/stack python scripts/run_ab.py lora_A "--mode lora --mb 2 --acc 16 --steps 24" "--tree $BASE" "--tree $PERF" 3 | cut -c1-300
AB_OUT=E:/laya-perf/out/stack python scripts/run_ab.py lora_B "--mode lora --mb 8 --acc 8 --steps 16" "--tree $BASE" "--tree $PERF" 3 | cut -c1-300
B="python scripts/bench.py --items train --steps 60 --eval"
for g in "A --mb 2 --acc 16" "B --mb 8 --acc 8"; do set -- $g; n=$1; shift
  $B --tree $PERF --mode lora "$@" --out out/g2/perf3_lora_$n.json > /dev/null 2>&1
  python scripts/g2compare.py out/g2/base_lora_$n.json out/g2/perf3_lora_$n.json | sed "s/^/G2 lora v3 $n /"
done
E="python scripts/bench.py --items train --epochs 2 --eval --mb 2 --acc 16 --warmup 4 --mode lora"
$E --tree $PERF --out out/epoch2/perf3_lora.json > out/epoch2/perf3_lora.log 2>&1
python scripts/g2compare.py out/epoch2/base_lora.json out/epoch2/perf3_lora.json | sed 's/^/epoch2 lora v3 /'
$E --tree $BASE --out out/epoch2/base2_lora.json > out/epoch2/base2_lora.log 2>&1
python scripts/g2compare.py out/epoch2/base_lora.json out/epoch2/base2_lora.json | sed 's/^/noise epoch2 lora /'
$E --tree $PERF --out out/epoch2/perf4_lora.json > out/epoch2/perf4_lora.log 2>&1
python scripts/g2compare.py out/epoch2/base_lora.json out/epoch2/perf4_lora.json | sed 's/^/epoch2 lora v3 run2 /'
echo B3DONE
bash scripts/final_win.sh
