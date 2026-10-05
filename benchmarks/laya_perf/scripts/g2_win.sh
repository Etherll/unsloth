source /e/laya-perf/env.sh; cd /e/laya-perf; mkdir -p out/g2
B="python scripts/bench.py --items train --steps 60 --eval"
for g in "A --mb 2 --acc 16" "B --mb 8 --acc 8"; do set -- $g; n=$1; shift
  $B --tree $BASE --mode full "$@" --out out/g2/base_full_$n.json > /dev/null 2>&1
  $B --tree $PERF --mode full "$@" --opts compile_always --out out/g2/perf_full_$n.json > /dev/null 2>&1
  python scripts/g2compare.py out/g2/base_full_$n.json out/g2/perf_full_$n.json | sed "s/^/full $n /"
  $B --tree $BASE --mode lora "$@" --out out/g2/base_lora_$n.json > /dev/null 2>&1
  $B --tree $PERF --mode lora "$@" --out out/g2/perf_lora_$n.json > /dev/null 2>&1
  python scripts/g2compare.py out/g2/base_lora_$n.json out/g2/perf_lora_$n.json | sed "s/^/lora $n /"
done
echo G2DONE
