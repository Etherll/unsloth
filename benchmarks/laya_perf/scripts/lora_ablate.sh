source /e/laya-perf/env.sh; cd /e/laya-perf; mkdir -p out/ablate
B="python scripts/bench.py --items train --epochs 2 --mb 2 --acc 16 --mode lora --stop-at 150"
$B --tree $BASE --out out/ablate/base_r2.json > out/ablate/base_r2.log 2>&1
$B --tree $BASE --opts lean_lora2 --out out/ablate/lean.json > out/ablate/lean.log 2>&1
$B --tree $BASE --optim adamw_torch_fused --out out/ablate/fused.json > out/ablate/fused.log 2>&1
$B --tree $PERF --out out/ablate/perf_r2.json > out/ablate/perf_r2.log 2>&1
echo ABLDONE
