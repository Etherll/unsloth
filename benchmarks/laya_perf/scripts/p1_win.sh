source /e/laya-perf/env.sh; cd /e/laya-perf
B="python scripts/bench.py --tree $BASE"
$B --mode full --mb 8 --acc 8 --steps 20 --out out/p1/base_full_B.json
$B --mode lora --mb 2 --acc 16 --steps 40 --out out/p1/base_lora_A.json
$B --mode lora --mb 8 --acc 8 --steps 20 --out out/p1/base_lora_B.json
$B --mode full --mb 2 --acc 16 --steps 40 --out out/p1/base_full_A3.json
$B --mode full --mb 8 --acc 8 --steps 9 --warmup 2 --prof out/prof/win_base_full_B --out out/prof/win_base_full_B.json
$B --mode lora --mb 2 --acc 16 --steps 9 --warmup 2 --prof out/prof/win_base_lora_A --out out/prof/win_base_lora_A.json
$B --mode lora --mb 8 --acc 8 --steps 9 --warmup 2 --prof out/prof/win_base_lora_B --out out/prof/win_base_lora_B.json
$B --mode full --mb 8 --acc 8 --steps 5 --warmup 1 --memsnap out/prof/win_base_full_B_mem.pickle --out out/prof/mem_full_B.json
for d in win_base_full_B win_base_lora_A win_base_lora_B; do python scripts/prof.py out/prof/$d/trace.json > /dev/null; done
echo P1DONE
