source /e/laya-perf/env.sh; cd /e/laya-perf; export AB_OUT=E:/laya-perf/out/ab
AB="python scripts/run_ab.py"
# C10: GC modes on the final build (report only)
$AB c10_off_full_B "--tree $PERF --mode full --mb 8 --acc 8 --steps 16" "--opts compile_always" "--opts compile_always --gc off" 2
$AB c10_true_full_B "--tree $PERF --mode full --mb 8 --acc 8 --steps 16" "--opts compile_always" "--opts compile_always --gc true" 2
$AB c10_off_full_A "--tree $PERF --mode full --mb 2 --acc 16 --steps 24" "--opts compile_always" "--opts compile_always --gc off" 2
$AB c10_off_lora_B "--tree $PERF --mode lora --mb 8 --acc 8 --steps 16" "" "--gc off" 2
# C11: 8-bit AdamW (report only)
$AB c11_8bit_full_B "--tree $PERF --mode full --mb 8 --acc 8 --steps 16" "--opts compile_always" "--opts compile_always --optim adamw_8bit" 2
# C3d: flex_attention unpadding
LP_STRICT=0 $AB c3flex_full_B "--tree $BASE --mode full --mb 8 --acc 8 --steps 16" "" "--opts unpad_flex" 1
# G1 fp32 + bf16 for the stacked build
P="python scripts/parity.py"
$P --tree $PERF --mode full --precision fp32 --gc true --opts compile --out out/g1/perf_full_fp32.json > /dev/null 2>&1; python scripts/pcompare.py out/g1/base_full_fp32.json out/g1/perf_full_fp32.json | sed 's/^/G1 perf full fp32 /'
$P --tree $PERF --mode full --precision bf16 --opts compile --out out/g1/perf_full_bf16.json > /dev/null 2>&1; python scripts/pcompare.py out/g1/base_full_bf16.json out/g1/perf_full_bf16.json | sed 's/^/G1 perf full bf16 vs PR /'; python scripts/pcompare.py out/g1/base_full_fp32.json out/g1/perf_full_bf16.json | sed 's/^/G1 perf full bf16 vs fp32ref /'
$P --tree $PERF --mode lora --precision fp32 --gc true --out out/g1/perf_lora_fp32.json > /dev/null 2>&1; python scripts/pcompare.py out/g1/base_lora_fp32.json out/g1/perf_lora_fp32.json | sed 's/^/G1 perf lora fp32 /'
$P --tree $PERF --mode lora --precision bf16 --out out/g1/perf_lora_bf16.json > /dev/null 2>&1; python scripts/pcompare.py out/g1/base_lora_bf16.json out/g1/perf_lora_bf16.json | sed 's/^/G1 perf lora bf16 /'
$P --tree $BASE --mode full --precision fp32 --gc true --opts cudnn_guard --out out/g1/cudnn_guard_full_fp32.json > /dev/null 2>&1; python scripts/pcompare.py out/g1/base_full_fp32.json out/g1/cudnn_guard_full_fp32.json | sed 's/^/G1 cudnn fp32 /'
$P --tree $BASE --mode full --precision fp32 --gc true --opts rope_once --out out/g1/rope_once_full_fp32.json > /dev/null 2>&1; python scripts/pcompare.py out/g1/base_full_fp32.json out/g1/rope_once_full_fp32.json | sed 's/^/G1 rope fp32 /'
# final profiles
python scripts/bench.py --tree $PERF --mode full --mb 2 --acc 16 --steps 9 --warmup 2 --opts compile_always --prof out/prof/win_final_full_A --out out/prof/win_final_full_A.json > /dev/null 2>&1
python scripts/bench.py --tree $PERF --mode full --mb 8 --acc 8 --steps 9 --warmup 2 --opts compile_always --prof out/prof/win_final_full_B --out out/prof/win_final_full_B.json > /dev/null 2>&1
python scripts/bench.py --tree $PERF --mode lora --mb 8 --acc 8 --steps 9 --warmup 2 --prof out/prof/win_final_lora_B --out out/prof/win_final_lora_B.json > /dev/null 2>&1
for d in win_final_full_A win_final_full_B win_final_lora_B; do python scripts/prof.py out/prof/$d/trace.json > /dev/null; done
echo FINALDONE
# noise floor: baseline vs baseline
python scripts/bench.py --tree $BASE --mode lora --mb 8 --acc 8 --items train --steps 60 --eval --out out/g2/base2_lora_B.json > /dev/null 2>&1
python scripts/g2compare.py out/g2/base_lora_B.json out/g2/base2_lora_B.json | sed 's/^/noise lora B /'
python scripts/bench.py --tree $BASE --mode full --items train --epochs 2 --eval --mb 2 --acc 16 --warmup 4 --out out/epoch2/base2_full.json > /dev/null 2>&1
python scripts/g2compare.py out/epoch2/base_full.json out/epoch2/base2_full.json | sed 's/^/noise epoch2 full /'
echo NOISEDONE
