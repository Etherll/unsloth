source /e/laya-perf/env.sh; cd /e/laya-perf; mkdir -p out/g1
P="python scripts/parity.py --tree $BASE"
for prec in bf16 fp32; do
  $P --mode full --precision $prec --out out/g1/base_full_$prec.json > /dev/null 2>&1
  for o in compile cudnn_guard rope_once; do $P --mode full --precision $prec --opts $o --out out/g1/${o}_full_$prec.json > out/g1/${o}_full_$prec.log 2>&1; python scripts/pcompare.py out/g1/base_full_$prec.json out/g1/${o}_full_$prec.json | sed "s/^/$o full $prec /"; done
  $P --mode lora --precision $prec --out out/g1/base_lora_$prec.json > /dev/null 2>&1
done
$P --mode lora --precision bf16 --opts lora_bf16 --out out/g1/lora_bf16_lora_bf16.json > out/g1/lora_bf16.log 2>&1; python scripts/pcompare.py out/g1/base_lora_bf16.json out/g1/lora_bf16_lora_bf16.json | sed "s/^/lora_bf16 lora bf16 /"
$P --mode lora --precision bf16 --opts compile --out out/g1/compile_lora_bf16.json > /dev/null 2>&1; python scripts/pcompare.py out/g1/base_lora_bf16.json out/g1/compile_lora_bf16.json | sed "s/^/compile lora bf16 /"
$P --mode full --precision bf16 --out out/g1/base_full_bf16_rerun.json > /dev/null 2>&1; python scripts/pcompare.py out/g1/base_full_bf16.json out/g1/base_full_bf16_rerun.json | sed "s/^/noise full bf16 /"
echo G1DONE
