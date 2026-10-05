source /e/laya-perf/env.sh; cd /e/laya-perf; export AB_OUT=E:/laya-perf/out/ab
AB="python scripts/run_ab.py"
FA="--tree $BASE --mode full --mb 2 --acc 16 --steps 24"
FB="--tree $BASE --mode full --mb 8 --acc 8 --steps 16"
LA="--tree $BASE --mode lora --mb 2 --acc 16 --steps 24"
LB="--tree $BASE --mode lora --mb 8 --acc 8 --steps 16"
$AB c4_full_A "$FA" "" "--opts compile" 2
$AB c11f_full_B "$FB" "" "--optim adamw_torch_fused" 2
$AB c11f_full_A "$FA" "" "--optim adamw_torch_fused" 2
$AB c4f_full_B "$FB" "" "--opts compile --optim adamw_torch_fused" 2
$AB c7_lora_B "$LB" "" "--opts lora_bf16" 2
$AB c7_lora_A "$LA" "" "--opts lora_bf16" 2
$AB c4_lora_B "$LB" "--opts lora_bf16" "--opts lora_bf16,compile" 2
$AB c1_full_B "$FB" "" "--opts cudnn_guard" 2
$AB c2p8_full_B "$FB" "" "--opts pad8" 2
$AB c2p64_full_B "$FB" "" "--opts pad64" 2
$AB c3sdpa_full_B "$FB" "" "--opts unpad_sdpa" 2
echo SCREEN1DONE
