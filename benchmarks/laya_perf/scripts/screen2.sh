source /e/laya-perf/env.sh; cd /e/laya-perf; export AB_OUT=E:/laya-perf/out/ab
AB="python scripts/run_ab.py"
FA="--tree $BASE --mode full --mb 2 --acc 16 --steps 24"
FB="--tree $BASE --mode full --mb 8 --acc 8 --steps 16"
LA="--tree $BASE --mode lora --mb 2 --acc 16 --steps 24"
LB="--tree $BASE --mode lora --mb 8 --acc 8 --steps 16"
$AB s2_c13_full_A "$FA" "" "--opts rope_once" 3
$AB s2_c4onf_full_A "$FA" "--optim adamw_torch_fused" "--optim adamw_torch_fused --opts compile" 3
$AB s2_c4onf_full_B "$FB" "--optim adamw_torch_fused" "--optim adamw_torch_fused --opts compile" 3
$AB s2_c11f_full_B "$FB" "" "--optim adamw_torch_fused" 3
$AB s2_c7_lora_B "$LB" "" "--opts lora_bf16" 3
$AB s2_c7_lora_A "$LA" "" "--opts lora_bf16" 3
$AB s2_c4l_lora_A "$LA" "--opts lora_bf16" "--opts lora_bf16,compile" 2
echo SCREEN2DONE
