source /mnt/e/laya-perf/env-wsl.sh
cd $LP_ROOT; export AB_OUT=$OUT/abw
AB="$PY $S/run_ab.py"
FA="--tree $BASE --mode full --mb 2 --acc 16 --steps 24"
FB="--tree $BASE --mode full --mb 8 --acc 8 --steps 16"
$AB w_c3a_full_B "$FB" "" "--opts fa2" 2
$AB w_c3b_full_B "$FB" "" "--opts unpad_flash" 2
$AB w_c3b_full_A "$FA" "" "--opts unpad_flash" 2
$AB w_c4_full_B "$FB" "" "--opts compile" 2
$AB w_c4_full_A "$FA" "" "--opts compile" 2
$AB w_c13_full_A "$FA" "" "--opts rope_once" 2
echo WSLSCREENDONE
