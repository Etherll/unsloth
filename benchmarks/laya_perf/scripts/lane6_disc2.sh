# GPU6 discovery round 2: whole-encoder graph (compile / CUDA graphs), LoRA compile (Q6), attention backends.
source $(dirname $0)/../env-linux.sh; lane 6; source $S/lanelib.sh; cd $LP_ROOT; export LP_JOB_LOAD_MAX=40
SC=$RES/screen; mkdir -p $SC $RES/attn
A="--mb 2 --acc 16"; B="--mb 8 --acc 8"
SA="$PY $S/bench.py --steps 24 --warmup 6"; SBB="$PY $S/bench.py --steps 16 --warmup 6"
job scr_enc_compile_full_A $SA --tree $PERF --mode full $A --opts enc_compile --out $SC/enc_compile_full_A.json
job scr_enc_cg_full_A      $SA --tree $PERF --mode full $A --opts enc_cg --out $SC/enc_cg_full_A.json
job scr_perf_lora_A        $SA --tree $PERF --mode lora $A --out $SC/perf_lora_A.json
job scr_lcomp_lora_A       $SA --tree $PERF --mode lora $A --opts lora_compile --out $SC/lcomp_lora_A.json
job scr_lstat_lora_A       $SA --tree $PERF --mode lora $A --opts lora_compile,static64,cpp_wrapper --out $SC/lstat_lora_A.json
job scr_perf_lora_B        $SBB --tree $PERF --mode lora $B --out $SC/perf_lora_B.json
job scr_lcomp_lora_B       $SBB --tree $PERF --mode lora $B --opts lora_compile --out $SC/lcomp_lora_B.json
job scr_lstat_lora_B       $SBB --tree $PERF --mode lora $B --opts lora_compile,static64,cpp_wrapper --out $SC/lstat_lora_B.json
job scr_enc_cg_full_B      $SBB --tree $PERF --mode full $B --opts enc_cg --out $SC/enc_cg_full_B.json
# Attention backends (xformers venv: torch 2.13 + xformers 0.0.35 + fbgemm genai 1.8.0).
job attn_mb2 $PYX $S/attn_bench.py --mb 2 --out $RES/attn/attn_mb2.json
job attn_mb8 $PYX $S/attn_bench.py --mb 8 --out $RES/attn/attn_mb8.json
job scr_varlen_full_B $PYX $S/bench.py --steps 16 --warmup 6 --tree $PERF --mode full $B --opts unpad_varlen --out $SC/varlen_full_B.json
job scr_xf_full_B     $PYX $S/bench.py --steps 16 --warmup 6 --tree $PERF --mode full $B --opts unpad_xf --out $SC/xf_full_B.json
job scr_xfbase_full_B $PYX $S/bench.py --steps 16 --warmup 6 --tree $PERF --mode full $B --out $SC/xfbase_full_B.json
echo LANE6_DISC2_DONE
