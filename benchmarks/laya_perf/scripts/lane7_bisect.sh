# GPU7: bisect cand's bf16 batch-1 outlier: repeat, cpp_wrapper off, cpp_wrapper global.
source $(dirname $0)/../env-linux.sh; lane 7; source $S/lanelib.sh; cd $LP_ROOT
G=$RES/g1; P="$PY $S/parity.py --tree $LP_ROOT/cand --mode full --precision bf16 --opts trainer_compile"
job bis_rep $P --out $G/bis_rep_bf16.json
LP_CAND_CPP=off job bis_off $P --out $G/bis_off_bf16.json
LP_CAND_CPP=global job bis_global $P --out $G/bis_global_bf16.json
