# GPU6: screen a candidate tree against the current perf tree (single runs, load-gated).
source $(dirname $0)/../env-linux.sh; lane 6; source $S/lanelib.sh; cd $LP_ROOT; export LP_JOB_LOAD_MAX=40
SC=$RES/screen; T=${TREE:-$LP_ROOT/cand}; t=${TAG:-cand}
job scr_${t}_full_A $PY $S/bench.py --steps 24 --warmup 6 --tree $T --mode full --mb 2 --acc 16 --out $SC/${t}_full_A.json
job scr_${t}_full_B $PY $S/bench.py --steps 16 --warmup 6 --tree $T --mode full --mb 8 --acc 8 --out $SC/${t}_full_B.json
