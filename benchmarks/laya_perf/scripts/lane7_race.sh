# GPU7: reproduce the one-off bf16 outlier under GPU contention (our own stressor on our own GPU).
source $(dirname $0)/../env-linux.sh; lane 7; source $S/lanelib.sh; cd $LP_ROOT
G=$RES/g1/race; mkdir -p $G
P="$PY $S/parity.py --mode full --precision bf16"
stress_job() {  # name, then the parity command: runs it with the stressor alongside
    local name=$1; shift
    job $name bash -c "$PY $S/gpu_stress.py 900 & s=\$!; sleep 5; $*; rc=\$?; kill \$s; wait \$s 2>/dev/null; exit \$rc"
}
for i in 1 2 3; do
    stress_job race_cand_$i "$P --tree $LP_ROOT/cand --opts trainer_compile --out $G/cand_$i.json"
    stress_job race_perf_$i "$P --tree $PERF --opts trainer_compile --out $G/perf_$i.json"
done
for i in 1 2; do
    stress_job race_candoff_$i "LP_CAND_CPP=off $P --tree $LP_ROOT/cand --opts trainer_compile --out $G/candoff_$i.json"
done
$PY - <<'PY' | tee $G/summary.txt
import glob, json, os
ref = json.load(open(os.environ["RES"] + "/g1/bis_rep_bf16.json"))["loss"]
pr = json.load(open(os.environ["RES"] + "/g1/perf_full_bf16.json"))["loss"]
for f in sorted(glob.glob(os.environ["RES"] + "/g1/race/*.json")):
    l = json.load(open(f))["loss"]; r = pr if "perf" in f else ref
    print(os.path.basename(f), "max|dloss| vs clean", round(max(abs(a - b) for a, b in zip(l, r)), 4))
PY
