# GPU7: which perf change makes LoRA (mb2 x acc16, lr 8e-4, no warmup) collapse on the B200?
# 60 steps per run: the collapse shows by step ~20 (train loss stuck near 1.20, grad norm ~0.3).
source $(dirname $0)/../env-linux.sh; lane 7; source $S/lanelib.sh; cd $LP_ROOT
D=$RES/lora_ablate; mkdir -p $D
B="$PY $S/bench.py --items train --steps 60 --warmup 4 --mode lora --mb 2 --acc 16"
for seed in 3409 3408; do
    job la_perf_$seed       $B --tree $PERF --seed $seed --out $D/perf_$seed.json
    job la_perf_cudnn_$seed $B --tree $PERF --seed $seed --opts allow_cudnn --out $D/perf_cudnn_$seed.json
    job la_perf_feach_$seed $B --tree $PERF --seed $seed --opts foreach_adam --out $D/perf_foreach_$seed.json
    job la_perf_peft_$seed  $B --tree $PERF --seed $seed --opts peft_forward --out $D/perf_peft_$seed.json
    job la_pr_$seed         $B --tree $BASE --seed $seed --out $D/pr_$seed.json
    job la_pr_c1_$seed      $B --tree $BASE --seed $seed --opts cudnn_guard --out $D/pr_c1_$seed.json
    job la_pr_fused_$seed   $B --tree $BASE --seed $seed --optim adamw_torch_fused --out $D/pr_fused_$seed.json
done
$PY - <<'PY' | tee $D/summary.txt
import glob, json, os
for f in sorted(glob.glob(os.environ["RES"] + "/lora_ablate/*.json")):
    r = json.load(open(f)); l = r["loss"]; g = r["grad_norm"]
    w = lambda a, b: sum(l[a:b]) / len(l[a:b])
    print(f"{os.path.basename(f):24s} loss 20-40 {w(20, 40):.3f} 40-60 {w(40, 60):.3f} gn 20-60 {sum(g[20:60]) / 40:.2f}",
          "COLLAPSED" if w(40, 60) > 1.15 and sum(g[20:60]) / 40 < 0.4 else "ok")
PY
