# Linux/AWS (shared 8x B200): `source env-linux.sh; lane 5|6|7` before every run, then prefix jobs with `pin`.
# Everything lives on the large volume. Only GPUs 5-7 are ours; each lane has its own GPU, cores and caches.
export LP_ROOT=/mnt/disks/unslothai/etherl/laya-perf
export PY=$LP_ROOT/venv/bin/python PYX=$LP_ROOT/venv-xf/bin/python
export BASE=$LP_ROOT/base PERF=$LP_ROOT/perf
export S=$LP_ROOT/bench/benchmarks/laya_perf/scripts R=$LP_ROOT/bench/benchmarks/laya_perf/results/linux
export HF_HOME=/mnt/disks/unslothai/etherl/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR=/mnt/disks/unslothai/etherl/.cache/uv_cache PIP_CACHE_DIR=$LP_ROOT/cache/pip
export XDG_CACHE_HOME=$LP_ROOT/cache PYTHONPYCACHEPREFIX=$LP_ROOT/cache/pyc TORCH_HOME=$LP_ROOT/cache/torch
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false UNSLOTH_DISABLE_STATISTICS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONWARNINGS=ignore
export PATH=/usr/local/cuda-13.1/bin:$PATH

# GPUs 5-7 sit on NUMA node 1 (CPUs 48-95 + HT siblings 144-191). Timing gets the largest block.
lane() {
    export LANE=$1 CUDA_VISIBLE_DEVICES=$1
    case $1 in
        5) export LP_CPUS=48-71,144-167 ;;
        6) export LP_CPUS=72-83,168-179 TORCHINDUCTOR_COMPILE_THREADS=4 ;;
        7) export LP_CPUS=84-95,180-191 ;;
        *) echo "lane: only GPUs 5, 6, 7" >&2; return 1 ;;
    esac
    # Unsloth pins the inductor cache to $TMPDIR/torchinductor_<user>, so TMPDIR is per lane.
    export TMPDIR=$LP_ROOT/tmp/gpu$1 TMP=$LP_ROOT/tmp/gpu$1 TEMP=$LP_ROOT/tmp/gpu$1
    export TRITON_CACHE_DIR=$LP_ROOT/cache/gpu$1/triton CUDA_CACHE_PATH=$LP_ROOT/cache/gpu$1/nv
    export UNSLOTH_MEGA_CACHE_DIR=$LP_ROOT/cache/gpu$1/mega UNSLOTH_COMPILE_LOCATION=$LP_ROOT/cache/gpu$1/unsloth_compiled_cache
    export OUT=$LP_ROOT/out/gpu$1 RES=$R/gpu$1 AB_OUT=$R/gpu$1/ab
    mkdir -p $TMPDIR $TRITON_CACHE_DIR $CUDA_CACHE_PATH $UNSLOTH_MEGA_CACHE_DIR $OUT $RES $AB_OUT
}
pin() { taskset -c "$LP_CPUS" nice -n 0 "$@"; }
mkdir -p $LP_ROOT/{tmp,cache,data,model,out,logs}
