# WSL: source before every Linux run. Caches live on the E:-backed WSL disk (E:\WSL\Ubuntu).
export LP_ROOT=$HOME/laya-perf
export PY=$HOME/pr-4460-20260922/environments/python/bin/python
export TMPDIR=$LP_ROOT/tmp TMP=$LP_ROOT/tmp TEMP=$LP_ROOT/tmp
export HF_HOME=$LP_ROOT/cache/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=$LP_ROOT/cache/inductor TRITON_CACHE_DIR=$LP_ROOT/cache/triton CUDA_CACHE_PATH=$LP_ROOT/cache/nv
export UV_CACHE_DIR=$LP_ROOT/cache/uv PIP_CACHE_DIR=$LP_ROOT/cache/pip XDG_CACHE_HOME=$LP_ROOT/cache PYTHONPYCACHEPREFIX=$LP_ROOT/cache/pyc TORCH_HOME=$LP_ROOT/cache/torch
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false UNSLOTH_DISABLE_STATISTICS=1
export BASE=/mnt/e/orca/workspaces/unsloth/studio-fine-tune-laya-decision-models-and-serve PERF=/mnt/e/orca/workspaces/unsloth/laya-decision-perf
export S=/mnt/e/laya-perf/scripts OUT=/mnt/e/laya-perf/out
mkdir -p $LP_ROOT/{tmp,cache,data,model} 
# Windows PATH entries (an unreadable nvcc under /mnt/c) break inductor's repro/system-info probes.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)
