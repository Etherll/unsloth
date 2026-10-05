# Git Bash on Windows: source before every Windows run. WSL uses env-wsl.sh.
export LP=E:/laya-perf
export TEMP=$LP/tmp TMP=$LP/tmp TMPDIR=$LP/tmp
export HF_HOME=E:/pr-4460-20260922/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=$LP/cache/inductor TRITON_CACHE_DIR=$LP/cache/triton CUDA_CACHE_PATH=$LP/cache/nv
export UV_CACHE_DIR=$LP/cache/uv PIP_CACHE_DIR=$LP/cache/pip XDG_CACHE_HOME=$LP/cache PYTHONPYCACHEPREFIX=$LP/cache/pyc TORCH_HOME=$LP/cache/torch
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export UNSLOTH_DISABLE_STATISTICS=1
export BASE=E:/orca/workspaces/unsloth/studio-fine-tune-laya-decision-models-and-serve PERF=E:/orca/workspaces/unsloth/laya-decision-perf
