$LP = 'E:\laya-perf'
$env:TEMP = "$LP\tmp"; $env:TMP = $env:TEMP; $env:TMPDIR = $env:TEMP
$env:HF_HOME = 'E:\pr-4460-20260922\cache\huggingface'; $env:HF_HUB_OFFLINE = '1'; $env:HF_DATASETS_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
$env:TORCHINDUCTOR_CACHE_DIR = "$LP\cache\inductor"; $env:TRITON_CACHE_DIR = "$LP\cache\triton"; $env:CUDA_CACHE_PATH = "$LP\cache\nv"
$env:UV_CACHE_DIR = "$LP\cache\uv"; $env:PIP_CACHE_DIR = "$LP\cache\pip"; $env:XDG_CACHE_HOME = "$LP\cache"; $env:PYTHONPYCACHEPREFIX = "$LP\cache\pyc"; $env:TORCH_HOME = "$LP\cache\torch"
$env:HF_HUB_DISABLE_TELEMETRY = '1'; $env:DO_NOT_TRACK = '1'; $env:TOKENIZERS_PARALLELISM = 'false'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUTF8 = '1'
$env:UNSLOTH_DISABLE_STATISTICS = '1'
