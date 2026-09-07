"""Identical filename-level snapshot regression assertion on main and PR."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "studio/backend"))
os.environ.update(UNSLOTH_ALLOW_CPU="1", UNSLOTH_IS_PRESENT="1",
                  UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE="1", HF_HUB_OFFLINE="1")
from core.inference import local_model_resolver as resolver
from hub.utils.gguf import select_gguf_cache_snapshot

print("IMPLEMENTATION_SHA", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), flush=True)
with tempfile.TemporaryDirectory(prefix="pr10210-proof-") as temporary:
    cache = Path(temporary)
    repo_id = "unsloth/Qwen3.8-Flash-Next-GGUF"
    repo = cache / "models--unsloth--Qwen3.8-Flash-Next-GGUF"
    old = repo / "snapshots/weights-revision"
    quant = old / "UD-Q4_K_XL"
    quant.mkdir(parents=True)
    for shard in range(1, 5):
        (quant / f"Qwen3.8-Flash-Next-UD-Q4_K_XL-{shard:05d}-of-00004.gguf").write_bytes(b"GGUF stub")
    info = SimpleNamespace(id=repo_id, path=str(repo), source="hf_cache")
    control = resolver._local_gguf_entry(repo_id, info)
    assert control is not None and control.variants == ("UD-Q4_K_XL",), "CONTROL_SINGLE_SNAPSHOT_FAILED"
    print("PASS single-snapshot control", flush=True)
    newer = repo / "snapshots/companion-revision"
    (newer / "MTP").mkdir(parents=True)
    (newer / "MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf").write_bytes(b"GGUF companion")
    os.utime(old, (1000, 1000))
    os.utime(newer, (2000, 2000))
    assert Path(resolver._resolve_load_dir(repo)).resolve() == newer.resolve(), "CONTROL_NEWEST_SNAPSHOT_FAILED"
    selected = select_gguf_cache_snapshot(repo_id, root=cache)
    assert selected is not None and Path(selected[3]).resolve() == old.resolve(), "CONTROL_HUB_SELECTION_FAILED"
    print("PASS generic-newest and Hub-older-complete controls", flush=True)
    entry = resolver._local_gguf_entry(repo_id, info)
    assert entry is not None, "REGRESSION_API_DROPPED_OLDER_COMPLETE_WEIGHTS"
    assert Path(entry.load_path).resolve() == old.resolve()
    assert entry.variants == ("UD-Q4_K_XL",)
    assert resolver.local_servable_model(info) == (True, ("UD-Q4_K_XL",))
    print("PASS API retains older complete weights after newer companion-only snapshot", flush=True)
