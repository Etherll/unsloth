"""Run bounded CPU suites; do not forgive setup errors or empty selections."""
import os
import subprocess
import sys

if os.environ.get("GITHUB_REF_NAME", "").endswith("-negative"):
    raise SystemExit("Negative branch unexpectedly reached positive-only suites")
files = ["test_anthropic_messages.py", "test_openai_auto_switch.py",
         "test_gguf_load_cache_reuse.py", "test_detect_mmproj_file.py",
         "test_native_gguf_companion.py"]
args = [sys.executable, "-m", "pytest", *["studio/backend/tests/" + f for f in files],
        "-q", "--timeout=90", "--junitxml=pr10210-results.xml"]
if sys.platform in ("win32", "darwin"):
    args += ["-k", "not test_hf_cache_entry_stays_within_the_scanned_case_variant"]
raise SystemExit(subprocess.call(args))
