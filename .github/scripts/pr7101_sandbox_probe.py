#!/usr/bin/env python3
"""Secret-free A/B probe for PR 7101 sandbox runtime paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import site
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required = True)
    parser.add_argument("--expect", choices = ("negative", "positive"), required = True)
    args = parser.parse_args()

    backend = Path(args.repo_root).resolve() / "studio" / "backend"
    sys.path.insert(0, str(backend))
    from core.inference import sandbox, tools

    if not sandbox.sandbox_available():
        raise SystemExit("bubblewrap sandbox unavailable; refusing vacuous probe")

    with tempfile.TemporaryDirectory() as workdir:
        tools._workdirs["_pr7101_probe"] = workdir
        awk_output = tools._bash_exec(
            "awk 'BEGIN {print 42}'",
            session_id = "_pr7101_probe",
            timeout = 30,
        ).strip()
        awk_ok = awk_output == "42"

    with tempfile.TemporaryDirectory() as user_site:
        os.environ["UNSLOTH_STUDIO_SANDBOX_ALLOW_USER_SITE"] = "1"
        original_user_site = site.getusersitepackages
        site.getusersitepackages = lambda: user_site
        try:
            child_paths = tools._build_safe_env(user_site)["PYTHONPATH"].split(os.pathsep)
            user_site_ok = os.path.realpath(user_site) in child_paths
        finally:
            site.getusersitepackages = original_user_site
            os.environ.pop("UNSLOTH_STUDIO_SANDBOX_ALLOW_USER_SITE", None)

    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)
        source = root / "src"
        source.mkdir()
        package = source / "probe_package"
        package.mkdir()
        (package / "__init__.py").write_text("VALUE = 1\n")
        (source / ".env").write_text("SHOULD_NOT_BE_MOUNTED=1\n")
        site_dir = root / "site-packages"
        site_dir.mkdir()
        (site_dir / "editable.pth").write_text(str(source) + "\n")
        original_site_packages = site.getsitepackages
        site.getsitepackages = lambda: [str(site_dir)]
        try:
            editable_paths = [os.path.realpath(path) for path in sandbox._editable_source_paths()]
            pth_ok = (
                os.path.realpath(str(source)) not in editable_paths
                and os.path.realpath(str(package)) in editable_paths
            )
        finally:
            site.getsitepackages = original_site_packages

    result = {
        "awk_alternatives": awk_ok,
        "user_site_import_path": user_site_ok,
        "plain_pth_source": pth_ok,
    }
    print(f"PROBE mode={args.expect} result={json.dumps(result, sort_keys = True)}")
    expected = args.expect == "positive"
    if any(value is not expected for value in result.values()):
        raise SystemExit(f"unexpected {args.expect} result: {result}")


if __name__ == "__main__":
    main()
