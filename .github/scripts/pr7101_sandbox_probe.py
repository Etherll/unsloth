#!/usr/bin/env python3
"""Secret-free A/B probe for the final PR 7101 sandbox remediation."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
from pathlib import Path
import site
import sys
import tempfile


def _contains_sequence(values: list[str], expected: list[str]) -> bool:
    return any(values[index : index + len(expected)] == expected for index in range(len(values)))


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

    with tempfile.TemporaryDirectory() as workdir_text:
        workdir = Path(workdir_text)

        previous_flags = {
            name: os.environ.get(name)
            for name in ("HSA_ENABLE_DXG_DETECTION", "HSA_OVERRIDE_GFX_VERSION")
        }
        os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        try:
            safe_env = tools._build_safe_env(str(workdir))
            hsa_flags_ok = (
                safe_env.get("HSA_ENABLE_DXG_DETECTION") == "1"
                and safe_env.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"
            )
        finally:
            for name, value in previous_flags.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        site_dir = workdir / "site-packages"
        source = workdir / "native-source"
        site_dir.mkdir()
        source.mkdir()
        extension = source / f"native_probe{importlib.machinery.EXTENSION_SUFFIXES[0]}"
        extension.write_bytes(b"secret-free-placeholder")
        namespace_root = source / "native_namespace"
        native_leaf = namespace_root / "nested" / "leaf"
        native_leaf.mkdir(parents = True)
        namespace_extension = native_leaf / f"_native{importlib.machinery.EXTENSION_SUFFIXES[0]}"
        namespace_extension.write_bytes(b"secret-free-placeholder")
        (site_dir / "editable.pth").write_text(str(source) + "\n")
        original_site_packages = site.getsitepackages
        site.getsitepackages = lambda: [str(site_dir)]
        try:
            native_paths = {os.path.realpath(path) for path in sandbox._editable_source_paths()}
            native_extension_ok = os.path.realpath(extension) in native_paths
            namespace_native_extension_ok = os.path.realpath(namespace_root) in native_paths
        finally:
            site.getsitepackages = original_site_packages

        alias_target = workdir / "alias-target"
        alias_target.mkdir()
        alias = workdir / "alias-source"
        alias.symlink_to(alias_target, target_is_directory = True)
        alias_package = alias / "probe_package"
        (alias_target / "probe_package").mkdir()
        original_editable_paths = sandbox._editable_source_paths
        sandbox._editable_source_paths = lambda: [str(alias_package)]
        try:
            read_paths = sandbox._python_read_paths()
            editable_alias_ok = (
                os.path.abspath(alias_package) in read_paths
                and os.path.realpath(alias_package) in read_paths
            )
        finally:
            sandbox._editable_source_paths = original_editable_paths

        fixed_probe_ok = (
            hasattr(sandbox, "_linux_bwrap_probe_path")
            and not hasattr(sandbox, "_BWRAP_PROBE_BIN")
            and sandbox._linux_bwrap_probe_path() is not None
        )

        installer_text = (Path(args.repo_root).resolve() / "install.sh").read_text(
            encoding = "utf-8"
        )
        installer_seccomp_probe_ok = all(
            marker in installer_text
            for marker in (
                'find_library("seccomp") or "libseccomp.so.2"',
                'hasattr(os, "memfd_create")',
                'bubblewrap_path_trusted "$PYTHON_BIN"',
            )
        )

        trusted_rlimit_python_ok = False
        nproc_host_budget_ok = False
        rlimit_site_disabled_ok = False
        if hasattr(sandbox, "_linux_rlimit_python_path"):
            trusted_rlimit_python = sandbox._linux_rlimit_python_path()
            trusted_rlimit_python_ok = bool(
                trusted_rlimit_python
                and sandbox._linux_executable_path_is_trusted(trusted_rlimit_python)
            )
        if hasattr(sandbox, "_linux_nproc_rlimit_target"):
            host_tasks = sandbox._linux_real_uid_task_count()
            target = sandbox._linux_nproc_rlimit_target()
            nproc_host_budget_ok = bool(
                host_tasks is not None
                and target is not None
                and target - host_tasks == sandbox._resolve_nproc_limit()
            )
        inner = sandbox._linux_inner_rlimit_wrapper(["/usr/bin/true"])
        rlimit_site_disabled_ok = inner[1:4] == ["-I", "-S", "-c"]

        identity_workdir = workdir / "identity-workdir"
        identity_workdir.mkdir()
        previous_tmpdir = os.environ.get("TMPDIR")
        previous_tempdir_cache = tempfile.tempdir
        os.environ["TMPDIR"] = str(identity_workdir)
        tempfile.tempdir = None
        sandbox._sandbox_identity_paths = None
        try:
            try:
                identity_paths = sandbox._linux_sandbox_identity_files(str(identity_workdir))
            except TypeError:
                identity_paths = sandbox._linux_sandbox_identity_files()
            identity_outside_workdir_ok = all(
                not sandbox._path_is_within(path, str(identity_workdir)) for path in identity_paths
            )
        finally:
            tempfile.tempdir = previous_tempdir_cache
            if previous_tmpdir is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous_tmpdir

        runtime = workdir / ".venv"
        runtime.mkdir()
        original_read_paths = sandbox._python_read_paths
        sandbox._python_read_paths = lambda: [os.path.realpath(runtime)]
        try:
            argv = sandbox._linux_bwrap_argv(["/usr/bin/true"], str(workdir))
            cgroup_ok = _contains_sequence(
                argv, ["--ro-bind-try", "/sys/fs/cgroup", "/sys/fs/cgroup"]
            )
            runtime_path = os.path.realpath(runtime)
            editable_runtime_writable_ok = not _contains_sequence(
                argv, ["--ro-bind-try", runtime_path, runtime_path]
            )
        finally:
            if "argv" in locals() and hasattr(sandbox, "close_sandbox_argv_fds"):
                sandbox.close_sandbox_argv_fds(argv)
            sandbox._python_read_paths = original_read_paths

        oneapi_root = workdir / "opt" / "intel" / "oneapi"
        oneapi_root.mkdir(parents = True)
        if hasattr(sandbox, "_linux_oneapi_runtime_bindings"):
            original_oneapi_roots = sandbox._LINUX_ONEAPI_ROOTS
            sandbox._LINUX_ONEAPI_ROOTS = (str(oneapi_root),)
            try:
                oneapi_ok = sandbox._linux_oneapi_runtime_bindings() == [
                    (os.path.realpath(oneapi_root), os.path.normpath(oneapi_root))
                ]
            finally:
                sandbox._LINUX_ONEAPI_ROOTS = original_oneapi_roots
        else:
            oneapi_ok = False

        original_read_paths = sandbox._python_read_paths
        sandbox._python_read_paths = lambda: []
        try:
            profile = sandbox._macos_seatbelt_profile(str(workdir))
            macos_host_ipc_denied_ok = (
                "(allow ipc-posix-shm)" not in profile
                and "(allow ipc-posix-sem)" not in profile
            )
        finally:
            sandbox._python_read_paths = original_read_paths

        tools._workdirs["_pr7101_socket_probe"] = str(workdir)
        try:
            socket_output = tools._python_exec(
                "import socket\n"
                "try:\n"
                "    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
                "    print('UNIX_ALLOWED')\n"
                "except OSError as exc:\n"
                "    print('UNIX_DENIED', exc.errno)\n",
                session_id = "_pr7101_socket_probe",
                timeout = 30,
            )
            unix_socket_denied_ok = "UNIX_DENIED 1" in socket_output
        finally:
            tools._workdirs.pop("_pr7101_socket_probe", None)

    result = {
        "cgroup_view": cgroup_ok,
        "editable_alias": editable_alias_ok,
        "editable_runtime_writable": editable_runtime_writable_ok,
        "fixed_probe_payload": fixed_probe_ok,
        "hsa_runtime_flags": hsa_flags_ok,
        "identity_outside_tmpdir": identity_outside_workdir_ok,
        "installer_seccomp_probe": installer_seccomp_probe_ok,
        "macos_host_posix_ipc_denied": macos_host_ipc_denied_ok,
        "native_pth_extension": native_extension_ok,
        "namespace_native_extension": namespace_native_extension_ok,
        "nproc_host_budget": nproc_host_budget_ok,
        "oneapi_runtime": oneapi_ok,
        "rlimit_site_disabled": rlimit_site_disabled_ok,
        "trusted_rlimit_python": trusted_rlimit_python_ok,
        "unix_socket_seccomp": unix_socket_denied_ok,
    }
    print(f"PROBE mode={args.expect} result={json.dumps(result, sort_keys = True)}")
    expected = args.expect == "positive"
    if any(value is not expected for value in result.values()):
        raise SystemExit(f"unexpected {args.expect} result: {result}")


if __name__ == "__main__":
    main()
