"""Apple Silicon end to end: an MLX vision chat whose prompt overflows the served window.

Boots Studio (API only) from the merge-base tree and from the checked-out tree in turn, loads one
real MLX VLM at a small context, and sends OpenAI chat requests with context_overflow =
truncate_oldest:

    old_turn       picture on the first user turn, long text history after it
    newest_turn    long text history, picture on the newest user turn (red square: answer says red)
    fits           short picture chat, no overflow (both arms must answer, say red)
    text_overflow  long text chat, no picture (both arms must fit it)

    python jobs/mlx_vision_compaction.py --base-sha <merge base> [--model R] [--ctx 2048]

Checks are on head; base results are recorded so the report can show the defect on base.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
UPSTREAM = "https://github.com/unslothai/unsloth"
FILLER = (
    "Here is a long note about gardening, soil, watering schedules, compost, mulch, seasonal "
    "planting, pruning, pests and harvest timing that the assistant should simply acknowledge. "
)


def _red_png(rgb = (230, 20, 20)) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (448, 448), rgb).save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _http(
    method,
    url,
    body = None,
    token = None,
    timeout = 900,
):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"}, method = method)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, raw.decode(errors = "replace")


def _transformers_imports() -> bool:
    return (
        subprocess.run(
            [sys.executable, "-c", "import transformers"], capture_output = True
        ).returncode
        == 0
    )


def _install_backend_deps(repo: Path):
    try:
        import fastapi, jwt, sqlalchemy  # noqa: F401
    except ImportError:
        _install_studio_deps(repo)
    if not _transformers_imports():
        # transformers pins tokenizers; a newer resolve breaks every model load identically on both arms.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "tokenizers>=0.22.0,<=0.23.0"],
            check = True,
        )
    if not _transformers_imports():
        raise RuntimeError("transformers does not import on this runner")


def _install_studio_deps(repo: Path):
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run([*pip, "-r", str(repo / "studio/backend/requirements/studio.txt")], check = True)
    subprocess.run(
        [
            *pip,
            "python-multipart",
            "aiofiles",
            "sqlalchemy",
            "cryptography",
            "pyyaml",
            "jinja2",
            "mammoth",
            "unpdf",
            "requests",
            "httpx",
            "pillow",
        ],
        check = True,
    )


def _base_tree(repo: Path, sha: str, root: Path) -> Path:
    dst = root / "base"
    subprocess.run(
        ["git", "-C", str(repo), "fetch", "-q", "--depth", "1", UPSTREAM, sha], check = True
    )
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-q", "--detach", str(dst), sha], check = True
    )
    return dst


def _thread(picture, where, turns):
    msgs = []
    for i in range(turns):
        text = f"Note {i}. " + FILLER * 22
        if where == "old" and i == 0:
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": picture}},
                        {"type": "text", "text": "Remember this picture. " + text},
                    ],
                }
            )
        else:
            msgs.append({"role": "user", "content": text})
        msgs.append({"role": "assistant", "content": f"Noted {i}."})
    question = "What colour is the square in the picture? Answer with one word."
    if where == "new":
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": picture}},
                    {"type": "text", "text": question},
                ],
            }
        )
    else:
        msgs.append({"role": "user", "content": question if where != "none" else "Say done."})
    if where == "q":
        msgs[-1] = {"role": "user", "content": question}
    return msgs


def _arm(label, tree: Path, model, ctx, port, picture, logdir: Path):
    tree = Path(tree).resolve()
    logdir.mkdir(parents = True, exist_ok = True)
    home = Path(tempfile.mkdtemp(prefix = f"studio_{label}_", dir = logdir))
    log = logdir / f"studio_{label}.log"
    env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home), "PYTHONUNBUFFERED": "1"}
    env.pop("GITHUB_ENV", None)
    proc = subprocess.Popen(
        [
            sys.executable,
            str(tree / "studio/backend/run.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-only",
        ],
        cwd = str(tree / "studio/backend"),
        env = env,
        stdout = open(log, "w"),
        stderr = subprocess.STDOUT,
        start_new_session = True,
    )
    base_url = f"http://127.0.0.1:{port}"
    out = {"tree": str(tree), "cases": {}}
    try:
        pw_file = home / "auth" / ".bootstrap_password"
        deadline = time.time() + 600
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"{label} studio exited {proc.returncode}: " + log.read_text()[-3000:]
                )
            try:
                if _http("GET", base_url + "/api/health", timeout = 5)[0] == 200 and pw_file.exists():
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise RuntimeError(f"{label} studio never became healthy: " + log.read_text()[-3000:])
        boot = pw_file.read_text().strip()
        new = boot + "-Rotated1!"
        _, tok = _http(
            "POST", base_url + "/api/auth/login", {"username": "unsloth", "password": boot}
        )
        tok = tok["access_token"]
        _http(
            "POST",
            base_url + "/api/auth/change-password",
            {"current_password": boot, "new_password": new},
            tok,
        )
        _, tok = _http(
            "POST", base_url + "/api/auth/login", {"username": "unsloth", "password": new}
        )
        tok = tok["access_token"]
        t0 = time.time()
        status, body = _http(
            "POST",
            base_url + "/api/inference/load",
            {"model_path": model, "max_seq_length": ctx},
            tok,
            timeout = 1800,
        )
        out["load"] = {"status": status, "s": round(time.time() - t0, 1), "body": str(body)[:3000]}
        if status != 200:
            raise RuntimeError(f"{label} load failed: {status} {str(body)[:2000]}")
        deadline = time.time() + 1800
        while time.time() < deadline:
            _, st = _http("GET", base_url + "/api/inference/status", token = tok)
            if isinstance(st, dict) and st.get("active_model"):
                break
            time.sleep(5)
        out["status"] = str(st)[:3000]
        if not (isinstance(st, dict) and st.get("active_model")):
            out["log_tail"] = log.read_text(errors = "replace")[-6000:]
            raise RuntimeError(f"{label}: no active model after load: {str(st)[:1500]}")
        cases = {
            "old_turn": _thread(picture, "old", 8),
            "newest_turn": _thread(picture, "new", 8),
            "fits": _thread(picture, "new", 0),
            "text_overflow": _thread(picture, "none", 8),
            "fits_green": _thread(_red_png((20, 200, 20)), "new", 0),
            "fits_blue": _thread(_red_png((20, 20, 230)), "new", 0),
            "fits_text_twin": _thread(picture, "q", 0),
        }
        for name, messages in cases.items():
            t0 = time.time()
            status, body = _http(
                "POST",
                base_url + "/v1/chat/completions",
                {
                    "model": model,
                    "messages": messages,
                    "max_tokens": 24,
                    "temperature": 0,
                    "stream": False,
                    "context_overflow": "truncate_oldest",
                },
                tok,
            )
            rec = {"status": status, "s": round(time.time() - t0, 2)}
            if status == 200 and isinstance(body, dict):
                rec["content"] = (body.get("choices") or [{}])[0].get("message", {}).get("content")
                rec["context_truncated"] = body.get("context_truncated")
                rec["usage"] = body.get("usage")
            else:
                rec["error"] = str(body)[:1500]
            out["cases"][name] = rec
            print(f"[{label}] {name}: {json.dumps(rec)[:600]}", flush = True)
        # Gate B1: a picture chat that fits and a short text chat, repeated, wall time per request.
        out["timing"] = {}
        for name, messages in (("fits", cases["fits"]), ("text_fits", _thread(picture, "none", 0))):
            times = []
            for _ in range(5):
                t0 = time.time()
                status, _body = _http(
                    "POST",
                    base_url + "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": messages,
                        "max_tokens": 16,
                        "temperature": 0,
                        "stream": False,
                        "context_overflow": "truncate_oldest",
                    },
                    tok,
                )
                times.append(round(time.time() - t0, 3) if status == 200 else None)
            out["timing"][name] = times
        print(f"[{label}] timing: {out['timing']}", flush = True)
    except Exception as exc:
        out["error"] = repr(exc)[:3000]
        out["log_tail"] = log.read_text(errors = "replace")[-6000:]
    finally:
        try:
            os.killpg(proc.pid, 15)
            proc.wait(timeout = 60)
        except Exception:
            try:
                os.killpg(proc.pid, 9)
            except Exception:
                pass
        shutil.rmtree(home, ignore_errors = True)
    return out


def main(argv = None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default = DEFAULT_MODEL)
    p.add_argument("--base-sha", default = None)
    p.add_argument("--ctx", type = int, default = 2048)
    p.add_argument("--out", default = "outputs/jobs/mlx_vision_compaction.json")
    p.add_argument("--tiny", action = "store_true")
    p.add_argument("--backend", default = "auto")
    p.add_argument("--max-steps", type = int, default = 0)
    args = p.parse_args(argv)
    args._job = "mlx_vision_compaction"
    with C.JobRecorder(args, backend_hint = "studio-mlx") as rec:
        import platform

        if sys.platform != "darwin" or platform.machine() != "arm64":
            rec.skip("needs Apple Silicon")
        repo = Path(
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text = True).strip()
        )
        _install_backend_deps(repo)
        logdir = Path(args.out).resolve().parent
        logdir.mkdir(parents = True, exist_ok = True)
        picture = _red_png()
        arms = {}
        if args.base_sha:
            root = Path(tempfile.mkdtemp(prefix = "mvc_"))
            arms["base"] = _arm(
                "base",
                _base_tree(repo, args.base_sha, root),
                args.model,
                args.ctx,
                18961,
                picture,
                logdir,
            )
            rec.summary(base = arms["base"])
        arms["head"] = _arm("head", repo, args.model, args.ctx, 18962, picture, logdir)
        rec.summary(head = arms["head"])
        head = arms["head"]["cases"]
        for name in ("old_turn", "newest_turn", "text_overflow"):
            r = head.get(name, {})
            rec.check(
                f"head_{name}_answers_truncated",
                r.get("status") == 200
                and bool((r.get("context_truncated") or {}).get("dropped_messages")),
                json.dumps(r)[:800],
            )
        twin = ((head.get("fits_text_twin") or {}).get("usage") or {}).get("prompt_tokens") or 0
        used = ((head.get("fits") or {}).get("usage") or {}).get("prompt_tokens") or 0
        rec.check(
            "head_fits_picture_in_prompt",
            used - twin > 50,
            f"prompt_tokens {used}, text twin {twin}",
        )
        rec.summary(
            colours = {
                a: {
                    n: arms[a]["cases"].get(n, {}).get("content")
                    for n in ("fits", "fits_green", "fits_blue")
                }
                for a in arms
            }
        )
        if "base" in arms:
            b = arms["base"]["cases"]
            rec.summary(
                base_defect = {
                    n: b.get(n, {}).get("status") != 200 for n in ("old_turn", "newest_turn")
                }
            )
            rec.check(
                "base_text_overflow_answers_truncated",
                bool(
                    (b.get("text_overflow", {}).get("context_truncated") or {}).get(
                        "dropped_messages"
                    )
                ),
                json.dumps(b.get("text_overflow"))[:800],
            )


if __name__ == "__main__":
    main()
