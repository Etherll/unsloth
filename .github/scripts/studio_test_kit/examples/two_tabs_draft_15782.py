"""Strict staging check for unsloth#15782: two open tabs on different chats keep their own draft.

Run against a live Studio: BASE_URL, STUDIO_PW. Drives Chromium, Firefox and WebKit; in each,
one context (shared localStorage, real `storage` events) opens tab A on chat A and tab B on
chat B. Exits 1 if tab A's typing, clearing or long paste reaches tab B, or if the draft stops
following the user across chats within one tab.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playwright.async_api import async_playwright  # noqa: E402

from studio_test_kit.auth import login, seed_init_script  # noqa: E402

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8888")
PW = os.environ["STUDIO_PW"]
COMPOSER = 'textarea[name="input"]'
MARK = "PASTEMARK15782"
PASTE = MARK + "\n" + "pasted log line\n" * 400


def _call(method: str, path: str, token: str, body: dict) -> None:
    req = urllib.request.Request(
        BASE + path,
        json.dumps(body).encode(),
        method = method,
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    urllib.request.urlopen(req, timeout = 60).read()


def _seed(token: str, tag: str) -> dict[str, tuple[str, str, str]]:
    threads = {
        "A": (f"tabs-a-{tag}", f"Chat A {tag}", f"Stag beetles {tag} wrestle."),
        "B": (f"tabs-b-{tag}", f"Chat B {tag}", f"Sourdough {tag} rises."),
    }
    for i, (tid, title, reply) in enumerate(threads.values()):
        _call(
            "POST",
            "/api/chat/threads",
            token,
            {
                "id": tid,
                "title": title,
                "modelType": "base",
                "modelId": "unsloth/Qwen3-0.6B",
                "createdAt": 1_767_225_600_000 + i,
                "updatedAt": 1_767_225_600_000 + i,
            },
        )
        for j, (mid, parent, role, text) in enumerate(
            [("u1", None, "user", "One fact."), ("a1", "u1", "assistant", reply)]
        ):
            _call(
                "PUT",
                f"/api/chat/threads/{tid}/messages/{tid}-{mid}",
                token,
                {
                    "id": f"{tid}-{mid}",
                    "threadId": tid,
                    "parentId": f"{tid}-{parent}" if parent else None,
                    "role": role,
                    "content": [{"type": "text", "text": text}],
                    "createdAt": 1_767_225_600_000 + 10 * i + j + 1,
                },
            )
    return threads


async def _open(page, threads, name):
    tid, _, reply = threads[name]
    await page.goto(f"{BASE}/chat?thread={tid}", wait_until = "domcontentloaded")
    await page.get_by_text(reply).first.wait_for(state = "visible", timeout = 90_000)
    await page.locator(COMPOSER).first.wait_for(state = "visible", timeout = 60_000)


async def run_engine(p, engine: str, auth, out: Path) -> dict:
    threads = _seed(auth.access_token, f"{engine}-{uuid.uuid4().hex[:6]}")
    browser = await getattr(p, engine).launch(headless = True)
    ctx = await browser.new_context(viewport = {"width": 1200, "height": 800})
    await ctx.add_init_script(seed_init_script(auth, []))
    tab_a, tab_b = await ctx.new_page(), await ctx.new_page()
    await _open(tab_a, threads, "A")
    await _open(tab_b, threads, "B")
    box_a, box_b = tab_a.locator(COMPOSER).first, tab_b.locator(COMPOSER).first
    facts: dict = {}
    await box_b.fill("tab B's own text")
    await tab_b.wait_for_timeout(1200)
    await box_a.fill("typed only in tab A")
    await tab_a.wait_for_timeout(1500)
    facts["b_after_a_typed"] = await box_b.input_value()
    await box_a.fill("")
    await tab_a.wait_for_timeout(1500)
    facts["b_after_a_cleared"] = await box_b.input_value()
    await tab_a.evaluate(
        """(text) => {
        const el = document.querySelector('textarea[name="input"]');
        const dt = new DataTransfer(); dt.setData('text/plain', text);
        // Firefox drops clipboardData passed to the constructor, so attach it directly.
        const ev = new ClipboardEvent('paste', {bubbles: true, cancelable: true});
        Object.defineProperty(ev, 'clipboardData', {value: dt});
        el.dispatchEvent(ev);
    }""",
        PASTE,
    )
    await tab_a.wait_for_timeout(2000)
    facts["a_paste_chips"] = await tab_a.get_by_text(MARK).count()
    await tab_b.get_by_text(threads["A"][1]).first.click()
    await tab_b.get_by_text(threads["A"][2]).first.wait_for(state = "visible", timeout = 60_000)
    await tab_b.wait_for_timeout(2500)
    facts["b_paste_chips_after_switch"] = await tab_b.get_by_text(MARK).count()
    facts["b_text_after_switch"] = await tab_b.locator(COMPOSER).first.input_value()
    await tab_b.screenshot(path = str(out / f"{engine}_tab_b.png"))
    await tab_a.screenshot(path = str(out / f"{engine}_tab_a.png"))
    await browser.close()
    return facts


def _ensure_engines() -> None:
    subprocess.run([sys.executable, "-m", "playwright", "install", "firefox", "webkit"], check = True)
    subprocess.run(
        [sys.executable, "-m", "playwright", "install-deps", "firefox", "webkit"], check = False
    )


async def main() -> int:
    _ensure_engines()
    out = Path(os.environ.get("PW_ART_DIR", "logs/playwright_staging"))
    out.mkdir(parents = True, exist_ok = True)
    auth = await login(BASE, "unsloth", PW)
    failures = []
    async with async_playwright() as p:
        for engine in ("chromium", "firefox", "webkit"):
            facts = await run_engine(p, engine, auth, out)
            print(f"TWO_TABS {engine} {json.dumps(facts)}", flush = True)
            checks = {
                "typing stays in tab A": facts["b_after_a_typed"] == "tab B's own text",
                "clearing stays in tab A": facts["b_after_a_cleared"] == "tab B's own text",
                "paste became a chip in tab A": facts["a_paste_chips"] >= 1,
                "paste stays in tab A": facts["b_paste_chips_after_switch"] == 0,
                "draft follows tab B to another chat": facts["b_text_after_switch"]
                == "tab B's own text",
            }
            for name, ok in checks.items():
                print(f"  {'PASS' if ok else 'FAIL'} {engine}: {name}", flush = True)
                if not ok:
                    failures.append(f"{engine}: {name}")
    print("TWO_TABS_RESULT", "FAIL" if failures else "PASS", failures, flush = True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
