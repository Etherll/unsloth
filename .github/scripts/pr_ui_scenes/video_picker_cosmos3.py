"""Scene: the Video page's model picker, curated rows (PR 14554).

PR 14554 adds the Cosmos3 video families (backend `video_families._FAMILIES` gains
`cosmos3-nano` and `cosmos3-edge`) and one curated row in the frontend VIDEO_CATALOG:
"Cosmos3 Edge" / "Text and image to video 480p, small" (Nano is deliberately left out
of the catalog: its checkpoint enables a guardrail package Studio does not install).

The picker is opened WITHOUT typing a query: a search would also list hub results,
and `nvidia/Cosmos3-Edge` is a real hub repo, so a query row could show it on BEFORE
too and the pair would prove nothing. Curated rows only.

Backend half: the family registry, read by importing THIS side's backend tree with
THIS side's venv python (there is no HTTP route listing video families). Listings
only: no weights, no GPU.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

TARGET = "Cosmos3 Edge"
TARGET_DESC = "Text and image to video 480p, small"
# A curated row present on both sides, to prove the list rendered at all.
ANCHOR = re.compile(r"LTX|Wan", re.I)

_PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from core.inference import video_families as vf
fam = vf.detect_video_family("nvidia/Cosmos3-Edge")
print("FACTS " + json.dumps({
    "families": [f.name for f in vf._FAMILIES],
    "detect_nvidia_Cosmos3_Edge": getattr(fam, "name", None),
}))
"""


def _backend_facts(session: Session, label: str, pr: int) -> dict:
    tree = WORKSPACE / "temp" / f"uidiff{pr}_{label.lower()}"
    backend = tree / "studio" / "backend"
    py = session.home / "unsloth_studio" / "bin" / "python"
    out: dict = {"backend_tree": str(backend)}
    try:
        out["tree_sha"] = subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "HEAD"], text = True, capture_output = True
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        out["tree_sha"] = f"error {exc}"
    if not py.exists() or not backend.is_dir():
        out["error"] = f"missing venv python ({py.exists()}) or backend ({backend.is_dir()})"
        return out
    r = subprocess.run(
        [str(py), "-c", _PROBE, str(backend)],
        text = True,
        capture_output = True,
        cwd = str(backend),
        timeout = 300,
    )
    for line in r.stdout.splitlines():
        if line.startswith("FACTS "):
            out.update(json.loads(line[6:]))
            out["cosmos3_edge_in_registry"] = "cosmos3-edge" in out["families"]
            return out
    out["error"] = (r.stderr or r.stdout).strip()[-400:]
    return out


def _put(session: Session, path: str, payload: dict) -> dict:
    import urllib.request
    req = urllib.request.Request(
        session.base_url + path,
        data = json.dumps(payload).encode(),
        method = "PUT",
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {session.access_token}",
        },
    )
    with urllib.request.urlopen(req, timeout = 60) as r:
        return json.loads(r.read().decode())


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    pr: int = 14554,
    clip: Optional[dict] = None,
    **_: object,
) -> tuple[list[Path], dict]:
    facts: dict = {"backend": _backend_facts(session, label, pr)}
    # This session has no visible GPU, so /video shows the "no accelerator" notice instead of the
    # picker. Developer mode's experimental CPU video (same PUT on both sides) renders the page;
    # nothing is loaded, so no weights are fetched.
    facts["developer_mode"] = _put(session, "/api/settings/developer-mode", {"video_cpu": True})
    out_dir.mkdir(parents = True, exist_ok = True)
    shots: list[Path] = []
    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1500, 1100), headless = True
    ) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/video", wait_until = "domcontentloaded")
        trigger = page.locator('[data-tour="video-model"]').first
        try:
            await trigger.wait_for(state = "visible", timeout = 90_000)
        except Exception:
            await page.screenshot(path = str(out_dir / f"{label.lower()}_debug.png"))
            (out_dir / f"{label.lower()}_debug.txt").write_text(
                page.url + "\n" + await page.locator("body").inner_text()
            )
            raise
        await page.wait_for_timeout(4_000)

        dialog = None
        for _attempt in range(5):
            await trigger.click()
            await page.wait_for_timeout(1_500)
            cand = page.locator("[role=dialog], [data-radix-popper-content-wrapper]").last
            if await cand.count() and await cand.get_by_text(ANCHOR).count():
                dialog = cand
                break
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(1_000)
        if dialog is None:
            raise RuntimeError("model picker did not open with curated video rows")
        await page.wait_for_timeout(3_000)

        text = await dialog.inner_text()
        lines = [t.strip() for t in text.splitlines() if t.strip()]
        facts["picker_lines"] = lines[:120]
        facts["cosmos_lines"] = [t for t in lines if "cosmos" in t.lower()]
        facts["target_row_shown"] = TARGET in text
        facts["target_desc_shown"] = TARGET_DESC in text

        # Bring the bottom of the curated list into view (Cosmos3 Edge is the last group).
        row = dialog.get_by_text(TARGET, exact = True)
        if await row.count():
            await row.first.scroll_into_view_if_needed()
        else:
            # Same scroll on BEFORE: the end of the curated list, where the row would be.
            await dialog.evaluate(
                "el => el.querySelectorAll('*').forEach(n => { if (n.scrollHeight > n.clientHeight + 4) n.scrollTop = n.scrollHeight; })"
            )
        await page.wait_for_timeout(1_500)

        shot = out_dir / f"{label.lower()}_video_picker.png"
        if clip:
            await page.screenshot(path = str(shot), clip = clip)
        else:
            await dialog.screenshot(path = str(shot))
        shots.append(shot)
        full = out_dir / f"{label.lower()}_video_page.png"
        await sp.screenshot(full, full_page = False)
        shots.append(full)

    print(
        f"[{label}] {json.dumps({k: v for k, v in facts.items() if k != 'picker_lines'})[:900]}",
        flush = True,
    )
    return shots, facts
