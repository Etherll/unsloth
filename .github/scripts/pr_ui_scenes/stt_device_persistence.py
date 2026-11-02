from __future__ import annotations

import json
from pathlib import Path

from pr_ui_scenes._common import Session, api_get
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def _voice(page, base_url):
    await page.goto(base_url + "/chat", wait_until = "domcontentloaded")
    await page.get_by_role("button", name = "Settings", exact = True).click()
    dialog = page.get_by_role("dialog")
    await dialog.get_by_role("button", name = "Voice", exact = True).click()
    trigger = page.get_by_test_id("stt-device-trigger")
    await trigger.wait_for(state = "visible")
    return dialog, trigger


async def drive(session: Session, out_dir: Path, label: str, **_):
    auth = type(
        "Auth",
        (),
        {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
        },
    )()
    init = seed_init_script(auth, [])
    voice = json.dumps({"state": {"dictationEngine": "model", "sttDevice": "auto"}, "version": 0})
    init += "\nlocalStorage.setItem('unsloth_voice_settings', " + json.dumps(voice) + ");"
    async with open_chat(session.base_url, init_scripts = [init], viewport = (1350, 950)) as sp:
        _, trigger = await _voice(sp.page, session.base_url)
        await trigger.click()
        await sp.page.get_by_test_id("stt-device-cpu").click()
        await sp.page.wait_for_function(
            "document.querySelector('[data-testid=stt-device-trigger]')?.innerText.includes('CPU')"
        )
        if label.upper() == "AFTER":
            await sp.page.wait_for_function("""async () => {
                const r = await fetch('/api/settings/stt-device', {headers: {
                    Authorization: 'Bearer ' + localStorage.getItem('unsloth_auth_token')}});
                return r.ok && (await r.json()).device === 'cpu';
            }""")
    async with open_chat(session.base_url, init_scripts = [init], viewport = (1350, 950)) as sp:
        dialog, trigger = await _voice(sp.page, session.base_url)
        expected = "CPU" if label.upper() == "AFTER" else "Auto"
        await sp.page.wait_for_function(
            "expected => document.querySelector('[data-testid=stt-device-trigger]')?.innerText.includes(expected)",
            arg = expected,
        )
        choice = await trigger.inner_text()
        shot = out_dir / "fresh_browser_voice.png"
        await dialog.screenshot(path = str(shot))
    facts = {"fresh_browser_load_into": choice}
    try:
        facts["server_preference"] = api_get(session, "/api/settings/stt-device")["device"]
    except Exception:
        facts["server_preference"] = "endpoint absent"
    return [shot], facts
