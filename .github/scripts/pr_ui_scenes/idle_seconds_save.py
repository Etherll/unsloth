from __future__ import annotations

import json
from pathlib import Path

from playwright.async_api import TimeoutError as PlaywrightTimeoutError, expect

from pr_ui_scenes._common import Session, api_get
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def drive(session: Session, out_dir: Path, label: str, **_: object):
    shots = []
    facts = {}
    auth = type(
        "Auth", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
    )()
    headers = {"Authorization": f"Bearer {session.access_token}"}
    async with open_chat(
        session.base_url,
        init_scripts = [seed_init_script(auth, [])],
        viewport = (1440, 1000),
        headless = True,
    ) as sp:
        page = sp.page

        async def seed(**values):
            res = await page.request.put(
                session.base_url + "/api/settings/openai-auto-switch", headers = headers, data = values
            )
            assert res.ok, await res.text()
            return await res.json()

        async def tab(name = "API"):
            await page.reload(wait_until = "domcontentloaded")
            await page.get_by_role("button", name = "Settings", exact = True).click()
            dialog = page.get_by_role("dialog")
            await dialog.get_by_role("button", name = name, exact = True).click()
            field_name = (
                "Idle auto-unload seconds"
                if name == "API"
                else "Image and video idle auto-unload seconds"
            )
            await expect(dialog.get_by_label(field_name, exact = True)).to_be_enabled()
            return dialog

        async def shot(dialog, name):
            target = out_dir / f"{label.lower()}_{name}.png"
            await dialog.screenshot(path = str(target))
            shots.append(target)

        async def enter(field):
            try:
                async with page.expect_response(
                    lambda r: r.url.endswith("/api/settings/openai-auto-switch")
                    and r.request.method == "PUT",
                    timeout = 2000,
                ) as response:
                    await field.press("Enter")
                res = await response.value
                assert res.ok, await res.text()
                return {
                    "sent": True,
                    "body": json.loads(res.request.post_data),
                    "response": await res.json(),
                }
            except PlaywrightTimeoutError:
                return {"sent": False}

        await seed(enabled = True, auto_unload_idle_seconds = 60)
        dialog = await tab()
        field = dialog.get_by_label("Idle auto-unload seconds", exact = True)
        await field.fill("0")
        row = field.locator("xpath=ancestor::*[@data-settings-label][1]")
        async with page.expect_response(
            lambda r: r.url.endswith("/api/settings/openai-auto-switch")
            and r.request.method == "PUT"
        ) as response:
            await row.get_by_role("button", name = "Save", exact = True).click()
        res = await response.value
        assert res.ok, await res.text()
        facts["click_save_zero_payload"] = json.loads(res.request.post_data)
        facts["click_save_zero_stored"] = api_get(session, "/api/settings/openai-auto-switch")
        assert facts["click_save_zero_stored"]["auto_unload_idle_seconds"] == 0
        dialog = await tab()
        assert (
            await dialog.get_by_label("Idle auto-unload seconds", exact = True).input_value() == "0"
        )

        await seed(enabled = True, auto_unload_idle_seconds = 60)
        dialog = await tab()
        field = dialog.get_by_label("Idle auto-unload seconds", exact = True)
        await field.fill("0")
        facts["chat_enter_zero"] = await enter(field)
        facts["chat_stored_after_enter"] = api_get(session, "/api/settings/openai-auto-switch")
        dialog = await tab()
        field = dialog.get_by_label("Idle auto-unload seconds", exact = True)
        facts["chat_reloaded_value"] = await field.input_value()
        await field.scroll_into_view_if_needed()
        await shot(dialog, "chat_enter_reloaded")

        await seed(media_auto_unload_idle_seconds = 60)
        dialog = await tab("System")
        field = dialog.get_by_label("Image and video idle auto-unload seconds", exact = True)
        await field.fill("0")
        facts["media_enter_zero"] = await enter(field)
        facts["media_stored_after_enter"] = api_get(session, "/api/settings/openai-auto-switch")
        dialog = await tab("System")
        field = dialog.get_by_label("Image and video idle auto-unload seconds", exact = True)
        facts["media_reloaded_value"] = await field.input_value()
        await field.scroll_into_view_if_needed()
        await shot(dialog, "media_enter_reloaded")

        await seed(enabled = True, auto_unload_idle_seconds = 60)
        dialog = await tab()
        field = dialog.get_by_label("Idle auto-unload seconds", exact = True)
        await field.fill("30")
        facts["invalid_enter"] = await enter(field)
        facts["invalid_error_visible"] = (
            await dialog.get_by_text(
                "Enter 0 to keep the model loaded, or at least 60 seconds.", exact = True
            ).count()
            > 0
        )
        facts["invalid_stored"] = api_get(session, "/api/settings/openai-auto-switch")[
            "auto_unload_idle_seconds"
        ]
        await field.scroll_into_view_if_needed()
        await shot(dialog, "invalid_enter")
    return shots, facts
