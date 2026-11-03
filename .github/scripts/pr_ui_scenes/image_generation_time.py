from pathlib import Path
import json
import os

from PIL import Image, PngImagePlugin

from pr_ui_scenes._common import api_get
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def drive(session, out_dir: Path, label: str, **kwargs):
    directory = session.home / "images"
    directory.mkdir(parents = True, exist_ok = True)
    common = dict(
        prompt = "Timing display fixture (not a benchmark)",
        negative_prompt = None,
        width = 512,
        height = 512,
        steps = 1,
        guidance = 1,
        seed = 7,
        batch_index = 0,
        batch_size = 2,
        model = None,
        created_at = 1700000000,
    )
    for name, extra, color in (
        ("timing-old", {}, "#415147"),
        ("timing-new", dict(generation_seconds = 2.25, generation_count = 2), "#415147"),
    ):
        info = PngImagePlugin.PngInfo()
        info.add_text("unsloth", json.dumps({**common, **extra}))
        path = directory / f"{name}.png"
        Image.new("RGB", (512, 512), color).save(path, pnginfo = info)
        os.utime(path, (1700000000, 1700000000))
    rows = api_get(session, "/api/inference/images/gallery")["images"]
    by_id = {row["id"]: row for row in rows}
    assert {"timing-new", "timing-old"} <= by_id.keys(), rows
    init = seed_init_script(
        type(
            "Auth", (), dict(access_token = session.access_token, refresh_token = session.refresh_token)
        )(),
        [],
    )
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1440, 1000), headless = True
    ) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/images", wait_until = "domcontentloaded")
        await page.locator('[data-image-id="timing-new"] > button').click()
        await page.get_by_role("button", name = "Recipe", exact = True).click()
        panel = page.locator('[data-slot="popover-content"]')
        await panel.wait_for(state = "visible")
        text = await panel.inner_text()
        expected = "2.3 s (batch of 2)"
        has_timing = by_id["timing-new"].get("generation_seconds") is not None
        assert (expected in text) == has_timing, text
        shot = out_dir / f"{label.lower()}_image_time.png"
        await panel.screenshot(path = str(shot))
        await page.keyboard.press("Escape")
        await page.locator('[data-image-id="timing-old"] > button').click()
        await page.get_by_role("button", name = "Recipe", exact = True).click()
        await panel.wait_for(state = "visible")
        old_text = await panel.inner_text()
        assert "Time\n" not in old_text, old_text
    return [shot], dict(
        timing_row = has_timing,
        generation_seconds = by_id["timing-new"].get("generation_seconds"),
        generation_count = by_id["timing-new"].get("generation_count"),
        legacy_timing_row = False,
        recipe_text = text,
        fixture = "Persisted deterministic PNG; display evidence only",
    )
