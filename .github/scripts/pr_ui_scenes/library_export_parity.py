from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

from pr_ui_scenes._common import Session, api_get, api_post
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


def _seed(session: Session) -> dict[str, bytes]:
    fixtures = {}
    for name, color in [("export-one.png", "#7c3aed"), ("export-two.png", "#0d9488")]:
        output = io.BytesIO()
        Image.new("RGB", (64, 64), color).save(output, format = "PNG")
        fixtures[name] = output.getvalue()
    for item in api_get(session, "/api/library")["items"]:
        if item["id"].startswith("upload:") and item["name"] in fixtures:
            api_post(session, "/api/library/items/delete", {"id": item["id"]})
    boundary = "unsloth-library-export-parity"
    body = bytearray()
    for name, content in fixtures.items():
        body.extend(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
                "Content-Type: image/png\r\n\r\n"
            ).encode()
        )
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        f"{session.base_url}/api/library/uploads",
        data = bytes(body),
        headers = {
            "Authorization": f"Bearer {session.access_token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method = "POST",
    )
    with urllib.request.urlopen(request, timeout = 60) as response:
        result = json.loads(response.read())
    assert len(result["ids"]) == 2, result
    return fixtures


async def drive(session: Session, out_dir: Path, label: str, **_: object):
    out_dir.mkdir(parents = True, exist_ok = True)
    fixtures = _seed(session)
    init = seed_init_script(
        type(
            "Auth",
            (),
            {"access_token": session.access_token, "refresh_token": session.refresh_token},
        )(),
        [],
    )
    settings = (
        'localStorage.setItem("theme", "dark");'
        'localStorage.setItem("unsloth_library_view", JSON.stringify({state:{view:"list"},version:0}));'
    )
    facts = {"fixture_names": sorted(fixtures), "image_count": len(fixtures), "theme": "dark"}
    async with open_chat(
        session.base_url, init_scripts = [init, settings], viewport = (1440, 900)
    ) as sp:
        page = sp.page
        await page.goto(
            f"{session.base_url}/library?show=images&sort=name", wait_until = "domcontentloaded"
        )
        await page.get_by_role("heading", name = "Library", exact = True).wait_for(state = "visible")
        for name in fixtures:
            await page.get_by_text(name, exact = True).wait_for(state = "visible", timeout = 60_000)
        await page.wait_for_function(
            "names => names.every(name => [...document.querySelectorAll('main img')].some(image => "
            "image.alt === name && image.complete && image.naturalWidth > 0 && getComputedStyle(image).opacity === '1'))",
            arg = list(fixtures),
            timeout = 60_000,
        )
        facts["loaded_thumbnail_names"] = sorted(fixtures)
        facts["library_heading"] = "Library"
        select_all = page.get_by_role("checkbox", name = "Select all", exact = True)
        await page.get_by_role("button", name = "Name", exact = True).hover()
        await select_all.click()
        toolbar = page.get_by_role("toolbar", name = "2 of 2 selected", exact = True)
        await toolbar.wait_for(state = "visible")
        download = toolbar.get_by_role("button", name = "Download", exact = True)
        await download.wait_for(state = "visible")
        facts["selection_label"] = await toolbar.get_attribute("aria-label")
        facts["select_all_checked"] = await select_all.is_checked()
        facts["download_enabled"] = await download.is_enabled()
        shot = out_dir / f"{label.lower()}_library_export.png"
        await sp.screenshot(shot, full_page = False)
        async with page.expect_download(timeout = 60_000) as event:
            await download.click()
        saved = await event.value
        archive = out_dir / "browser-export.zip"
        await saved.save_as(archive)
        with zipfile.ZipFile(archive) as bundle:
            assert sorted(bundle.namelist()) == sorted(fixtures), bundle.namelist()
            for name, content in fixtures.items():
                assert bundle.read(name) == content, name
            facts["archive_names"] = sorted(bundle.namelist())
            facts["archive_content_sha256"] = {
                name: hashlib.sha256(bundle.read(name)).hexdigest() for name in sorted(fixtures)
            }
        facts["browser_zip_preserves_bytes"] = True
    return [shot], facts
