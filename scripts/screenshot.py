"""Capture dashboard screenshots. Development tooling, not a runtime dependency.

Firefox's built-in ``--screenshot`` fires before async fetches settle, so the
dashboard captures blank. Playwright waits for network idle and can drive the
tab switches, which is what makes the shots representative.
"""
from __future__ import annotations

import asyncio
import sys

from playwright.async_api import async_playwright

VIEWS = ["overview", "forecasts", "telemetry", "adaptation", "experiments", "chat"]


async def main(url: str, outdir: str, theme: str = "light") -> int:
    async with async_playwright() as p:
        browser = await p.firefox.launch()
        page = await browser.new_page(viewport={"width": 1480, "height": 1150})
        errors: list[str] = []
        page.on("console", lambda m: errors.append(f"{m.type}: {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        await page.goto(url, wait_until="networkidle")
        if theme == "dark":
            # The button cycles auto -> dark -> light, so one click from a fresh
            # page selects dark. Setting the attribute *and* clicking cancels out
            # and silently produces light-mode captures.
            await page.click("#themebtn")
            await page.wait_for_timeout(500)
            got = await page.evaluate("document.documentElement.getAttribute('data-theme')")
            if got != "dark":
                raise SystemExit(f"expected dark theme, got {got!r}")
        for view in VIEWS:
            await page.click(f'#tabs button[data-view="{view}"]')
            await page.wait_for_timeout(1400)
            suffix = f"-{theme}" if theme != "light" else ""
            await page.screenshot(path=f"{outdir}/{view}{suffix}.png", full_page=True)
            print(f"captured {view}{suffix}")
        await browser.close()
        if errors:
            print("\nconsole messages:")
            for e in dict.fromkeys(errors):
                print(" ", e)
            return 1
        print("\nno console errors")
        return 0


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8791/"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "docs/screenshots"
    theme = sys.argv[3] if len(sys.argv) > 3 else "light"
    raise SystemExit(asyncio.run(main(url, outdir, theme)))
