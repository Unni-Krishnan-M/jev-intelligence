"""Drive the running app through the core user journey in a real browser and save screenshots.

Requires the API (default :8000) and the frontend (default :3000) to be running, plus Chrome or
Chromium: `uv run python scripts/capture_screenshots.py [--base http://localhost:3000]`.
`--only 22 23 24` still walks the whole journey but saves only the screenshots whose names start
with those prefixes (the others on disk stay untouched).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
ONLY: tuple[str, ...] = ()


def shot(page: Page, name: str, full: bool = True) -> None:
    if ONLY and not name.startswith(ONLY):
        return
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(500)
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
    print("saved", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:3000")
    ap.add_argument("--admin-email", default="admin@example.com")
    ap.add_argument("--admin-password", default="admin-pass-123")
    ap.add_argument(
        "--channel", default="chrome", help="playwright browser channel ('' for bundled chromium)"
    )
    ap.add_argument("--only", nargs="*", default=[], help="save only names with these prefixes")
    args = ap.parse_args()
    global ONLY
    ONLY = tuple(args.only)
    OUT.mkdir(parents=True, exist_ok=True)
    base = args.base.rstrip("/")
    email = f"viewer{int(time.time())}@example.com"

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel or None, headless=True)
        ctx = browser.new_context(viewport={"width": 1366, "height": 768}, color_scheme="dark")
        page = ctx.new_page()
        page.goto(f"{base}/")
        shot(page, "01-landing")

        page.goto(f"{base}/register")
        page.fill("#display_name", "Ana")
        page.fill("#email", email)
        page.fill("#password", "a-long-password")
        page.click("button[type=submit]")
        page.wait_for_url("**/onboarding")
        for g in ("Sci-Fi", "Thriller", "Crime"):
            page.get_by_role("button", name=g, exact=False).first.click()
        shot(page, "02-onboarding-genres", full=False)
        page.get_by_role("button", name="Continue").click()
        page.wait_for_selector("button[aria-pressed]")
        for title in ("The Matrix", "Inception", "Pulp Fiction"):
            btn = page.get_by_role("button", name=f"Select {title}")
            if btn.count():
                btn.first.click()
        shot(page, "03-onboarding-favourites", full=False)
        page.get_by_role("button", name="Continue").click()
        stars = page.get_by_role("button", name="4.5 stars")
        for i in range(min(3, stars.count())):
            stars.nth(i).click()
            page.wait_for_timeout(150)
        page.get_by_role("button", name="See my programme").click()
        page.wait_for_url("**/home")
        shot(page, "04-home")

        page.goto(f"{base}/recommendations")
        page.wait_for_selector("ol li")
        shot(page, "05-recommendations", full=False)
        page.get_by_role("button", name="Why this?").first.click()
        page.wait_for_selector("[role=dialog]")
        shot(page, "06-why-this", full=False)
        page.keyboard.press("Escape")

        page.goto(f"{base}/movies/79132")
        page.wait_for_selector("h1")
        shot(page, "07-movie-detail")
        page.goto(f"{base}/profile")
        page.wait_for_selector("h1")
        shot(page, "08-taste-profile")
        page.goto(f"{base}/discover?genre=Film-Noir&sort=rating")
        shot(page, "09-discover", full=False)

        # admin
        page.goto(f"{base}/")
        ctx.clear_cookies()
        page.goto(f"{base}/login")
        page.fill("#email", args.admin_email)
        page.fill("#password", args.admin_password)
        page.click("button[type=submit]")
        page.wait_for_url("**/home")
        page.goto(f"{base}/admin")
        shot(page, "10-admin-overview")
        page.goto(f"{base}/admin/models")
        shot(page, "11-admin-models")
        page.goto(f"{base}/admin/experiments")
        shot(page, "12-admin-experiments")

        # intelligence console (needs at least one pipeline run; the API starts one on boot)
        for name, path in (
            ("14-intel-overview", ""),
            ("15-intel-trends", "/trends"),
            ("16-intel-predictions", "/predictions"),
            ("17-intel-warning", "/warnings/1"),
            ("18-intel-decision", "/decisions/1"),
            ("20-intel-evaluation", "/evaluation"),
        ):
            page.goto(f"{base}/intel{path}")
            shot(page, name)
        page.goto(f"{base}/intel/scenarios")
        page.get_by_role("button", name="Run scenarios").click()
        page.wait_for_timeout(1500)
        shot(page, "19-intel-scenarios")

        # v1.1 pages: evidence search, recommender monitoring, audit log
        page.goto(f"{base}/intel/evidence")
        page.get_by_label("Search evidence").fill("horror")
        page.wait_for_timeout(800)
        shot(page, "22-intel-evidence", full=False)
        page.goto(f"{base}/intel/recommendations")
        shot(page, "23-intel-recommender", full=False)
        page.goto(f"{base}/intel/audit")
        shot(page, "24-intel-audit", full=False)

        # small screens: phone and light theme
        m = browser.new_context(
            viewport={"width": 390, "height": 844}, device_scale_factor=2, color_scheme="light"
        )
        mp = m.new_page()
        mp.goto(f"{base}/")
        mp.evaluate("localStorage.setItem('theme','light')")
        mp.goto(f"{base}/discover")
        shot(mp, "13-mobile-discover-light", full=False)
        mp.goto(f"{base}/login")
        mp.fill("#email", args.admin_email)
        mp.fill("#password", args.admin_password)
        mp.click("button[type=submit]")
        mp.wait_for_url("**/home")
        mp.goto(f"{base}/intel")
        shot(mp, "21-mobile-intel-light", full=False)
        browser.close()


if __name__ == "__main__":
    main()
