from __future__ import annotations

from typing import TypedDict

from langgraph.runtime import get_runtime
from playwright.async_api import Browser, Page, Playwright, async_playwright

from fastbrowse.logging import get_logger

logger = get_logger("context")


class BrowserContext(TypedDict):
    """Run-scoped LangGraph runtime context holding the Playwright browser handles.

    These are *not* persisted in state — they live for a single graph run and
    are injected into nodes via ``runtime.context``.
    """

    browser: Browser
    page: Page
    playwright: Playwright


async def build_context(*, headless: bool = True) -> BrowserContext:
    """Launch Playwright and build the runtime context (browser + fresh page)."""
    logger.info("launching browser (headless=%s)", headless)
    playwright = await async_playwright().start()
    try:
        browser = await playwright.chromium.launch(headless=headless)
        page = await browser.new_page()
    except Exception:
        await playwright.stop()
        raise
    logger.info("browser ready")
    return {"browser": browser, "page": page, "playwright": playwright}


async def destroy_context(context: BrowserContext) -> None:
    """Close the browser and stop the Playwright driver to tear down the context."""
    browser = context.get("browser")
    if browser is not None:
        await browser.close()
    playwright = context.get("playwright")
    if playwright is not None:
        await playwright.stop()
    logger.info("browser closed")


def current() -> BrowserContext:
    """Fetch the runtime context from inside a node / tool."""
    return get_runtime(BrowserContext).context