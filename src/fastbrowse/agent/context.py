from __future__ import annotations

from typing import TypedDict

from langgraph.runtime import get_runtime
from playwright.async_api import Browser, Page, async_playwright


class BrowserContext(TypedDict):
    """Run-scoped LangGraph runtime context holding the Playwright browser handles.

    These are *not* persisted in state — they live for a single graph run and
    are injected into nodes via ``runtime.context``.
    """

    browser: Browser
    page: Page


async def build_context(*, headless: bool = True) -> BrowserContext:
    """Launch Playwright and build the runtime context (browser + fresh page)."""
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=headless)
    page = await browser.new_page()
    return {"browser": browser, "page": page}


async def destroy_context(context: BrowserContext) -> None:
    """Close the browser to tear down the context."""
    browser = context.get("browser")
    if browser is not None:
        await browser.close()


def current() -> BrowserContext:
    """Fetch the runtime context from inside a node / tool."""
    return get_runtime(BrowserContext).context