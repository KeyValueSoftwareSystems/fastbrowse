"""Browser actions used by the agent graph.

Each function takes a Playwright ``Page`` directly so graph nodes can call them
without tool-call plumbing.
"""

from __future__ import annotations

import json
from typing import Literal

from playwright.async_api import Locator, Page

from fastbrowse.logging import get_logger

logger = get_logger("tools")


def _locator(page: Page, selector: str, frame_index: int = 0) -> Locator:
    """Resolve a locator, piercing into the given frame when frame_index > 0."""
    if frame_index and frame_index < len(page.frames):
        return page.frames[frame_index].locator(selector).first
    return page.locator(selector).first


async def goto(page: Page, url: str) -> str:
    """Navigate `page` to `url` and wait for load."""
    logger.debug("goto(%s)", url)
    await page.goto(url, wait_until="load")
    return f"Navigated to {page.url} | title: {await page.title()}"


async def page_text(page: Page) -> str:
    """Return the visible text of `page`."""
    logger.debug("page_text()")
    return await page.inner_text("body")


async def page_url(page: Page) -> str:
    """Return `page`'s URL and title."""
    logger.debug("page_url()")
    return f"{page.url}\n{await page.title()}"


async def click(page: Page, selector: str, frame_index: int = 0) -> str:
    """Click the first element matching `selector` in `page`."""
    logger.debug("click(%s, frame_index=%d)", selector, frame_index)
    await _locator(page, selector, frame_index).click()
    return f"Clicked: {selector}"


async def fill(page: Page, selector: str, value: str, frame_index: int = 0) -> str:
    """Fill the field matching `selector` with `value`."""
    logger.debug("fill(%s, frame_index=%d)", selector, frame_index)
    await _locator(page, selector, frame_index).fill(value)
    return f"Filled {selector} with '{value}'"


async def scroll(page: Page, delta_y: int) -> str:
    """Scroll `page` vertically by `delta_y` pixels."""
    logger.debug("scroll(%d)", delta_y)
    await page.evaluate("(dy) => window.scrollBy(0, dy)", delta_y)
    return f"Scrolled by {delta_y}px"


async def take_screenshot(page: Page) -> str:
    """Take a full-page screenshot and return its file path."""
    logger.debug("take_screenshot()")
    path = "/tmp/fastbrowse_screenshot.png"
    await page.screenshot(path=path, full_page=True)
    return f"Screenshot saved to {path}"


async def read_cookies(page: Page) -> str:
    """Return all cookies for the current page as JSON."""
    logger.debug("read_cookies()")
    cookies = await page.context.cookies()
    return json.dumps(cookies, ensure_ascii=False)


async def write_cookie(page: Page, name: str, value: str, url: str) -> str:
    """Set a cookie `name=value` for `url` on the current session."""
    logger.debug("write_cookie(%s, url=%s)", name, url)
    await page.context.add_cookies([{"name": name, "value": value, "url": url}])
    return f"Set cookie {name}={value} for {url}"


async def send_request(
    page: Page,
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"],
    url: str,
    body: str | None = None,
) -> str:
    """Issue an HTTP `method` request to `url` using the page's session."""
    logger.debug("send_request(%s, %s)", method, url)
    resp = await page.request.fetch(url, method=method, data=body or None)
    return f"{resp.status} {resp.status_text}\n{await resp.text()}"


async def read_local_storage(page: Page) -> str:
    """Return the current origin's localStorage as JSON."""
    logger.debug("read_local_storage()")
    data = await page.evaluate("() => JSON.stringify(localStorage)")
    return str(data)


async def write_local_storage(page: Page, key: str, value: str) -> str:
    """Set `key=value` in the current origin's localStorage."""
    logger.debug("write_local_storage(%s)", key)
    await page.evaluate("([k, v]) => localStorage.setItem(k, v)", [key, value])
    return f"Set localStorage {key}={value}"
