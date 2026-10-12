from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from fastbrowse.agent.context import BrowserContext
from fastbrowse.agent.state import State


@tool
async def navigate(url: str, runtime: ToolRuntime[BrowserContext, State]) -> str:
    """Navigate the browser to `url` and wait for the page to load."""
    page = runtime.context["page"]
    await page.goto(url, wait_until="load")
    return f"Navigated to {page.url} | title: {await page.title()}"


@tool
async def get_page_text(runtime: ToolRuntime[BrowserContext, State]) -> str:
    """Return the visible text of the current page."""
    return await runtime.context["page"].inner_text("body")


@tool
async def get_page_url(runtime: ToolRuntime[BrowserContext, State]) -> str:
    """Return the current page URL and title."""
    page = runtime.context["page"]
    return f"{page.url}\n{await page.title()}"


def _locator(page, selector: str, frame_index: int = 0):
    """Resolve a locator, piercing into the given frame when frame_index > 0."""
    if frame_index and frame_index < len(page.frames):
        return page.frames[frame_index].locator(selector).first
    return page.locator(selector).first


@tool
async def click_element(
    selector: str, runtime: ToolRuntime[BrowserContext, State], frame_index: int = 0
) -> str:
    """Click the first element matching the CSS `selector` (optionally inside `frame_index`)."""
    page = runtime.context["page"]
    await _locator(page, selector, frame_index).click()
    return f"Clicked: {selector}"


@tool
async def fill_input(
    selector: str,
    value: str,
    runtime: ToolRuntime[BrowserContext, State],
    frame_index: int = 0,
) -> str:
    """Fill the form field matching the CSS `selector` (optionally inside `frame_index`) with `value`."""
    page = runtime.context["page"]
    await _locator(page, selector, frame_index).fill(value)
    return f"Filled {selector} with '{value}'"


@tool
async def scroll_page(delta_y: int, runtime: ToolRuntime[BrowserContext, State]) -> str:
    """Scroll the page vertically by `delta_y` pixels (positive scrolls down)."""
    page = runtime.context["page"]
    await page.evaluate("(dy) => window.scrollBy(0, dy)", delta_y)
    return f"Scrolled by {delta_y}px"


@tool
async def screenshot(runtime: ToolRuntime[BrowserContext, State]) -> str:
    """Take a full-page screenshot and return its file path."""
    page = runtime.context["page"]
    path = "/tmp/fastbrowse_screenshot.png"
    await page.screenshot(path=path, full_page=True)
    return f"Screenshot saved to {path}"


@tool
async def get_cookies(runtime: ToolRuntime[BrowserContext, State]) -> str:
    """Return all cookies for the current page."""
    cookies = await runtime.context["page"].context.cookies()
    return json.dumps(cookies, ensure_ascii=False)


@tool
async def set_cookie(
    name: str, value: str, url: str, runtime: ToolRuntime[BrowserContext, State]
) -> str:
    """Set a cookie `name=value` for `url` on the current session."""
    await runtime.context["page"].context.add_cookies(
        [{"name": name, "value": value, "url": url}]
    )
    return f"Set cookie {name}={value} for {url}"


@tool
async def api_request(
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"],
    url: str,
    runtime: ToolRuntime[BrowserContext, State],
    body: str | None = None,
) -> str:
    """Issue an HTTP `method` request to `url` (optionally with a JSON `body`) using the page's session."""
    page = runtime.context["page"]
    resp = await page.request.fetch(url, method=method, data=body or None)
    return f"{resp.status} {resp.status_text}\n{await resp.text()}"


@tool
async def get_local_storage(runtime: ToolRuntime[BrowserContext, State]) -> str:
    """Return the current origin's localStorage as JSON."""
    data = await runtime.context["page"].evaluate("() => JSON.stringify(localStorage)")
    return str(data)


@tool
async def set_local_storage(
    key: str, value: str, runtime: ToolRuntime[BrowserContext, State]
) -> str:
    """Set `key=value` in the current origin's localStorage."""
    await runtime.context["page"].evaluate(
        "([k, v]) => localStorage.setItem(k, v)", [key, value]
    )
    return f"Set localStorage {key}={value}"


BROWSER_TOOLS = [
    navigate,
    get_page_text,
    get_page_url,
    click_element,
    fill_input,
    scroll_page,
    screenshot,
    get_cookies,
    set_cookie,
    api_request,
    get_local_storage,
    set_local_storage,
]