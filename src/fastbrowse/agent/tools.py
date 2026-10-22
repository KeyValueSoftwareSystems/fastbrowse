"""Browser actions used by the agent graph.

Each function takes a Playwright ``Page`` directly so graph nodes can call them
without tool-call plumbing.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from playwright.async_api import Locator, Page

from fastbrowse.logging import get_logger

logger = get_logger("tools")

ACTION_TIMEOUT = 5000  # ms an action may take before it is treated as failed

# Marker the graph uses to spot a failed action in an observation.
FAILURE_MARKER = " FAILED: "


def failure(action: str, target: str, exc: Exception) -> str:
    """Format an action failure as an observation instead of raising."""
    lines = str(exc).strip().splitlines()
    reason = lines[0] if lines else type(exc).__name__
    return f"{action}{FAILURE_MARKER}{target} ({reason})"


def is_failure(observation: str) -> bool:
    """Whether an observation produced by a browser action records a failure."""
    return FAILURE_MARKER in observation


async def _locator(page: Page, selector: str, frame_index: int = 0) -> Locator:
    """Resolve a locator, piercing into the given frame when frame_index > 0.

    Duplicate selectors are common (hidden cookie banners reuse ids from their
    visible modal counterparts), so an ambiguous selector resolves to the first
    *visible* match rather than blindly to the first DOM node.
    """
    if frame_index and frame_index < len(page.frames):
        base = page.frames[frame_index].locator(selector)
    else:
        base = page.locator(selector)
    try:
        count = await base.count()
    except Exception:  # noqa: BLE001 - fall back to the plain first match
        return base.first
    if count > 1:
        for index in range(min(count, 10)):
            candidate = base.nth(index)
            try:
                if await candidate.is_visible():
                    return candidate
            except Exception:  # noqa: BLE001 - detached nodes are not matches
                continue
    return base.first


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
    """Click the first visible element matching `selector` in `page`.

    A click that cannot be performed is reported as a failure observation so the
    agent loop keeps running instead of aborting on a bad element.
    """
    logger.debug("click(%s, frame_index=%d)", selector, frame_index)
    locator = await _locator(page, selector, frame_index)
    try:
        await locator.click(timeout=ACTION_TIMEOUT)
        return f"Clicked: {selector}"
    except Exception:  # noqa: BLE001 - retry with the actionability checks relaxed
        pass
    try:
        await locator.scroll_into_view_if_needed(timeout=ACTION_TIMEOUT)
        await locator.click(timeout=ACTION_TIMEOUT, force=True)
        return f"Clicked: {selector} (forced)"
    except Exception:  # noqa: BLE001 - last resort for covered or animated nodes
        pass
    try:
        await locator.dispatch_event("click", timeout=ACTION_TIMEOUT)
        return f"Clicked: {selector} (dispatched)"
    except Exception as exc:  # noqa: BLE001 - report instead of raising
        return failure("click", selector, exc)


async def fill(page: Page, selector: str, value: str, frame_index: int = 0) -> str:
    """Fill the field matching `selector` with `value`, reporting failures.

    Falls back to focusing the field and typing when a direct fill is rejected
    (masked inputs, custom editors, fields that only accept keystrokes).
    """
    logger.debug("fill(%s, frame_index=%d)", selector, frame_index)
    locator = await _locator(page, selector, frame_index)
    try:
        await locator.fill(value, timeout=ACTION_TIMEOUT)
        return f"Filled {selector} with '{value}'"
    except Exception:  # noqa: BLE001 - retry by typing into the field
        pass
    try:
        await locator.focus(timeout=ACTION_TIMEOUT)
        await locator.press_sequentially(value, timeout=ACTION_TIMEOUT)
        return f"Filled {selector} with '{value}' (typed)"
    except Exception as exc:  # noqa: BLE001 - report instead of raising
        return failure("fill", selector, exc)


# Key names accepted verbatim by Playwright's ``keyboard.press``.
_KEY_NAMES = {
    "enter": "Enter",
    "return": "Enter",
    "backspace": "Backspace",
    "delete": "Delete",
    "tab": "Tab",
    "escape": "Escape",
    "space": "Space",
    "arrowup": "ArrowUp",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
}


async def type_text(page: Page, text: str) -> str:
    """Type `text` on the page keyboard.

    Sends real keyboard events, which is the only way to drive widgets that
    listen for keydown on the document (games, canvas editors) rather than
    reading a form field.
    """
    logger.debug("type_text(%r)", text)
    value = text.strip()
    if not value:
        return failure("type", repr(text), ValueError("no text generated"))
    try:
        await page.keyboard.type(value)
        return f"Typed '{value}'"
    except Exception as exc:  # noqa: BLE001 - report instead of raising
        return failure("type", value, exc)


async def press_key(page: Page, key: str) -> str:
    """Press a single key (``Enter``, ``Backspace``, ...) on the page keyboard."""
    logger.debug("press_key(%r)", key)
    value = key.strip()
    if not value:
        return failure("key", repr(key), ValueError("no key generated"))
    canonical = _KEY_NAMES.get(value.lower(), value)
    try:
        await page.keyboard.press(canonical)
        return f"Pressed {canonical}"
    except Exception as exc:  # noqa: BLE001 - report instead of raising
        return failure("key", value, exc)


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


async def write_cookie(
    page: Page,
    name: str,
    value: str,
    url: str | None = None,
    *,
    domain: str | None = None,
    path: str | None = None,
    expires: float | None = None,
    http_only: bool | None = None,
    secure: bool | None = None,
    same_site: Literal["Strict", "Lax", "None"] | None = None,
) -> str:
    """Set a cookie `name=value`, either scoped to `url` or `domain`/`path`."""
    logger.debug("write_cookie(%s, url=%s, domain=%s)", name, url, domain)
    cookie: dict[str, Any] = {"name": name, "value": value}
    if url is not None:
        cookie["url"] = url
    if domain is not None:
        cookie["domain"] = domain
    if path is not None:
        cookie["path"] = path
    if expires is not None:
        cookie["expires"] = expires
    if http_only is not None:
        cookie["httpOnly"] = http_only
    if secure is not None:
        cookie["secure"] = secure
    if same_site is not None:
        cookie["sameSite"] = same_site
    await page.context.add_cookies([cookie])  # pyright: ignore[reportArgumentType]
    return f"Set cookie {name}={value}"


async def delete_cookie(
    page: Page,
    name: str,
    *,
    domain: str | None = None,
    path: str | None = None,
) -> str:
    """Delete cookies named `name` (optionally scoped by `domain`/`path`)."""
    logger.debug("delete_cookie(%s)", name)
    cookies = await page.context.cookies()
    keep = [
        cookie
        for cookie in cookies
        if not (
            cookie.get("name") == name
            and (domain is None or cookie.get("domain") == domain)
            and (path is None or cookie.get("path") == path)
        )
    ]
    removed = len(cookies) - len(keep)
    await page.context.clear_cookies()
    if keep:
        await page.context.add_cookies(keep)  # pyright: ignore[reportArgumentType]
    return f"Deleted {removed} cookie(s) named {name}"


async def clear_cookies(page: Page) -> str:
    """Delete every cookie in the browser context."""
    logger.debug("clear_cookies()")
    await page.context.clear_cookies()
    return "Cleared all cookies"


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


_STORAGE_ATTR = {"local": "localStorage", "session": "sessionStorage"}


async def read_storage(page: Page, which: Literal["local", "session"] = "local") -> str:
    """Return the current origin's `which` storage as JSON."""
    logger.debug("read_storage(%s)", which)
    data = await page.evaluate(
        "(store) => JSON.stringify(window[store])", _STORAGE_ATTR[which]
    )
    return str(data)


async def write_storage(
    page: Page,
    which: Literal["local", "session"],
    key: str,
    value: str,
) -> str:
    """Set `key=value` in the current origin's `which` storage."""
    logger.debug("write_storage(%s, %s)", which, key)
    await page.evaluate(
        "([store, k, v]) => window[store].setItem(k, v)",
        [_STORAGE_ATTR[which], key, value],
    )
    return f"Set {which}Storage {key}={value}"


async def remove_storage(
    page: Page,
    which: Literal["local", "session"],
    key: str,
) -> str:
    """Remove `key` from the current origin's `which` storage."""
    logger.debug("remove_storage(%s, %s)", which, key)
    await page.evaluate(
        "([store, k]) => window[store].removeItem(k)",
        [_STORAGE_ATTR[which], key],
    )
    return f"Removed {which}Storage {key}"


async def clear_storage(page: Page, which: Literal["local", "session"] = "local") -> str:
    """Remove every entry from the current origin's `which` storage."""
    logger.debug("clear_storage(%s)", which)
    await page.evaluate("(store) => window[store].clear()", _STORAGE_ATTR[which])
    return f"Cleared {which}Storage"
