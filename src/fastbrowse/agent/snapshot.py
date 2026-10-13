from __future__ import annotations

import json
from typing import Any, cast

from playwright.async_api import Frame, Locator, Page
from pydantic import BaseModel

from fastbrowse.logging import get_logger

logger = get_logger("snapshot")

_INTERACTIVE_SELECTOR = "a, button, input, textarea, select, [role='button'], [role='link'], [role='textbox'], [role='checkbox'], [role='tab'], [role='menuitem'], [contenteditable='true']"

_SNAPSHOT_SCRIPT = f"""
() => {{
  const sel = {json.dumps(_INTERACTIVE_SELECTOR)};
  const nodes = Array.from(document.querySelectorAll(sel));
  function cssPath(el) {{
    if (el.id) return '#' + CSS.escape(el.id);
    const tid = el.getAttribute('data-testid');
    if (tid) return '[data-testid="' + tid + '"]';
    const parts = [];
    let cur = el;
    while (cur && cur.nodeType === 1 && parts.length < 6) {{
      let part = cur.tagName.toLowerCase();
      const parent = cur.parentElement;
      if (parent) {{
        const sibs = Array.from(parent.children).filter(c => c.tagName === cur.tagName);
        if (sibs.length > 1) part += ':nth-of-type(' + (sibs.indexOf(cur) + 1) + ')';
      }}
      parts.unshift(part);
      cur = cur.parentElement;
    }}
    return parts.join(' > ');
  }}
  return nodes.map((el) => ({{
    tag: el.tagName.toLowerCase(),
    selector: cssPath(el),
    text: (
      el.innerText ||
      el.value ||
      el.getAttribute('placeholder') ||
      el.getAttribute('aria-label') ||
      ''
    ).trim().slice(0, 200) || null,
    role: el.getAttribute('role'),
  }}));
}}
"""


class Element(BaseModel):
    """A single interactive element captured from a frame of the page."""

    index: int
    frame_index: int = 0
    frame_url: str | None = None
    tag: str
    selector: str
    text: str | None = None
    role: str | None = None


async def take_snapshot(page: Page) -> list[Element]:
    """Collect interactive elements across every frame, tagging each with its frame.

    Elements get a flat global ``index``; ``frame_index`` identifies which
    ``page.frames`` entry the element belongs to so actions can target the right
    frame (including inside iframes).
    """
    elements: list[Element] = []
    for frame_index, frame in enumerate(page.frames):
        raw = await _snapshot_frame(frame)
        logger.debug("frame %d (%s) -> %d elements", frame_index, frame.url, len(raw))
        for item in raw:
            elements.append(
                Element(
                    index=len(elements),
                    frame_index=frame_index,
                    frame_url=frame.url,
                    **item,
                )
            )
    logger.info(
        "snapshot captured %d elements across %d frames",
        len(elements),
        len(page.frames),
    )
    return elements


async def _snapshot_frame(frame: Frame) -> list[dict[str, Any]]:
    try:
        raw = await frame.evaluate(_SNAPSHOT_SCRIPT)
        return cast("list[dict[str, Any]]", raw)
    except Exception:  # noqa: BLE001 - skip frames that cannot be evaluated
        return []


def frame_locator(page: Page, element: Element) -> Locator:
    """Return a Playwright locator for the element's selector within its frame."""
    if element.frame_index and element.frame_index < len(page.frames):
        return page.frames[element.frame_index].locator(element.selector)
    return page.locator(element.selector)
