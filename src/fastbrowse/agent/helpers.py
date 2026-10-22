"""Non-node helpers for the browser agent.

Planning, element selection, LLM text generation, and action execution live here
so ``agent.py`` stays focused on node definitions, graph assembly, and the run
entry point.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_typesafe import Choice, ClassifierResponse
from playwright.async_api import Page
from pydantic import BaseModel

from fastbrowse.adapter.llm import llm
from fastbrowse.agent.prompt import (
    ARG_SYSTEM,
    COOKIE_NAME_INSTRUCTION,
    COOKIE_VALUE_INSTRUCTION,
    FILL_SYSTEM,
    PLAN_GOAL_INSTRUCTION,
    PLAN_SYSTEM,
    REQUEST_BODY_INSTRUCTION,
    REQUEST_URL_INSTRUCTION,
    STORAGE_KEY_INSTRUCTION,
    STORAGE_VALUE_INSTRUCTION,
)
from fastbrowse.agent.snapshot import Element
from fastbrowse.agent.state import State
from fastbrowse.agent.tools import (
    clear_cookies,
    clear_storage,
    delete_cookie,
    failure,
    goto,
    read_cookies,
    read_storage,
    remove_storage,
    send_request,
    write_cookie,
    write_storage,
)
from fastbrowse.agent.utils import classify, summarize
from fastbrowse.logging import get_logger

logger = get_logger("helpers")

# Decision thresholds for the planning loop.
UPDATE_T = 0.6  # noul probability above which the plan is rewritten by the LLM

# Loop guards: bound the number of act cycles and detect pages that stopped changing.
MAX_STEPS = 25  # hard cap on act cycles before answering with what we have
STALL_LIMIT = 2  # consecutive no-change snapshots before scroll is withdrawn
MAX_STALL = 4  # consecutive no-change snapshots before forcing the final answer
FAILURE_LIMIT = 2  # consecutive failed actions before the plan is rewritten
MAX_FAILURES = 4  # consecutive failed actions before forcing the final answer

# Reserved Choice key the goal selector returns once every plan item is done.
COMPLETE = "✓ complete"

MAX_TARGETS = 100

HTTP_METHOD = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]


class PlanUpdate(BaseModel):
    plan: list[str]


async def navigate_to(page: Page, url: str) -> str:
    """Navigate to `url`, recording a navigation error as an observation."""
    try:
        return f"navigate -> {await goto(page, url)}"
    except Exception as exc:  # noqa: BLE001 - keep the loop alive on nav errors
        return failure("navigate", url, exc)


def snapshot_change(state: State, url: str, elements: list[Element]) -> str | None:
    """Describe how the page changed since the last snapshot, or ``None`` if unchanged."""
    prev_url = state.get("current_url", "")
    prev_keys = {(e.frame_index, e.selector) for e in state.get("elements", [])}
    new_keys = {(e.frame_index, e.selector) for e in elements}
    if not prev_keys:
        return f"snapshot -> {len(elements)} elements at {url}"

    url_changed = bool(prev_url) and url != prev_url
    if not url_changed and new_keys == prev_keys:
        return None

    added = len(new_keys - prev_keys)
    removed = len(prev_keys - new_keys)
    parts: list[str] = []
    if url_changed:
        parts.append(f"url {prev_url} -> {url}")
    parts.append(f"{len(elements)} elements (+{added}/-{removed})")
    return f"snapshot -> page changed: {', '.join(parts)}"


async def generate_plan(state: State) -> list[str]:
    """Build a fresh ordered plan from the task, page, and observations (LLM)."""
    elements = state.get("elements", [])
    update = cast(
        PlanUpdate,
        await llm().with_structured_output(PlanUpdate).ainvoke(
            [
                SystemMessage(content=PLAN_SYSTEM),
                HumanMessage(
                    content=(
                        f"Task: {state.get('prompt', '')}\n"
                        f"Current goal: {state.get('current_goal', '')}\n"
                        f"Current plan: {state.get('plan', [])}\n"
                        f"Page snapshot:\n{state.get('snapshot', '')[:4000]}\n"
                        f"Interactive elements:\n"
                        f"{[summarize(e) for e in elements[:100]]}\n"
                        f"Observations:\n{state.get('observations', '')}"
                    )
                ),
            ]
        ),
    )
    logger.info("plan: generated %d items", len(update.plan))
    return update.plan


def goal_criteria(plan_items: list[str]) -> dict[str, Any]:
    """Goal-selector criteria keyed by unique plan item text, plus a completion key."""
    return {
        **{item: f"Pursue: {item}" for item in dict.fromkeys(plan_items)},
        COMPLETE: "All plan items are done and the task is complete.",
    }


async def select_goal(state: State, plan_items: list[str]) -> str:
    """Ask jev which plan item is the current goal; "" when done or no plan."""
    items = list(dict.fromkeys(plan_items))
    if not items:
        return ""
    chosen = await classify(
        "plan_goal",
        state,
        "goal",
        Choice(instructions=PLAN_GOAL_INSTRUCTION, criteria=goal_criteria(items)),
        extra={"plan": items},
    )
    return "" if chosen == COMPLETE else chosen


def targets(elements: list[Element], kind: str) -> dict[str, Any]:
    """Index up to `MAX_TARGETS` elements of `kind`, keyed by element index."""
    selected = [element for element in elements if element.kind == kind][:MAX_TARGETS]
    return {str(element.index): summarize(element) for element in selected}


def element_by_index(elements: list[Element], chosen: str) -> Element:
    """Return the element whose index matches a `Choice` key."""
    index = int(chosen)
    return next(element for element in elements if element.index == index)


async def fill_text(state: State, element: Element) -> str:
    response = await llm().ainvoke(
        [
            SystemMessage(content=FILL_SYSTEM),
            HumanMessage(
                content=(
                    f"Task: {state.get('prompt', '')}\n"
                    f"Goal: {state.get('current_goal', '')}\n"
                    f"Field: {element.model_dump()}"
                )
            ),
        ]
    )
    return str(response.content)


async def generate(state: State, instruction: str) -> str:
    """Generate a single argument value for a tool action (plain text, LLM)."""
    response = await llm().ainvoke(
        [
            SystemMessage(content=ARG_SYSTEM),
            HumanMessage(
                content=(
                    f"Task: {state.get('prompt', '')}\n"
                    f"Goal: {state.get('current_goal', '')}\n"
                    f"Current URL: {state.get('current_url', '')}\n"
                    f"Observations:\n{state.get('observations', '')}\n"
                    f"{instruction}"
                )
            ),
        ]
    )
    return str(response.content).strip()


async def run_cookie(state: State, page: Page, response: ClassifierResponse) -> str:
    """Execute the chosen cookie operation and return an observation."""
    action = str(response.choices["cookie_action"].choice)
    if action == "read":
        return f"cookie -> {await read_cookies(page)}"
    if action == "clear":
        return f"cookie -> {await clear_cookies(page)}"

    name = await generate(state, COOKIE_NAME_INSTRUCTION)
    if action == "set":
        value = await generate(state, COOKIE_VALUE_INSTRUCTION)
        return f"cookie -> {await write_cookie(page, name, value, page.url)}"
    return f"cookie -> {await delete_cookie(page, name)}"


async def run_storage(state: State, page: Page, response: ClassifierResponse) -> str:
    """Execute the chosen storage operation and return an observation."""
    action = str(response.choices["storage_action"].choice)
    operation, _, which_name = action.partition("_")
    which: Literal["local", "session"] = "local" if which_name == "local" else "session"

    if operation == "read":
        return f"storage -> {await read_storage(page, which)}"
    if operation == "clear":
        return f"storage -> {await clear_storage(page, which)}"

    key = await generate(state, STORAGE_KEY_INSTRUCTION)
    if operation == "set":
        value = await generate(state, STORAGE_VALUE_INSTRUCTION)
        return f"storage -> {await write_storage(page, which, key, value)}"
    return f"storage -> {await remove_storage(page, which, key)}"


async def run_request(state: State, page: Page, response: ClassifierResponse) -> str:
    """Execute the chosen HTTP request and return an observation."""
    method = cast(HTTP_METHOD, str(response.choices["request_method"].choice))
    url = await generate(state, REQUEST_URL_INSTRUCTION)
    body = None
    if method in {"POST", "PUT", "PATCH"}:
        body = await generate(state, REQUEST_BODY_INSTRUCTION)
    return f"request -> {await send_request(page, method, url, body)}"
