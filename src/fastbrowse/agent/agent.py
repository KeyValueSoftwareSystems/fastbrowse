from __future__ import annotations

import uuid
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_typesafe import Choice
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

from fastbrowse.adapter.llm import llm
from fastbrowse.agent.context import BrowserContext, build_context, destroy_context
from fastbrowse.agent.prompt import (
    ACTION_INSTRUCTION,
    FILL_SYSTEM,
    FIND_ELEMENT_INSTRUCTION,
    NAVIGATE_INSTRUCTION,
    PLAN_SYSTEM,
    ROUTER_INSTRUCTION,
)
from fastbrowse.agent.snapshot import Element, take_snapshot
from fastbrowse.agent.state import State
from fastbrowse.agent.tools import click, fill, goto, scroll
from fastbrowse.agent.utils import append, classify, summarize
from fastbrowse.logging import get_logger, setup_logging

logger = get_logger("agent")


class PlanUpdate(BaseModel):
    plan: list[str]
    current_goal: str


async def navigate(state: State, runtime: Runtime[BrowserContext]) -> dict[str, Any]:
    """Navigate to a URL chosen from state, or stay on the current page."""
    page = runtime.context["page"]
    urls = sorted(state.get("urls", set()))
    if not urls:
        logger.info("navigate: no urls in state, staying")
        return {}

    if len(urls) == 1:
        target = urls[0]
        if page.url != target:
            logger.info("navigate: single url -> %s", target)
            await goto(page, target)
        else:
            logger.info("navigate: already at %s", target)
        return {}

    chosen = await classify(
        "navigate",
        state,
        "url",
        Choice(
            instructions=NAVIGATE_INSTRUCTION,
            criteria={
                **{url: f"Navigate to {url}" for url in urls},
                "none": "Stay on the current page.",
            },
        ),
        extra={"urls": urls},
    )
    if chosen != "none" and page.url != chosen:
        logger.info("navigate: -> %s", chosen)
        await goto(page, chosen)
    else:
        logger.info("navigate: staying on %s", page.url)
    return {}


async def snapshot(state: State, runtime: Runtime[BrowserContext]) -> dict[str, Any]:
    """Capture an aria snapshot and the interactive elements into state."""
    page = runtime.context["page"]
    aria = await page.aria_snapshot()
    elements = await take_snapshot(page)
    logger.info(
        "snapshot: %d aria chars, %d elements", len(aria), len(elements)
    )
    return {"snapshot": aria, "elements": elements}


async def plan(state: State, runtime: Runtime[BrowserContext]) -> dict[str, Any]:
    """(Re)build the plan and current goal from the task and observations."""
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
                        f"Observations:\n{state.get('observations', '')}"
                    )
                ),
            ]
        ),
    )
    logger.info("plan: goal=%r plan=%r", update.current_goal, update.plan)
    return {"plan": update.plan, "current_goal": update.current_goal}


async def router(
    state: State, runtime: Runtime[BrowserContext]
) -> Command[str]:
    """Decide the next step and route the graph accordingly."""
    route = await classify(
        "router",
        state,
        "route",
        Choice(
            instructions=ROUTER_INSTRUCTION,
            criteria={
                "find_element": (
                    "Locate the most relevant element on the current page and "
                    "act on it."
                ),
                "navigate": "Go back and navigate to a different URL.",
                "__end__": "The goal has been achieved; stop.",
            },
        ),
    )
    logger.info("router: -> %s", route)
    return Command(goto=route)


async def find_element(
    state: State, runtime: Runtime[BrowserContext]
) -> dict[str, Any]:
    """Binary-search the snapshot for the element most relevant to the current state."""
    elements = state.get("elements", [])
    if not elements:
        return {"selected_index": -1}

    candidates: list[Element] = list(elements)
    step = 0

    while len(candidates) > 1:
        step += 1
        mid = len(candidates) // 2
        top = candidates[:mid]
        bottom = candidates[mid:]

        chosen = await classify(
            f"find_element[{step}]",
            state,
            "half",
            Choice(
                instructions=FIND_ELEMENT_INSTRUCTION,
                criteria={
                    "none": "No element on the page is relevant; stop.",
                    "top": {
                        "half": "first half",
                        "elements": [summarize(element) for element in top],
                    },
                    "bottom": {
                        "half": "second half",
                        "elements": [summarize(element) for element in bottom],
                    },
                },
            ),
            extra={
                "candidates": [summarize(element) for element in candidates]
            },
        )
        logger.info(
            "find_element: step %d candidates=%d -> %s",
            step,
            len(candidates),
            chosen,
        )
        if chosen == "none":
            logger.info("find_element: no relevant element")
            return {"selected_index": -1}
        if chosen == "bottom":
            candidates = bottom
        else:
            candidates = top

    if len(candidates) == 1:
        logger.info("find_element: selected index=%d", candidates[0].index)
        return {"selected_index": candidates[0].index}
    return {"selected_index": -1}


async def _fill_text(state: State, element: Element) -> str:
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


async def _run_action(
    state: State,
    runtime: Runtime[BrowserContext],
    element: Element | None,
    tool_name: str,
) -> str:
    """Run the chosen tool against the page and return its result string."""
    page = runtime.context["page"]
    if tool_name == "click" and element is not None:
        return await click(page, element.selector, frame_index=element.frame_index)
    if tool_name == "fill" and element is not None:
        value = await _fill_text(state, element)
        return await fill(
            page, element.selector, value, frame_index=element.frame_index
        )
    if tool_name == "scroll":
        return await scroll(page, 600)
    return "No action taken."


async def action(state: State, runtime: Runtime[BrowserContext]) -> dict[str, Any]:
    """Pick a tool for the selected element, run it, and record an observation."""
    elements = state.get("elements", [])
    index = state.get("selected_index", -1)
    element = next((e for e in elements if e.index == index), None)

    tool_name = await classify(
        "action",
        state,
        "tool",
        Choice(
            instructions=ACTION_INSTRUCTION,
            criteria={
                "click": "Click the selected element.",
                "fill": "Type text into the selected element.",
                "scroll": "Scroll the page.",
                "none": "Take no action.",
            },
        ),
        extra={"element": element.model_dump() if element else None},
    )

    if tool_name in ("click", "fill") and element is None:
        note = f"{tool_name} skipped: no element selected"
        logger.info("action: %s", note)
        return {"observations": append(state.get("observations", ""), note)}

    result = await _run_action(state, runtime, element, tool_name)

    observation = f"{tool_name} -> {result}"
    logger.info("action: %s", observation)
    return {
        "observations": append(state.get("observations", ""), observation),
        "messages": [AIMessage(content=observation)],
    }


_checkpointer = InMemorySaver(
    serde=JsonPlusSerializer(allowed_msgpack_modules=[Element])
)


def build_graph() -> CompiledStateGraph[State, BrowserContext, State, State]:
    """Assemble and compile the browser-agent StateGraph."""
    graph = StateGraph(State, context_schema=BrowserContext)

    graph.add_node("navigate", navigate)
    graph.add_node("snapshot", snapshot)
    graph.add_node("plan", plan)
    graph.add_node("router", router)
    graph.add_node("find_element", find_element)
    graph.add_node("action", action)

    graph.add_edge(START, "navigate")
    graph.add_edge("navigate", "snapshot")
    graph.add_edge("snapshot", "plan")
    graph.add_edge("plan", "router")
    graph.add_edge("find_element", "action")
    graph.add_edge("action", "snapshot")

    return graph.compile(checkpointer=_checkpointer)


async def run(prompt: str, url: str, *, headless: bool = True) -> dict[str, Any]:
    """Run the browser agent on a prompt starting at a URL, returning the final state."""
    setup_logging()
    logger.info("run start | url=%s prompt=%r", url, prompt)
    context = await build_context(headless=headless)
    try:
        initial = cast("State", {"prompt": prompt, "urls": {url}})
        result: dict[str, Any] = await build_graph().ainvoke(
            initial,
            config={"configurable": {"thread_id": uuid.uuid4().hex}},
            context=context,
        )
    finally:
        await destroy_context(context)
    logger.info(
        "run end | goal=%r observations=%d chars",
        result.get("current_goal"),
        len(result.get("observations", "")),
    )
    return result