from __future__ import annotations

import uuid
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_typesafe import Choice, Noul, NoulCriteria, Question
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from netra import Netra  # type: ignore[import-untyped]
from netra.decorators import workflow  # type: ignore[import-untyped]
from pydantic import BaseModel

from fastbrowse.adapter.llm import llm
from fastbrowse.agent.context import BrowserContext, build_context, destroy_context
from fastbrowse.agent.prompt import (
    ACTION_INSTRUCTION,
    FILL_SYSTEM,
    FIND_ELEMENT_INSTRUCTION,
    NAVIGATE_INSTRUCTION,
    PLAN_ITEM_DONE_INSTRUCTION,
    PLAN_SYSTEM,
    PLAN_UPDATE_INSTRUCTION,
    ROUTER_INSTRUCTION,
)
from fastbrowse.agent.snapshot import Element, take_snapshot
from fastbrowse.agent.state import State
from fastbrowse.agent.tools import click, fill, goto, scroll
from fastbrowse.agent.utils import append, ask, classify, summarize
from fastbrowse.logging import get_logger, setup_logging

logger = get_logger("agent")

# Decision thresholds for the planning loop.
UPDATE_T = 0.6  # noul probability above which the plan is rewritten by the LLM
DONE_T = 0.5  # noul probability above which a plan item counts as completed


class PlanUpdate(BaseModel):
    plan: list[str]
    current_goal: str


@workflow  # type: ignore[untyped-decorator]
async def navigate(state: State, runtime: Runtime[BrowserContext]) -> dict[str, Any]:
    """Navigate to a URL chosen from state, or stay on the current page."""
    page = runtime.context["page"]
    urls = sorted(state.get("urls", set()))
    Netra.set_input({"urls": urls, "current_url": page.url})

    if not urls:
        logger.info("navigate: no urls in state, staying")
    elif len(urls) == 1:
        target = urls[0]
        if page.url != target:
            logger.info("navigate: single url -> %s", target)
            await goto(page, target)
        else:
            logger.info("navigate: already at %s", target)
    else:
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

    Netra.set_output({"url": page.url})
    return {}


@workflow  # type: ignore[untyped-decorator]
async def snapshot(state: State, runtime: Runtime[BrowserContext]) -> dict[str, Any]:
    """Capture an aria snapshot and the interactive elements into state."""
    page = runtime.context["page"]
    Netra.set_input({"url": page.url})
    aria = await page.aria_snapshot()
    elements = await take_snapshot(page)
    logger.info("snapshot: %d aria chars, %d elements", len(aria), len(elements))
    Netra.set_output({"snapshot_chars": len(aria), "elements": len(elements)})
    return {"snapshot": aria, "elements": elements}


async def _generate_plan(state: State) -> dict[str, Any]:
    """Build a fresh plan and current goal from the task and observations (LLM)."""
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


def _next_goal(plan_items: list[str], completed: list[bool]) -> str:
    """Return the item after the latest completed one, or "" if all are done."""
    last = max((i for i, done in enumerate(completed) if done), default=-1)
    nxt = last + 1
    return plan_items[nxt] if nxt < len(plan_items) else ""


@workflow  # type: ignore[untyped-decorator]
async def plan(state: State, runtime: Runtime[BrowserContext]) -> dict[str, Any]:
    """Advance the plan: rewrite it when needed, otherwise move to the next goal."""
    plan_items = state.get("plan", [])
    Netra.set_input(
        {
            "task": state.get("prompt", ""),
            "goal": state.get("current_goal", ""),
            "plan": plan_items,
            "observations": state.get("observations", ""),
        }
    )

    if not plan_items:
        result = await _generate_plan(state)
    else:
        questions: dict[str, Question] = {
            "needs_update": Noul(
                instructions=PLAN_UPDATE_INSTRUCTION,
                criteria=NoulCriteria(
                    true="The plan must be rewritten.",
                    false="The plan still holds.",
                ),
            ),
            **{
                f"done_{i}": Noul(
                    instructions=PLAN_ITEM_DONE_INSTRUCTION.format(item=item),
                    criteria=NoulCriteria(
                        true="The item is completed.",
                        false="The item is not completed.",
                    ),
                )
                for i, item in enumerate(plan_items)
            },
        }
        response = await ask("plan", state, questions)

        if response.nouls["needs_update"].noul >= UPDATE_T:
            logger.info("plan: needs update, rewriting")
            result = await _generate_plan(state)
        else:
            completed = [
                response.nouls[f"done_{i}"].noul >= DONE_T
                for i in range(len(plan_items))
            ]
            goal = _next_goal(plan_items, completed)
            logger.info("plan: completed=%s goal=%r", completed, goal)
            result = {"current_goal": goal}

    Netra.set_output(result)
    return result


@workflow  # type: ignore[untyped-decorator]
async def router(
    state: State, runtime: Runtime[BrowserContext]
) -> Command[str]:
    """Decide the next step and route the graph accordingly."""
    Netra.set_input(
        {
            "goal": state.get("current_goal", ""),
            "plan": state.get("plan", []),
            "observations": state.get("observations", ""),
        }
    )
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
    Netra.set_output({"route": route})
    return Command(goto=route)


@workflow  # type: ignore[untyped-decorator]
async def find_element(
    state: State, runtime: Runtime[BrowserContext]
) -> dict[str, Any]:
    """Binary-search the snapshot for the element most relevant to the current state."""
    elements = state.get("elements", [])
    Netra.set_input(
        {
            "goal": state.get("current_goal", ""),
            "element_count": len(elements),
        }
    )
    selected = await _search_element(state, elements)
    result = {"selected_index": selected}
    Netra.set_output(result)
    return result


async def _search_element(state: State, elements: list[Element]) -> int:
    """Binary-search `elements` for the index most relevant to the current state."""
    if not elements:
        return -1

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
            return -1
        if chosen == "bottom":
            candidates = bottom
        else:
            candidates = top

    if len(candidates) == 1:
        logger.info("find_element: selected index=%d", candidates[0].index)
        return candidates[0].index
    return -1


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


@workflow  # type: ignore[untyped-decorator]
async def action(state: State, runtime: Runtime[BrowserContext]) -> dict[str, Any]:
    """Pick a tool for the selected element, run it, and record an observation."""
    elements = state.get("elements", [])
    index = state.get("selected_index", -1)
    element = next((e for e in elements if e.index == index), None)
    Netra.set_input(
        {
            "goal": state.get("current_goal", ""),
            "selected_index": index,
            "element": summarize(element) if element else None,
        }
    )

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

    result: dict[str, Any]
    if tool_name in ("click", "fill") and element is None:
        note = f"{tool_name} skipped: no element selected"
        logger.info("action: %s", note)
        result = {"observations": append(state.get("observations", ""), note)}
        observation = note
    else:
        tool_result = await _run_action(state, runtime, element, tool_name)
        observation = f"{tool_name} -> {tool_result}"
        logger.info("action: %s", observation)
        result = {
            "observations": append(state.get("observations", ""), observation),
            "messages": [AIMessage(content=observation)],
        }

    Netra.set_output({"observation": observation})
    return result


_checkpointer = InMemorySaver(
    serde=JsonPlusSerializer(allowed_msgpack_modules=[Element])
)

def build_graph() -> CompiledStateGraph[State, BrowserContext, State, State]:
    """Assemble and compile the browser-agent StateGraph."""
    graph = StateGraph(State, context_schema=BrowserContext)

    graph.add_node("navigate", navigate) #type: ignore
    graph.add_node("snapshot", snapshot) #type: ignore
    graph.add_node("plan", plan) #type: ignore
    graph.add_node("router", router) #type: ignore
    graph.add_node("find_element", find_element) #type: ignore
    graph.add_node("action", action) #type: ignore

    graph.add_edge(START, "navigate")
    graph.add_edge("navigate", "snapshot")
    graph.add_edge("snapshot", "plan")
    graph.add_edge("plan", "router")
    graph.add_edge("find_element", "action")
    graph.add_edge("action", "snapshot")

    return graph.compile(checkpointer=_checkpointer)

@workflow(name="Fastbrowse")
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