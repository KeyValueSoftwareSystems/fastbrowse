from __future__ import annotations

import uuid
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_typesafe import Choice, Noul, NoulCriteria, Question
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from netra import Netra
from netra.decorators import workflow
from pydantic import BaseModel

from fastbrowse.adapter.llm import llm
from fastbrowse.agent.context import BrowserContext, build_context, destroy_context
from fastbrowse.agent.prompt import (
    ACT_INSTRUCTION,
    ANSWER_SYSTEM,
    CLICK_TARGET_INSTRUCTION,
    FILL_SYSTEM,
    FILL_TARGET_INSTRUCTION,
    NAVIGATE_INSTRUCTION,
    PLAN_ITEM_DONE_INSTRUCTION,
    PLAN_SYSTEM,
    PLAN_UPDATE_INSTRUCTION,
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


@workflow
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


@workflow
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
    logger.info("plan: goal=%r plan=%r", update.current_goal, update.plan)
    return {"plan": update.plan, "current_goal": update.current_goal}


def _next_goal(plan_items: list[str], completed: list[bool]) -> str:
    """Return the item after the latest completed one, or "" if all are done."""
    last = max((i for i, done in enumerate(completed) if done), default=-1)
    nxt = last + 1
    return plan_items[nxt] if nxt < len(plan_items) else ""


@workflow
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


_MAX_TARGETS = 100


def _targets(elements: list[Element], kind: str) -> dict[str, Any]:
    """Index up to `_MAX_TARGETS` elements of `kind`, keyed by element index."""
    selected = [element for element in elements if element.kind == kind][:_MAX_TARGETS]
    return {str(element.index): summarize(element) for element in selected}


def _element_by_index(elements: list[Element], chosen: str) -> Element:
    """Return the element whose index matches a `Choice` key."""
    index = int(chosen)
    return next(element for element in elements if element.index == index)


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


@workflow
async def act(state: State, runtime: Runtime[BrowserContext]) -> Command[str]:
    """Choose one operation and its target in a single request, then execute it."""
    elements = state.get("elements", [])
    click_targets = _targets(elements, "click")
    fill_targets = _targets(elements, "fill")

    criteria: dict[str, Any] = {
        "navigate": "Go back and navigate to a different URL.",
        "scroll": "Scroll the page to reveal more content.",
        "answer": "The task is complete; produce the final answer.",
    }
    if click_targets:
        criteria["click"] = "Click one of the clickable elements."
    if fill_targets:
        criteria["fill"] = "Type text into one of the fields."

    questions: dict[str, Question] = {
        "route": Choice(instructions=ACT_INSTRUCTION, criteria=criteria)
    }
    if click_targets:
        questions["click_target"] = Choice(
            instructions=CLICK_TARGET_INSTRUCTION, criteria=click_targets
        )
    if fill_targets:
        questions["fill_target"] = Choice(
            instructions=FILL_TARGET_INSTRUCTION, criteria=fill_targets
        )

    Netra.set_input(
        {
            "goal": state.get("current_goal", ""),
            "plan": state.get("plan", []),
            "element_count": len(elements),
            "click_targets": len(click_targets),
            "fill_targets": len(fill_targets),
            "criteria": questions["route"]
        }
    )

    response = await ask("act", state, questions)
    route = str(response.choices["route"].choice)
    logger.info("act: route=%s", route)

    if route == "navigate":
        Netra.set_output({"route": route})
        return Command(goto="navigate")
    if route == "answer":
        Netra.set_output({"route": route})
        return Command(goto="answer")

    page = runtime.context["page"]
    if route == "click" and click_targets:
        element = _element_by_index(elements, str(response.choices["click_target"].choice))
        clicked = await click(page, element.selector, frame_index=element.frame_index)
        observation = f"click -> {clicked}"
    elif route == "fill" and fill_targets:
        element = _element_by_index(elements, str(response.choices["fill_target"].choice))
        value = await _fill_text(state, element)
        filled = await fill(page, element.selector, value, frame_index=element.frame_index)
        observation = f"fill -> {filled}"
    else:
        scrolled = await scroll(page, 600)
        observation = f"scroll -> {scrolled}"

    logger.info("act: %s", observation)
    Netra.set_output({"route": route, "observation": observation})
    return Command(
        goto="snapshot",
        update={
            "observations": append(state.get("observations", ""), observation),
            "messages": [AIMessage(content=observation)],
        },
    )


@workflow
async def answer(state: State, runtime: Runtime[BrowserContext]) -> dict[str, Any]:
    """Generate the final answer from the current state, then end the run."""
    Netra.set_input(
        {
            "goal": state.get("current_goal", ""),
            "observations": state.get("observations", ""),
        }
    )
    response = await llm().ainvoke(
        [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(
                content=(
                    f"Goal: {state.get('current_goal', '')}\n"
                    f"Observations:\n{state.get('observations', '')}\n"
                    f"Page text:\n{state.get('snapshot', '')}"
                )
            ),
        ]
    )
    text = str(response.content)
    logger.info("answer: %r", text)
    Netra.set_output({"answer": text})
    return {"answer": text, "messages": [AIMessage(content=text)]}


_checkpointer = InMemorySaver(
    serde=JsonPlusSerializer(allowed_msgpack_modules=[Element])
)


def build_graph() -> CompiledStateGraph[State, BrowserContext, State, State]:
    """Assemble and compile the browser-agent StateGraph."""
    graph = StateGraph(State, context_schema=BrowserContext)

    graph.add_node("navigate", navigate)  # pyright: ignore[reportArgumentType]
    graph.add_node("snapshot", snapshot)  # pyright: ignore[reportArgumentType]
    graph.add_node("plan", plan)  # pyright: ignore[reportArgumentType]
    graph.add_node("act", act)  # pyright: ignore[reportArgumentType]
    graph.add_node("answer", answer)  # pyright: ignore[reportArgumentType]

    graph.add_edge(START, "navigate")
    graph.add_edge("navigate", "snapshot")
    graph.add_edge("snapshot", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("answer", END)

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