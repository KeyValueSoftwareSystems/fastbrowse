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

from fastbrowse.adapter.llm import llm
from fastbrowse.agent.context import BrowserContext, build_context, destroy_context
from fastbrowse.agent.helpers import (
    COMPLETE,
    FAILURE_LIMIT,
    MAX_FAILURES,
    MAX_STALL,
    MAX_STEPS,
    STALL_LIMIT,
    UPDATE_T,
    element_by_index,
    fill_text,
    generate,
    generate_plan,
    goal_criteria,
    navigate_to,
    run_cookie,
    run_request,
    run_storage,
    select_goal,
    snapshot_change,
    targets,
)
from fastbrowse.agent.prompt import (
    ACT_INSTRUCTION,
    ANSWER_SYSTEM,
    CLICK_TARGET_INSTRUCTION,
    COOKIE_ACTION_INSTRUCTION,
    FILL_TARGET_INSTRUCTION,
    KEY_NAME_INSTRUCTION,
    NAVIGATE_INSTRUCTION,
    PLAN_GOAL_INSTRUCTION,
    PLAN_UPDATE_INSTRUCTION,
    REQUEST_METHOD_INSTRUCTION,
    STORAGE_ACTION_INSTRUCTION,
    TYPE_INSTRUCTION,
)
from fastbrowse.agent.snapshot import Element, take_snapshot
from fastbrowse.agent.state import State
from fastbrowse.agent.tools import (
    click,
    failure,
    fill,
    is_failure,
    press_key,
    scroll,
    take_screenshot,
    type_text,
)
from fastbrowse.agent.utils import append, ask, classify
from fastbrowse.logging import get_logger, setup_logging

logger = get_logger("agent")


@workflow
async def navigate(state: State, runtime: Runtime[BrowserContext]) -> dict[str, Any]:
    """Navigate to a URL chosen from state, or stay on the current page."""
    page = runtime.context["page"]
    urls = sorted(state.get("urls", set()))
    Netra.set_input({"urls": urls, "current_url": page.url})

    if not urls:
        observation = f"navigate -> staying on {page.url} (no urls)"
        logger.info("navigate: no urls in state, staying")
    elif len(urls) == 1:
        target = urls[0]
        if page.url != target:
            logger.info("navigate: single url -> %s", target)
            observation = await navigate_to(page, target)
        else:
            logger.info("navigate: already at %s", target)
            observation = f"navigate -> already at {page.url}"
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
            observation = await navigate_to(page, chosen)
        else:
            logger.info("navigate: staying on %s", page.url)
            observation = f"navigate -> staying on {page.url}"

    Netra.set_output({"url": page.url, "observation": observation})
    return {
        "observations": append(state.get("observations", ""), observation),
        "messages": [AIMessage(content=observation)],
        "current_url": page.url,
    }


@workflow
async def snapshot(state: State, runtime: Runtime[BrowserContext]) -> dict[str, Any]:
    """Capture an aria snapshot and the interactive elements into state."""
    page = runtime.context["page"]
    Netra.set_input({"url": page.url})
    aria = await page.aria_snapshot()
    elements = await take_snapshot(page)
    logger.info("snapshot: %d aria chars, %d elements", len(aria), len(elements))
    observation = snapshot_change(state, page.url, elements)
    stall = 0 if observation else state.get("stall", 0) + 1

    update: dict[str, Any] = {
        "snapshot": aria,
        "elements": elements,
        "current_url": page.url,
        "stall": stall,
    }
    if observation:
        update["observations"] = append(state.get("observations", ""), observation)
        update["messages"] = [AIMessage(content=observation)]

    Netra.set_output(
        {
            "snapshot_chars": len(aria),
            "elements": len(elements),
            "observation": observation,
            "stall": stall,
        }
    )
    return update


@workflow
async def plan(state: State, runtime: Runtime[BrowserContext]) -> dict[str, Any]:
    """Advance the plan: rewrite it when needed, otherwise select the current goal."""
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
        new_plan = await generate_plan(state)
        result: dict[str, Any] = {
            "plan": new_plan,
            "current_goal": await select_goal(state, new_plan),
        }
    elif state.get("stall", 0) >= STALL_LIMIT or state.get("failures", 0) >= FAILURE_LIMIT:
        reason = "stalled" if state.get("stall", 0) >= STALL_LIMIT else "actions failing"
        logger.info("plan: %s, rewriting", reason)
        new_plan = await generate_plan(state)
        result = {
            "plan": new_plan,
            "current_goal": await select_goal(state, new_plan),
        }
    else:
        questions: dict[str, Question] = {
            "needs_update": Noul(
                instructions=PLAN_UPDATE_INSTRUCTION,
                criteria=NoulCriteria(
                    true="The plan must be rewritten.",
                    false="The plan still holds.",
                ),
            ),
            "goal": Choice(
                instructions=PLAN_GOAL_INSTRUCTION,
                criteria=goal_criteria(plan_items),
            ),
        }
        response = await ask("plan", state, questions)

        if response.nouls["needs_update"].noul >= UPDATE_T:
            logger.info("plan: needs update, rewriting")
            new_plan = await generate_plan(state)
            result = {
                "plan": new_plan,
                "current_goal": await select_goal(state, new_plan),
            }
        else:
            chosen = str(response.choices["goal"].choice)
            logger.info("plan: goal=%r", chosen)
            if chosen == COMPLETE:
                result = {"plan": [], "current_goal": ""}
            else:
                result = {"current_goal": chosen}

    Netra.set_output(result)
    return result


@workflow
async def act(state: State, runtime: Runtime[BrowserContext]) -> Command[str]:
    """Choose one operation and its target in a single request, then execute it."""
    steps = state.get("steps", 0) + 1
    stall = state.get("stall", 0)
    failures = state.get("failures", 0)

    # Loop guards: never spin forever when the task cannot be completed.
    if steps > MAX_STEPS or stall >= MAX_STALL or failures >= MAX_FAILURES:
        if steps > MAX_STEPS:
            reason = "step limit"
        elif stall >= MAX_STALL:
            reason = "page stopped changing"
        else:
            reason = "too many failed actions"
        observation = f"act -> stopping ({reason}); producing final answer"
        logger.info("act: %s", observation)
        Netra.set_output({"route": "answer", "observation": observation, "steps": steps})
        return Command(
            goto="answer",
            update={
                "steps": steps,
                "observations": append(state.get("observations", ""), observation),
                "messages": [AIMessage(content=observation)],
            },
        )

    elements = state.get("elements", [])
    click_targets = targets(elements, "click")
    fill_targets = targets(elements, "fill")

    criteria: dict[str, Any] = {
        "navigate": "Go back and navigate to a different URL.",
        "cookie": "Read, set, delete, or clear browser cookies.",
        "storage": "Read, write, remove, or clear localStorage / sessionStorage.",
        "request": "Send an HTTP request using the page's session.",
        "screenshot": "Take a full-page screenshot of the current page.",
        "key": "Press a single key such as Enter, Backspace, or Escape on the page keyboard.",
        "type": "Type text (e.g. a word or letters) on the page keyboard.",
        "answer": "The task is complete; produce the final answer.",
    }
    # Only offer scroll while the page is still responding to it.
    if stall < STALL_LIMIT:
        criteria["scroll"] = "Scroll the page to reveal more content."
    if click_targets:
        criteria["click"] = "Click one of the clickable elements."
    if fill_targets:
        criteria["fill"] = "Type text into one of the fields."

    questions: dict[str, Question] = {
        "route": Choice(instructions=ACT_INSTRUCTION, criteria=criteria),
        "cookie_action": Choice(
            instructions=COOKIE_ACTION_INSTRUCTION,
            criteria={
                "read": "Read the cookies currently set in the session.",
                "set": "Set a cookie with a generated name and value.",
                "delete": "Delete cookies with a generated name.",
                "clear": "Clear every cookie in the session.",
            },
        ),
        "storage_action": Choice(
            instructions=STORAGE_ACTION_INSTRUCTION,
            criteria={
                "read_local": "Read all localStorage entries.",
                "set_local": "Set a localStorage entry.",
                "remove_local": "Remove a localStorage entry.",
                "clear_local": "Clear all localStorage entries.",
                "read_session": "Read all sessionStorage entries.",
                "set_session": "Set a sessionStorage entry.",
                "remove_session": "Remove a sessionStorage entry.",
                "clear_session": "Clear all sessionStorage entries.",
            },
        ),
        "request_method": Choice(
            instructions=REQUEST_METHOD_INSTRUCTION,
            criteria={
                "GET": "Issue a GET request.",
                "POST": "Issue a POST request with a body.",
                "PUT": "Issue a PUT request with a body.",
                "PATCH": "Issue a PATCH request with a body.",
                "DELETE": "Issue a DELETE request.",
            },
        ),
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
    try:
        if route == "click" and click_targets:
            element = element_by_index(
                elements, str(response.choices["click_target"].choice)
            )
            clicked = await click(page, element.selector, frame_index=element.frame_index)
            observation = f"click -> {clicked}"
        elif route == "fill" and fill_targets:
            element = element_by_index(
                elements, str(response.choices["fill_target"].choice)
            )
            value = await fill_text(state, element)
            filled = await fill(
                page, element.selector, value, frame_index=element.frame_index
            )
            observation = f"fill -> {filled}"
        elif route == "key":
            value = await generate(state, KEY_NAME_INSTRUCTION)
            pressed = await press_key(page, value)
            observation = f"key -> {pressed}"
        elif route == "type":
            value = await generate(state, TYPE_INSTRUCTION)
            typed = await type_text(page, value)
            observation = f"type -> {typed}"
        elif route == "cookie":
            observation = await run_cookie(state, page, response)
        elif route == "storage":
            observation = await run_storage(state, page, response)
        elif route == "request":
            observation = await run_request(state, page, response)
        elif route == "screenshot":
            observation = f"screenshot -> {await take_screenshot(page)}"
        else:
            scrolled = await scroll(page, 600)
            observation = f"scroll -> {scrolled}"
    except Exception as exc:  # noqa: BLE001 - a failed action must not end the run
        observation = failure(route, "action", exc)

    # Failures are recorded as observations so the planner can react to them.
    failed = is_failure(observation)
    failures = 0 if not failed else failures + 1
    logger.info("act: %s", observation)
    Netra.set_output({"route": route, "observation": observation, "steps": steps})
    return Command(
        goto="snapshot",
        update={
            "steps": steps,
            "failures": failures,
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
            config={
                "configurable": {"thread_id": uuid.uuid4().hex},
                # Backstop in case a node keeps routing to itself: each cycle is
                # ~4 supersteps, so this allows the step cap plus headroom.
                "recursion_limit": MAX_STEPS * 5 + 25,
            },
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