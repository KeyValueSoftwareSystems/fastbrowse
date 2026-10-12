from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_typesafe import Choice
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

from fastbrowse.adapter.decision import decision
from fastbrowse.adapter.llm import llm
from fastbrowse.agent.context import BrowserContext
from fastbrowse.agent.snapshot import Element, take_snapshot
from fastbrowse.agent.state import State
from fastbrowse.agent.tools import click_element, fill_input, scroll_page

_PLAN_SYSTEM = (
    "You are the planning module of a web-browsing agent. Given the user's task, "
    "the current plan, the current goal, and observations so far, produce an updated "
    "ordered plan and the single goal to pursue next. If the current goal is complete, "
    "advance to the next plan item. If new information invalidates the plan, rewrite it."
)

_FILL_SYSTEM = (
    "You generate text to type into a web form field. Reply with only the exact text "
    "to enter, with no quotes, labels, or explanation."
)


class PlanUpdate(BaseModel):
    plan: list[str]
    current_goal: str


def _append(existing: str, new: str) -> str:
    return f"{existing}\n{new}".strip()


def _decision_state(state: State) -> dict:
    return {
        "goal": state.get("current_goal", ""),
        "plan": state.get("plan", []),
        "observations": state.get("observations", ""),
        "snapshot": state.get("snapshot", "")[:4000],
    }


def _tool_runtime(state: State, runtime: Runtime[BrowserContext]) -> ToolRuntime:
    return ToolRuntime(
        state=state,
        context=runtime.context,
        config=get_config(),
        stream_writer=runtime.stream_writer,
        tool_call_id=None,
        store=runtime.store,
    )


async def navigate(state: State, runtime: Runtime[BrowserContext]) -> dict:
    """Navigate to a URL chosen from state, or stay on the current page."""
    page = runtime.context["page"]
    urls = sorted(state.get("urls", set()))
    if not urls:
        return {}

    if len(urls) == 1:
        target = urls[0]
        if page.url != target:
            await page.goto(target, wait_until="load")
        return {}

    response = await decision().ainvoke(
        {
            "state": {**_decision_state(state), "urls": urls},
            "questions": {
                "url": Choice(
                    instructions=(
                        "Which URL should the agent navigate to next to best serve the "
                        "current goal, or choose 'none' to stay on the current page?"
                    ),
                    criteria={
                        **{url: f"Navigate to {url}" for url in urls},
                        "none": "Stay on the current page.",
                    },
                )
            },
        }
    )
    chosen = response.choices["url"].choice
    if chosen != "none" and page.url != chosen:
        await page.goto(chosen, wait_until="load")
    return {}


async def snapshot(state: State, runtime: Runtime[BrowserContext]) -> dict:
    """Capture an aria snapshot and the interactive elements into state."""
    page = runtime.context["page"]
    return {
        "snapshot": await page.aria_snapshot(),
        "elements": await take_snapshot(page),
    }


async def plan(state: State, runtime: Runtime[BrowserContext]) -> dict:
    """(Re)build the plan and current goal from the task and observations."""
    update = await llm().with_structured_output(PlanUpdate).ainvoke(
        [
            SystemMessage(content=_PLAN_SYSTEM),
            HumanMessage(
                content=(
                    f"Task: {state.get('prompt', '')}\n"
                    f"Current goal: {state.get('current_goal', '')}\n"
                    f"Current plan: {state.get('plan', [])}\n"
                    f"Observations:\n{state.get('observations', '')}"
                )
            ),
        ]
    )
    return {"plan": update.plan, "current_goal": update.current_goal}


async def router(state: State, runtime: Runtime[BrowserContext]) -> Command:
    """Decide the next step and route the graph accordingly."""
    response = await decision().ainvoke(
        {
            "state": _decision_state(state),
            "questions": {
                "route": Choice(
                    instructions=(
                        "Given the goal, plan, and observations, what should the agent "
                        "do next?"
                    ),
                    criteria={
                        "find_element": (
                            "Locate the most relevant element on the current page and "
                            "act on it."
                        ),
                        "navigate": "Go back and navigate to a different URL.",
                        "__end__": "The goal has been achieved; stop.",
                    },
                )
            },
        }
    )
    return Command(goto=response.choices["route"].choice)


async def find_element(state: State, runtime: Runtime[BrowserContext]) -> dict:
    """Choose the snapshot element most relevant to the current state."""
    elements = state.get("elements", [])
    if not elements:
        return {"selected_index": -1}

    criteria: dict = {
        f"el_{element.index}": {
            "tag": element.tag,
            "selector": element.selector,
            "text": element.text,
            "role": element.role,
            "frame_index": element.frame_index,
            "frame_url": element.frame_url,
        }
        for element in elements
    }
    criteria["none"] = "No element is relevant."

    response = await decision().ainvoke(
        {
            "state": _decision_state(state),
            "questions": {
                "element": Choice(
                    instructions=(
                        "Which element is most relevant to act on next? Choose 'none' "
                        "if no element applies."
                    ),
                    criteria=criteria,
                )
            },
        }
    )
    chosen = response.choices["element"].choice
    if chosen == "none":
        return {"selected_index": -1}
    return {"selected_index": int(chosen.removeprefix("el_"))}


async def _fill_text(state: State, element: Element) -> str:
    response = await llm().ainvoke(
        [
            SystemMessage(content=_FILL_SYSTEM),
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


async def action(state: State, runtime: Runtime[BrowserContext]) -> dict:
    """Pick a tool for the selected element, run it, and record an observation."""
    elements = state.get("elements", [])
    index = state.get("selected_index", -1)
    element = next((e for e in elements if e.index == index), None)

    response = await decision().ainvoke(
        {
            "state": {
                **_decision_state(state),
                "element": element.model_dump() if element else None,
            },
            "questions": {
                "tool": Choice(
                    instructions=(
                        "Which action should be performed on the selected element (or "
                        "the page)?"
                    ),
                    criteria={
                        "click": "Click the selected element.",
                        "fill": "Type text into the selected element.",
                        "scroll": "Scroll the page.",
                        "none": "Take no action.",
                    },
                )
            },
        }
    )
    tool_name = response.choices["tool"].choice

    if tool_name in ("click", "fill") and element is None:
        note = f"{tool_name} skipped: no element selected"
        return {"observations": _append(state.get("observations", ""), note)}

    tool_runtime = _tool_runtime(state, runtime)
    if tool_name == "click" and element is not None:
        result = await click_element.coroutine(
            selector=element.selector,
            frame_index=element.frame_index,
            runtime=tool_runtime,
        )
    elif tool_name == "fill" and element is not None:
        value = await _fill_text(state, element)
        result = await fill_input.coroutine(
            selector=element.selector,
            value=value,
            frame_index=element.frame_index,
            runtime=tool_runtime,
        )
    elif tool_name == "scroll":
        result = await scroll_page.coroutine(delta_y=600, runtime=tool_runtime)
    else:
        result = "No action taken."

    observation = f"{tool_name} -> {result}"
    return {
        "observations": _append(state.get("observations", ""), observation),
        "messages": [AIMessage(content=observation)],
    }


def build_graph():
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

    return graph.compile()