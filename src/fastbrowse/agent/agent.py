from __future__ import annotations

from langgraph.graph import START, END, StateGraph

from fastbrowse.agent.state import State


# --- placeholder nodes (no implementation) -------------------------------
async def navigate(state: State) -> None:
    """Navigate the browser to a URL (not implemented)."""


async def snapshot(state: State) -> None:
    """Capture the current page state (not implemented)."""


async def plan(state: State) -> None:
    """Build a plan of steps (not implemented)."""


async def router(state: State) -> None:
    """Decide the next step (not implemented)."""


async def find_element(state: State) -> None:
    """Locate an element on the page (not implemented)."""


async def action(state: State) -> None:
    """Perform an action on an element (not implemented)."""


def build_graph():
    """Assemble and compile the StateGraph."""
    graph = StateGraph(State)

    graph.add_node("navigate", navigate)
    graph.add_node("snapshot", snapshot)
    graph.add_node("plan", plan)
    graph.add_node("router", router)
    graph.add_node("find_element", find_element)
    graph.add_node("action", action)

    # Minimal wiring so the graph compiles. Real control-flow edges are TBD.
    graph.add_edge(START, "navigate")
    graph.add_edge("navigate", END)

    return graph.compile()