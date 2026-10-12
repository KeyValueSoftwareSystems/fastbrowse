from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from fastbrowse.agent.snapshot import Element


class State(TypedDict):
    urls: set[str]
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[str]
    current_goal: str
    observations: str
    prompt: str
    snapshot: str
    elements: list[Element]
    selected_index: int