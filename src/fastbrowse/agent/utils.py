"""Shared helpers for the browser agent."""

from __future__ import annotations

from typing import Any

from langchain_typesafe import Choice, ClassifierRequest, ClassifierResponse, Question

from fastbrowse.adapter.decision import decision
from fastbrowse.agent.snapshot import Element
from fastbrowse.agent.state import State
from fastbrowse.logging import get_logger, log_decision

logger = get_logger("utils")


def append(existing: str, new: str) -> str:
    """Join `new` onto `existing` observations with a newline."""
    return f"{existing}\n{new}".strip()


def decision_state(state: State) -> dict[str, Any]:
    """Build the decision-model view of the graph state."""
    return {
        "goal": state.get("current_goal", ""),
        "plan": state.get("plan", []),
        "observations": state.get("observations", ""),
        "snapshot": state.get("snapshot", "")[:4000],
    }


def summarize(element: Element) -> dict[str, Any]:
    """Reduce an `Element` to the fields shown to the decision model."""
    return {
        "index": element.index,
        "tag": element.tag,
        "selector": element.selector,
        "text": element.text,
        "frame_index": element.frame_index,
    }


async def ask(
    tag: str,
    state: State,
    questions: dict[str, Question],
    *,
    extra: dict[str, Any] | None = None,
) -> ClassifierResponse:
    """Send one decision request (one or more questions) and return the response."""
    request: ClassifierRequest = {
        "state": {**decision_state(state), **(extra or {})},
        "questions": questions,
    }
    response = await decision().ainvoke(request)
    log_decision(logger, tag, request, response)
    return response


async def classify(
    tag: str,
    state: State,
    question_id: str,
    choice: Choice,
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    """Ask the decision model a single-choice question and return the chosen key."""
    response = await ask(tag, state, {question_id: choice}, extra=extra)
    return str(response.choices[question_id].choice)
