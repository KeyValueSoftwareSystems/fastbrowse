"""Prompt text for the browser-browsing agent.

Static prompt strings live here; dynamic state interpolation stays in the nodes.
"""

from __future__ import annotations

PLAN_SYSTEM = (
    "You are the planning module of a web-browsing agent. Given the user's task, "
    "the current plan, the current goal, and observations so far, produce an updated "
    "ordered plan and the single goal to pursue next. If the current goal is complete, "
    "advance to the next plan item. If new information invalidates the plan, rewrite it."
)

FILL_SYSTEM = (
    "You generate text to type into a web form field. Reply with only the exact text "
    "to enter, with no quotes, labels, or explanation."
)

NAVIGATE_INSTRUCTION = (
    "Which URL should the agent navigate to next to best serve the "
    "current goal, or choose 'none' to stay on the current page?"
)

ROUTER_INSTRUCTION = (
    "Given the goal, plan, and observations, what should the agent do next?"
)

FIND_ELEMENT_INSTRUCTION = (
    "You are narrowing down which candidate element is most relevant to the goal "
    "and observations. Each option keeps half the list. Pick the half more likely "
    "to contain the relevant element, or 'none' if no element is relevant."
)

ACTION_INSTRUCTION = (
    "Which action should be performed on the selected element (or the page)?"
)

PLAN_UPDATE_INSTRUCTION = (
    "Given the task, the current plan, and the observations so far, does the plan "
    "need to be rewritten (for example because the goal is complete or new "
    "information invalidates the plan)?"
)

PLAN_ITEM_DONE_INSTRUCTION = (
    "Has the following plan item been completed, based on the observations?\n{item}"
)
