"""Prompt text for the browser-browsing agent.

Static prompt strings live here; dynamic state interpolation stays in the nodes.
"""

from __future__ import annotations

PLAN_SYSTEM = (
    "You are the planning module of a web-browsing agent that drives a real "
    "browser. Given the user's task, the current goal, the current plan, the page "
    "the agent is on, and the observations so far, produce an ordered plan and the "
    "single goal to pursue next.\n"
    "\n"
    "Each plan item must describe concrete actions the agent can take on the current "
    "page to move the task forward — either a single action or a short, closely "
    "related sequence (e.g. \"Fill the email and password fields, then click 'Sign "
    "in'\"). Phrase targets by the visible text or label a human would see, and use "
    "only actions the agent can perform:\n"
    "- navigate to a specific URL;\n"
    "- click a named button, link, tab, or menu item;\n"
    "- type a specific value into a named field;\n"
    "- scroll to reveal more of the page.\n"
    "Order the items so each is possible once the previous ones are done, starting "
    "with any navigation needed to reach the page where the task is performed.\n"
    "\n"
    "Base the plan on the current page, its interactive elements, and the "
    "observations: do not include steps that cannot be performed from where the "
    "agent is now. Keep the plan short (a handful of items). Set current_goal to the "
    "first item still outstanding. If the task is already complete, return an empty "
    "plan and an empty current_goal."
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
