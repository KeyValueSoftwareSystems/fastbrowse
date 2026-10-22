"""Prompt text for the browser-browsing agent.

Static prompt strings live here; dynamic state interpolation stays in the nodes.
"""

from __future__ import annotations

PLAN_SYSTEM = (
    "You are the planning module of a web-browsing agent that drives a real "
    "browser. Given the user's task, the current goal, the current plan, the page "
    "the agent is on, and the observations so far, produce an ordered plan of the "
    "actions still to perform.\n"
    "\n"
    "Each plan item must describe concrete actions the agent can take on the current "
    "page to move the task forward — either a single action or a short, closely "
    "related sequence (e.g. \"Fill the email and password fields, then click 'Sign "
    "in'\"). Phrase targets by the visible text or label a human would see, and use "
    "only actions the agent can perform:\n"
    "- navigate to a specific URL;\n"
    "- click a named button, link, tab, or menu item;\n"
    "- type a specific value into a named field;\n"
    "- scroll to reveal more of the page;\n"
    "- read, set, or delete a cookie;\n"
    "- read, write, remove, or clear localStorage / sessionStorage;\n"
    "- send an HTTP request using the page's session;\n"
    "- take a screenshot of the page.\n"
    "Order the items so each is possible once the previous ones are done, starting "
    "with any navigation needed to reach the page where the task is performed.\n"
    "\n"
    "Reading the page is not an action: the page text is already available to every "
    "step, so never make \"read the page\" a plan item. Always end the plan with a "
    "final item phrased as \"Answer the question using the information on the page\" "
    "— that is how the agent produces its final answer from the page text and the "
    "observations.\n"
    "\n"
    "Base the plan on the current page, its interactive elements, and the "
    "observations: do not include steps that cannot be performed from where the "
    "agent is now. Keep the plan short (a handful of items). If the task is already "
    "complete, return an empty plan."
)

FILL_SYSTEM = (
    "You generate text to type into a web form field. Reply with only the exact text "
    "to enter, with no quotes, labels, or explanation."
)

NAVIGATE_INSTRUCTION = (
    "Which URL should the agent navigate to next to best serve the "
    "current goal, or choose 'none' to stay on the current page?"
)

ACT_INSTRUCTION = (
    "Given the goal, plan, and observations, what should the agent do next on the "
    "current page: click an element, fill a form field, type text on the page "
    "keyboard, press a single key (e.g. Enter), scroll, navigate to another URL, "
    "read/set/delete a cookie, read/write/remove/clear localStorage or "
    "sessionStorage, send an HTTP request with the page's session, take a "
    "screenshot, or finish and answer?"
)

TYPE_INSTRUCTION = (
    "Reply with the exact text to type on the page keyboard. If the goal is to "
    "enter a guess in a word game, reply with one English word of the required "
    "length that fits the feedback recorded in the observations. Reply with only "
    "the text."
)

KEY_NAME_INSTRUCTION = (
    "Reply with the name of the single key to press to serve the goal, for example "
    "Enter, Backspace, Tab, Escape, or an arrow key. Reply with only the key name."
)

COOKIE_ACTION_INSTRUCTION = (
    "If the next action involves cookies, which cookie operation is most relevant to "
    "the goal: read, set, delete, or clear?"
)

STORAGE_ACTION_INSTRUCTION = (
    "If the next action involves browser storage, which storage operation is most "
    "relevant to the goal: read, set, remove, or clear localStorage or sessionStorage?"
)

REQUEST_METHOD_INSTRUCTION = (
    "If the next action is to send an HTTP request, which HTTP method should be used "
    "to best serve the goal?"
)

ARG_SYSTEM = (
    "You generate a single argument for a browser-automation action. Reason from the "
    "task, goal, current URL, and observations. Reply with only the exact value to use, "
    "with no quotes, labels, or explanation. If no sensible value exists, reply with an "
    "empty string."
)

COOKIE_NAME_INSTRUCTION = "Reply with the cookie name to operate on for the goal."
COOKIE_VALUE_INSTRUCTION = "Reply with the cookie value to set for the goal."
STORAGE_KEY_INSTRUCTION = "Reply with the storage key to operate on for the goal."
STORAGE_VALUE_INSTRUCTION = "Reply with the storage value to set for the goal."
REQUEST_URL_INSTRUCTION = (
    "Reply with the absolute request URL to fetch for the goal (use the current URL "
    "as the base when a relative path is needed)."
)
REQUEST_BODY_INSTRUCTION = (
    "Reply with the raw request body to send for the goal, or an empty string if none."
)

CLICK_TARGET_INSTRUCTION = (
    "If the next action is to click, which clickable element is most relevant to the "
    "goal? Choose one of the offered element indices."
)

FILL_TARGET_INSTRUCTION = (
    "If the next action is to type, which field is most relevant to the goal? Choose "
    "one of the offered field indices."
)

PLAN_UPDATE_INSTRUCTION = (
    "Given the task, the current plan, and the observations so far, does the plan "
    "need to be rewritten (for example because the goal is complete or new "
    "information invalidates the plan)?"
)

PLAN_GOAL_INSTRUCTION = (
    "Which plan item is the current goal to pursue next, given the task, plan, and "
    "observations? Choose one of the offered plan items, or '✓ complete' if every "
    "item is done."
)

ANSWER_SYSTEM = (
    "You are the final-answer module of a web-browsing agent. Given the goal, the "
    "page text, and the observations recorded while acting, write a concise, direct "
    "answer to the goal. Use only information gathered in the observations and page "
    "text; do not invent details. If the task could not be completed, say so and "
    "explain briefly why."
)
