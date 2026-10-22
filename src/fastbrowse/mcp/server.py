from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from fastbrowse.agent.agent import run
from fastbrowse.logging import setup_logging

mcp = FastMCP(
    "FastBrowse",
    instructions=(
        "Browse the web toward a natural-language goal. Call the `browse` tool "
        "with a task prompt and a starting URL; it drives a real browser and "
        "returns the final answer plus the plan and observations from the run."
    ),
)


@mcp.tool
async def browse(prompt: str, url: str, headed: bool = False) -> dict[str, Any]:
    """Drive a browser toward a natural-language goal.

    Args:
        prompt: The task or question to answer.
        url: The URL to start browsing from.
        headed: Show the browser window instead of running headless.

    Returns:
        The final answer, current goal, plan, and accumulated observations.
    """
    result = await run(prompt, url, headless=not headed)
    return {
        "answer": result.get("answer", ""),
        "goal": result.get("current_goal", ""),
        "plan": result.get("plan", []),
        "observations": result.get("observations", ""),
    }


def run_server() -> None:
    """Run the FastBrowse MCP server over stdio."""
    setup_logging()
    mcp.run()
