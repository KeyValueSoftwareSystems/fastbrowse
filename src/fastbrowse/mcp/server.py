from __future__ import annotations

from pydantic import BaseModel, Field

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


class BrowseResult(BaseModel):
    """Outcome of one FastBrowse run."""

    ok: bool = Field(description="False when the agent stopped before finishing.")
    answer: str = Field(description="The agent's final answer.")
    goal: str = Field(description="The goal the agent was pursuing at the end.")
    plan: list[str] = Field(description="The remaining plan at the end of the run.")
    observations: str = Field(description="Accumulated browser action observations.")
    final_url: str = Field(description="The URL of the page when the run ended.")
    steps: int = Field(description="Number of act cycles executed.")
    stop_reason: str | None = Field(
        default=None,
        description="Why the agent was forced to stop, or null if it finished normally.",
    )


@mcp.tool
async def browse(prompt: str, url: str, headed: bool = False) -> BrowseResult:
    """Drive a browser toward a natural-language goal.

    Args:
        prompt: The task or question to answer.
        url: The URL to start browsing from.
        headed: Show the browser window instead of running headless.

    Returns:
        A structured result including whether the run completed, the final
        answer, plan, observations, final URL, and any forced-stop reason.
    """
    result = await run(prompt, url, headless=not headed)
    stop_reason = result.get("stop_reason")
    return BrowseResult(
        ok=not stop_reason,
        answer=result.get("answer", ""),
        goal=result.get("current_goal", ""),
        plan=result.get("plan", []),
        observations=result.get("observations", ""),
        final_url=result.get("current_url", ""),
        steps=result.get("steps", 0),
        stop_reason=stop_reason,
    )


def run_server() -> None:
    """Run the FastBrowse MCP server over stdio."""
    setup_logging()
    mcp.run()
