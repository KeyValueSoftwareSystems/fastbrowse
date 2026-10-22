from __future__ import annotations

import argparse
import asyncio
import sys

from fastbrowse.agent.agent import run
from fastbrowse.logging import setup_logging
from fastbrowse.mcp import run_server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastbrowse",
        description=(
            "Run the FastBrowse web agent, or launch the MCP server when no "
            "arguments are given."
        ),
    )
    parser.add_argument("--prompt", help="Task/goal for the agent.")
    parser.add_argument("--url", help="Starting URL.")
    parser.add_argument("--headed", action="store_true", help="Show the browser window.")
    return parser


async def _run(prompt: str, url: str, headed: bool) -> None:
    result = await run(prompt, url, headless=not headed)
    print("CURRENT GOAL:", result.get("current_goal"))
    print("PLAN:", result.get("plan"))
    print("OBSERVATIONS:")
    print(result.get("observations", ""))
    print("ANSWER:")
    print(result.get("answer", ""))


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv

    # No arguments means "act as an MCP server"; any arguments run the CLI agent.
    if not args:
        run_server()
        return

    setup_logging()
    parser = _parser()
    parsed = parser.parse_args(args)
    if not parsed.prompt or not parsed.url:
        parser.error("--prompt and --url are required when running the agent")
    asyncio.run(_run(parsed.prompt, parsed.url, parsed.headed))


if __name__ == "__main__":
    main()
