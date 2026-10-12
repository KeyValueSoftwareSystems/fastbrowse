from __future__ import annotations

import argparse
import asyncio

from fastbrowse.agent.agent import run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastbrowse", description="Run the FastBrowse web agent."
    )
    parser.add_argument("--prompt", required=True, help="Task/goal for the agent.")
    parser.add_argument("--url", required=True, help="Starting URL.")
    parser.add_argument("--headed", action="store_true", help="Show the browser window.")
    return parser


async def _run(prompt: str, url: str, headed: bool) -> None:
    result = await run(prompt, url, headless=not headed)
    print("CURRENT GOAL:", result.get("current_goal"))
    print("PLAN:", result.get("plan"))
    print("OBSERVATIONS:")
    print(result.get("observations", ""))


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    asyncio.run(_run(args.prompt, args.url, args.headed))


if __name__ == "__main__":
    main()
