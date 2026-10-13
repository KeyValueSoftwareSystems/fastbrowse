# FastBrowse

An LLM-driven web-browsing agent built on [LangGraph](https://langchain-ai.github.io/langgraph/)
and [Playwright](https://playwright.dev/python/).

FastBrowse drives a real browser toward a natural-language goal. Planning and form
text generation use an OpenAI chat model, while routing and element selection use
[`langchain-typesafe`](https://pypi.org/project/langchain-typesafe/) categorical
decisions. The agent loops — snapshot the page, plan, route, pick an element, act —
until it decides the goal is complete.

## How it works

```mermaid
flowchart TD
    START([START]) --> navigate
    navigate --> snapshot
    snapshot --> plan
    plan --> act
    act -->|"click / fill / scroll"| snapshot
    act -->|navigate| navigate
    act -->|answer| answer
    answer --> END([END])
```

The planning node asks the chat model for a structured `PlanUpdate` (an ordered
plan plus the single goal to pursue next). The `act` node sends a **single**
TypeSafe request per cycle carrying the operation (`click` / `fill` / `scroll` /
`navigate` / `answer`) plus one speculative target head per offered operation
(`click_target`, `fill_target`); only the head matching the chosen operation is
executed. When the task is complete, `act` routes to the `answer` node, which asks
the chat model for the final answer.

### Nodes

| Node | Purpose |
| --- | --- |
| `navigate` | Choose a URL (via the decision model when several are queued) and navigate the page. |
| `snapshot` | Capture an aria snapshot and the interactive elements across every frame into state. |
| `plan` | Ask the chat model for an updated plan and current goal. |
| `act` | One decision request: pick an operation and its target, run it, then loop or route to `navigate`/`answer`. |
| `answer` | Ask the chat model for the final answer from the goal, page text, and observations, then end. |

## Requirements

- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Playwright's Chromium browser

## Install

```bash
uv sync
uv run playwright install chromium
```

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | API key for the chat model. |
| `OPENAI_BASE_URL` | Optional base URL for the chat model endpoint. |
| `OPENAI_MODEL` | Model used for planning and generating form text. |
| `TYPESAFE_API_KEY` | API key for the decision model. |
| `TYPESAFE_BASE_URL` | Base URL for the decision model endpoint. |
| `TYPESAFE_MODEL` | Model used for routing and element selection. |

Settings are loaded with `pydantic-settings` from the environment and `.env`.

## Usage

```bash
uv run fastbrowse \
  --prompt "Find the price of the Pro plan" \
  --url "https://example.com"
```

Show the browser window instead of running headless:

```bash
uv run fastbrowse --prompt "..." --url "https://example.com" --headed
```

When the run finishes it prints the current goal, the plan, and the accumulated
observations.

| Flag | Required | Description |
| --- | --- | --- |
| `--prompt` | yes | Task/goal for the agent. |
| `--url` | yes | Starting URL. |
| `--headed` | no | Show the browser window (default is headless). |

## Architecture

```
src/fastbrowse/
├── cli.py              # argparse entry point (the `fastbrowse` script)
├── config.py           # pydantic-settings, reads .env
├── logging.py          # logger setup + decision logging
├── adapter/
│   ├── llm.py          # ChatOpenAI factory
│   └── decision.py     # TypeSafeClassifier factory
└── agent/
    ├── agent.py        # graph nodes, graph assembly, run()
    ├── state.py        # State TypedDict
    ├── context.py      # Playwright browser/page runtime context
    ├── snapshot.py     # Element model + page snapshot capture
    ├── tools.py        # browser actions (page-based helpers)
    ├── prompt.py       # prompt text
    └── utils.py        # shared helpers (decision classify, etc.)
```

### State

The graph state (`agent/state.py`) is a `TypedDict`:

| Field | Description |
| --- | --- |
| `prompt` | The user's task. |
| `urls` | Candidate URLs to navigate to. |
| `snapshot` | Latest aria snapshot text. |
| `elements` | Interactive `Element`s captured from the page. |
| `plan` | Current ordered plan. |
| `current_goal` | The single goal being pursued. |
| `observations` | Accumulated action results. |
| `messages` | LangChain messages (reducer-accumulated). |
| `answer` | Final answer produced by the `answer` node. |

## Development

Type checking uses `basedpyright` (configured under `[tool.basedpyright]` in
`pyproject.toml`):

```bash
uv run basedpyright
```

There is no test suite yet.

## Notes

- Both the OpenAI and TypeSafe endpoints require valid credentials.
- The checkpointer is in-memory, so runs are not persisted between processes.
- The agent operates on a single page but snapshots and acts across all frames on
  that page (including iframes).
