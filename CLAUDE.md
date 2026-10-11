# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An autonomous bug-fixing agent for Python repos, built on LangGraph and scored on SWE-bench Lite. It is built in phases (`docs/PLAN.md`): Phase 1, the evaluation harness, is done; Phase 2, the bare agent loop, is in progress, with its spec and seven numbered tasks in `docs/PHASE2.md`. Tasks 1-5 (model client, tools, prompts, loop, `solve` and saved runs) and the `solve` command from task 6 are built; the dev-task review (rest of 6) and isolation and limit checks (7) are not. Read the phase doc before starting a task: it fixes scope, proposed types and what is out of scope.

## Commands

```bash
uv sync                                    # install
uv run make test                           # unit tests: fast, no Docker or API keys
uv run make integration_tests              # needs Docker and the dev task images (~3 GB each)
uv run make lint                           # ruff check, ruff format --diff, mypy --strict on the whole repo
uv run make format                         # apply ruff formatting and import sorting
uv run pytest tests/unit_tests/test_graph.py::test_max_turns   # one test
uv run harness run pallets__flask-4992 --gold                  # evaluate a patch (--empty, --patch FILE|-)
uv run harness validate                                         # gold/empty/broken on every dev task
uv run harness solve psf__requests-3362                         # run the agent on a task (spends API credits)
uv run harness solve --dev [--model anthropic:claude-sonnet-5-5] [--run-id NAME]
```

The pre-commit hook runs `uv run make lint`, so a commit fails on any ruff or mypy error. mypy runs `--strict` over `tests/` too.

SWE-bench images are x86_64 only; on Apple Silicon they run emulated, so container commands are slow. Integration tests and the agent's `bash` should run single test files, not whole suites.

## Architecture

Two layers, with a one-way dependency: the agent (`src/agent/*.py`) imports the harness (`src/agent/harness/`), never the reverse. That is why the `harness` command's entry point is `agent/cli.py`: it adds `solve` and reuses `run`/`validate` through `harness.cli.add_commands` and `dispatch`.

**Harness (Phase 1).** `evaluate(task, diff)` starts a fresh container from the task's published SWE-bench image, applies the diff with `git apply`, runs the official SWE-bench eval script (which resets test files and applies the hidden `test_patch`), and grades the log with SWE-bench's own parser. `Sandbox` is one Docker container per attempt: `exec` and `write_file`, resource limits, removed on exit, repo at `/testbed` (`WORKDIR`).

**Isolation is the core invariant.** `Task` holds evaluation-only fields (`gold_patch`, `test_patch`, `eval_script`, FAIL_TO_PASS tests); `Task.for_agent()` returns an `AgentTask` with only id, repo, base commit and issue text. Anything that reaches the model must be built from `AgentTask`. The agent's sandbox runs with the network off; only `evaluate()` turns it on, in a separate container. The five `DEV_TASK_IDS` are for building and tuning, and are excluded from benchmark numbers.

**Agent (Phase 2).** One run is: `Sandbox.start(task)` → `Toolbox(sandbox, ...)` → `run_agent(task.for_agent(), make_model(config), toolbox, config)` → `AgentResult`. `solve(task, config)` wraps this: it captures the diff in the same offline sandbox (`git add -A && git diff --cached HEAD`, with git's messages kept out of it), scores it with `evaluate()` in a fresh container, and saves a `Run` under `runs/<run_id>/<instance_id>/` (`run.json` summary, `patch.diff`, `trajectory.json`). `runs/` is git-ignored; a run id names one batch.

- `config.py`: `AgentConfig`, the model string (`"provider:model"`, passed to `init_chat_model`) and the limits.
- `model.py`: every provider difference lives here; the rest of the agent sees a LangChain `BaseChatModel`. Claude's prompt caching is set through `model_kwargs`, because a `.bind(cache_control=...)` is dropped by the later `bind_tools()`. `make_model` refuses a model with no entry in `_PRICES`. `cost_usd` prices one turn, since price tiers depend on that turn's prompt size.
- `tools.py`: `Toolbox.run(name, args)` returns a `ToolResult` and never raises; errors come back as `is_error=True` results the model can act on. Only `output` is sent to the model; the other fields are for the trajectory. A non-zero `bash` exit is a normal result (only a timeout is an error). `write`/`edit` copy through a temp file so existing files keep their mode and the diff shows no mode changes.
- `prompts.py`: `SYSTEM_PROMPT` contains nothing task-specific so it caches across turns and tasks. `first_message(task)` is where Phase 3's context strategies will plug in.
- `graph.py`: a model node and a tool node, plus nudge and give-up nodes. Limits (turns, tokens, wall clock) are checked before each model call, so runs stop between turns with a `stop_reason`. A reply without tool calls is nudged once per stall. Every call in a reply gets a `ToolMessage`, including `invalid_tool_calls` (unanswered calls make the Anthropic API reject the next request).

Conventions in the loop that are easy to break:

- **Messages are append-only.** Never edit or remove earlier messages: it invalidates the prompt cache and Claude's thinking blocks. A test checks every request extends the previous one.
- **Node updates are partial.** Nodes return `StateUpdate` (a `total=False` mirror of `State` minus `started`); keep the two in step. List keys have an `operator.add` reducer, so return only new items; counters return new totals.
- **Routers return `Literal` types**, which LangGraph reads to build the conditional edges (there are no destination lists). Return `_END`, not `END`, which is typed as plain `str`.
- **Use `AnyMessage`, not `BaseMessage`**, for anything stored or validated by Pydantic; `BaseMessage` loads back from JSON without tool calls or usage.

## Tests

Unit tests need no Docker or keys. `tests/unit_tests/shell.py` has `LocalShell`, which runs the tools' real shell commands with the host's bash in a temp dir; `tests/unit_tests/fakes.py` has `ScriptedModel`, a fake chat model that replays fixed replies (`ai(call(...))` builds them). `tests/` is a package, so integration tests import these too; `tests/integration_tests/test_solve.py` runs `solve()` end to end with a scripted agent that applies the gold patch, so it needs Docker but no API key. Integration tests use the real dev-task containers, and share one per module where tests clean up after themselves.

## Style

- Docstrings follow ruff's Google convention (`D` rules, imperative first line) in `src/`; `tests/` is exempt. `print` is banned outside the two CLI modules.
- Commit messages: an imperative subject, then a body explaining the change and why; tasks end with a line like `Phase 2, task 4.`
