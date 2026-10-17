# coding-agent

An autonomous bug-fixing agent for Python repos, built with LangGraph and evaluated on [SWE-bench Lite](https://www.swebench.com/). It takes an issue, works in a sandbox, and iterates against real test output. The context strategy is a swappable plugin, and the end result is a benchmark comparing strategies. See [docs/PLAN.md](docs/PLAN.md) for the full plan.

## Status

| Phase | Name | Status |
| --- | --- | --- |
| 1 | Sandbox + test runner (no AI) | Done |
| 2 | Bare agent loop | Done |
| 3 | Context strategy plugins | In progress |
| 4 | Graph traversal strategy | Not started |
| 5 | Eval harness + experiments | Not started |
| 6 | Polish | Not started |

Phase 1 built the evaluation harness: given a task and a candidate patch, it starts a fresh container, applies the patch, runs the hidden tests and reports resolved or not. Spec in [docs/PHASE1.md](docs/PHASE1.md), dev tasks and measured runtimes in [docs/DEV_TASKS.md](docs/DEV_TASKS.md).

Phase 2 builds the agent: an LLM loop that reads the issue, explores and edits the repo in an offline sandbox with four tools (`read`, `write`, `edit`, `bash`), and submits a diff that the harness scores. Spec in [docs/PHASE2.md](docs/PHASE2.md).

## Setup

Requirements:

- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- Docker. SWE-bench publishes x86_64 images only; on Apple Silicon they run emulated, so install Rosetta 2 (`softwareupdate --install-rosetta`) and give Docker at least 8 GB of memory. Each task image is about 3 GB.

```bash
uv sync
uv run pre-commit install   # run `make lint` before every commit
cp .env.example .env        # then add GOOGLE_API_KEY or ANTHROPIC_API_KEY
```

Only the key for the provider you run is needed.

## Run the agent

```bash
uv run harness solve psf__requests-3362                    # one task (or several ids)
uv run harness solve --dev                                 # every dev task
uv run harness solve --dev --model anthropic:claude-sonnet-5-5
uv run harness solve --dev --max-turns 20 --run-id my-batch
uv run harness solve --dev --context none,full_file,repo_map --repeat 3 --run-id cmp
```

`solve` runs the agent on each task in a fresh offline sandbox, captures its diff, scores it with the harness and prints a line per task, then a total. It exits 0 if every task is resolved and 1 if not. These calls spend API credits.

`--context` picks the context strategy, the code put in the first message before the agent starts (default `none`, the Phase 2 baseline). With several strategies or `--repeat N`, `solve` runs N rounds; in each, every task runs under every strategy in turn, and the strategy that goes first rotates between rounds. Each round and strategy is saved as its own batch, `<run_id>-<round>-<strategy>`, and the summary has a line per strategy.

```text
pytest-dev__pytest-5227      RESOLVED      submitted       26 turns    362k tok  $0.1617   0 tool errors   136s
```

The model is a `provider:model` string; the default is `google_genai:gemini-2.5-flash`. A model needs an entry in the price table in `src/agent/model.py`, so every run records its cost. Each run is saved under `runs/<run_id>/<instance_id>/` (git-ignored): `run.json` with the config, stop reason, tokens, cost and evaluation, `patch.diff`, and `trajectory.json` with every message and tool call. A run stops when the agent calls `submit` or hits a limit (30 turns, a token budget, a wall-clock budget); either way its diff is scored.

## Evaluate a patch

```bash
uv run harness run pallets__flask-4992 --gold             # the known-good fix
uv run harness run pallets__flask-4992 --empty            # no change
uv run harness run pallets__flask-4992 --patch fix.diff   # any unified diff ('-' reads stdin)
uv run harness validate                                   # gold, empty and broken patches on every dev task
```

`run` prints the verdict, the FAIL_TO_PASS / PASS_TO_PASS counts and the runtime, and explains a failure (git's error, the failing tests, or the end of the log). It exits 0 if the task is resolved and 1 if not. The first run of a task pulls its image; after that an evaluation takes about 4-6 s.

```text
pallets__flask-4992          RESOLVED      F2P 1/1  P2P 18/18  (tests 3.4s, total 4.4s)
```

From Python:

```python
from agent.harness.dataset import load_task
from agent.harness.evaluate import evaluate

task = load_task("pallets__flask-4992")
result = evaluate(task, diff)   # diff: a unified diff as a string
result.resolved, result.score, result.runtime_s
```

## How evaluation works

1. Start a container from the task's published SWE-bench image, with memory, CPU and process limits, and check that the repo is at the task's `base_commit`.
2. Apply the candidate patch with `git apply`. If it does not apply, stop: not resolved, with git's error.
3. Run the task's official SWE-bench eval script: reset the test files, apply `test_patch` (the hidden tests), reinstall the package if needed, and run the repo's test command, under a timeout.
4. Parse the log and grade it with SWE-bench's own parser and grading: resolved means every FAIL_TO_PASS test passes and every PASS_TO_PASS test still passes.
5. Remove the container.

Tasks come from [`SWE-bench/SWE-bench_Lite`](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite), which ships each task's image, eval script and log parser. Reusing them keeps scores comparable with the official leaderboard. `gold_patch`, `test_patch` and `eval_script` are evaluation-only and never reach the agent.

## Layout

```text
src/agent/
  config.py             AgentConfig: the model string and the run limits
  model.py              make_model (one place for provider differences) and cost_usd
  tools.py              read, write, edit, bash, submit, run inside the sandbox
  prompts.py            the system prompt and the first message, built from the issue only
  graph.py              the LangGraph loop: model and tools until submit or a limit
  solve.py              solve(task, config) -> Run: run, capture the diff, score, save
  runs.py               save and load runs under runs/
  isolation.py          flag runs where the agent may have seen the answer
  cli.py                the `harness` command (solve, plus run and validate)
  harness/
    dataset.py          load SWE-bench Lite into Task models; the dev task list
    sandbox.py          one Docker container per attempt: exec, write_file, limits, cleanup
    patch.py            apply_patch -> ApplyResult
    runner.py           run_tests -> TestResult (official eval script + log parser)
    scoring.py          score / is_resolved (official SWE-bench grading)
    evaluate.py         evaluate(task, diff) -> Evaluation, the whole pipeline
    cli.py              the run and validate subcommands
tests/
  unit_tests/           fast, no Docker
  integration_tests/    need Docker and the dev task images
docs/                   plan, Phase 1 and 2 specs, dev tasks and runtimes
```

## Development

```bash
make test                # unit tests
make integration_tests   # Docker tests; pulls about 14 GB of images on first run
make lint                # ruff + mypy --strict
make format              # ruff format and import sorting
```

Unit tests need no Docker or API keys: the loop is tested with a scripted fake model. Integration tests need Docker; `tests/integration_tests/test_solve.py` runs `solve()` end to end with a scripted agent, so it needs no API key either.

The five dev tasks in `DEV_TASK_IDS` are used to build and tune the harness and agent, so they are excluded from benchmark numbers.
