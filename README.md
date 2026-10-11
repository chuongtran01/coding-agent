# coding-agent

An autonomous bug-fixing agent for Python repos, built with LangGraph and evaluated on [SWE-bench Lite](https://www.swebench.com/). It takes an issue, works in a sandbox, and iterates against real test output. The context strategy is a swappable plugin, and the end result is a benchmark comparing strategies. See [docs/PLAN.md](docs/PLAN.md) for the full plan.

## Status

| Phase | Name | Status |
| --- | --- | --- |
| 1 | Sandbox + test runner (no AI) | Tasks 1-6 done; Task 7 (isolation checks) next |
| 2 | Bare agent loop | Not started |
| 3 | Context strategy plugins | Not started |
| 4 | Graph traversal strategy | Not started |
| 5 | Eval harness + experiments | Not started |
| 6 | Polish | Not started |

Phase 1 built the evaluation harness: given a task and a candidate patch, it starts a fresh container, applies the patch, runs the hidden tests and reports resolved or not. Spec in [docs/PHASE1.md](docs/PHASE1.md), dev tasks and measured runtimes in [docs/DEV_TASKS.md](docs/DEV_TASKS.md).

## Setup

Requirements:

- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- Docker. SWE-bench publishes x86_64 images only; on Apple Silicon they run emulated, so install Rosetta 2 (`softwareupdate --install-rosetta`) and give Docker at least 8 GB of memory. Each task image is about 3 GB.

```bash
uv sync
uv run pre-commit install   # run `make lint` before every commit
```

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
  graph.py              LangGraph scaffold (replaced in Phase 2)
  harness/
    dataset.py          load SWE-bench Lite into Task models; the dev task list
    sandbox.py          one Docker container per attempt: exec, write_file, limits, cleanup
    patch.py            apply_patch -> ApplyResult
    runner.py           run_tests -> TestResult (official eval script + log parser)
    scoring.py          score / is_resolved (official SWE-bench grading)
    evaluate.py         evaluate(task, diff) -> Evaluation, the whole pipeline
    cli.py              the `harness` command
tests/
  unit_tests/           fast, no Docker
  integration_tests/    need Docker and the dev task images
docs/                   plan, Phase 1 spec, dev tasks and runtimes
```

## Development

```bash
make test                # unit tests
make integration_tests   # Docker tests; pulls about 14 GB of images on first run
make lint                # ruff + mypy --strict
make format              # ruff format and import sorting
```

`tests/integration_tests/test_graph.py` exercises the LangGraph scaffold and needs a valid `LANGSMITH_API_KEY` in `.env` (see `.env.example`).

The five dev tasks in `DEV_TASK_IDS` are used to build and tune the harness and agent, so they are excluded from benchmark numbers.
