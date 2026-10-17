import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from agent.config import AgentConfig
from agent.context import STRATEGIES, Context, RepoView, UnknownStrategyError
from agent.harness.dataset import AgentTask, Task
from agent.harness.evaluate import Evaluation
from agent.harness.patch import ApplyResult
from agent.harness.sandbox import SandboxError
from agent.runs import load_run
from agent.solve import capture_diff, solve

from .fakes import ScriptedModel, ai, call
from .shell import LocalShell

TASK = Task(
    instance_id="psf__requests-0000",
    repo="psf/requests",
    base_commit="a" * 40,
    problem_statement="Something is broken.",
    FAIL_TO_PASS=("test_x.py::test_new",),
    PASS_TO_PASS=(),
    gold_patch="GOLD",
    test_patch="TEST_PATCH",
    version="2.0",
    image="img",
    eval_type="pass_and_fail",
    log_parser="parse_log_pytest",
    eval_script="EVAL",
)
CONFIG = AgentConfig(model="anthropic:claude-sonnet-5-5", max_turns=5)


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / "mod.py").write_text("x = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    return repo


@pytest.fixture
def evaluated(monkeypatch: pytest.MonkeyPatch, repo: Path) -> list[str]:
    """Run the 'sandbox' on the host repo and record the diffs sent to evaluate()."""
    diffs: list[str] = []

    class FakeSandbox:
        @classmethod
        @contextmanager
        def start(cls, task: Task) -> Iterator[LocalShell]:
            yield LocalShell(repo)

    def fake_evaluate(task: Task, diff: str) -> Evaluation:
        diffs.append(diff)
        return Evaluation(
            instance_id=task.instance_id,
            resolved=bool(diff),
            apply=ApplyResult(applied=True),
            tests=None,
            score=None,
            runtime_s=1.0,
        )

    monkeypatch.setattr("agent.solve.Sandbox", FakeSandbox)
    monkeypatch.setattr("agent.solve.evaluate", fake_evaluate)
    return diffs


def solve_with(replies: list[Any], tmp_path: Path, **config: Any) -> Any:
    # The agent only uses bash here: read/write/edit resolve paths against /testbed.
    return solve(
        TASK,
        CONFIG.model_copy(update=config),
        run_id="test-run",
        model=ScriptedModel(replies=replies, seen=[]),
        runs_dir=tmp_path / "runs",
    )


def test_solve_scores_the_agents_diff(evaluated: list[str], tmp_path: Path) -> None:
    run = solve_with(
        [
            ai(
                call("bash", "c1", command="echo 'x = 2' > mod.py && echo 'y' > new.py")
            ),
            ai(call("submit", "c2")),
        ],
        tmp_path,
    )
    assert run.stop_reason == "submitted" and run.resolved
    assert evaluated == [run.diff]
    assert "-x = 1\n+x = 2" in run.diff
    assert "new file mode" in run.diff and "+y" in run.diff  # new files are included
    assert run.config.model == "anthropic:claude-sonnet-5-5"
    assert run.turns == 2 and run.cost_usd > 0
    assert run.runtime_s >= run.agent_runtime_s


def test_a_limit_still_ends_in_a_scored_diff(
    evaluated: list[str], tmp_path: Path
) -> None:
    replies = [
        ai(call("bash", f"c{i}", command=f"echo {i} >> mod.py")) for i in range(5)
    ]
    run = solve_with(replies, tmp_path, max_turns=2)
    assert run.stop_reason == "max_turns"
    assert "+0\n+1" in run.diff and evaluated == [run.diff]


def test_no_changes_scores_an_empty_diff(evaluated: list[str], tmp_path: Path) -> None:
    run = solve_with([ai(call("submit", "c1"))], tmp_path)
    assert run.diff == "" and evaluated == [""]
    assert not run.resolved


def test_run_is_saved(evaluated: list[str], tmp_path: Path) -> None:
    run = solve_with(
        [
            ai(call("bash", "c1", command="echo 'x = 2' > mod.py")),
            ai(call("submit", "c2")),
        ],
        tmp_path,
    )
    directory = tmp_path / "runs" / "test-run" / TASK.instance_id
    assert {p.name for p in directory.iterdir()} == {
        "run.json",
        "patch.diff",
        "trajectory.json",
    }
    assert (directory / "patch.diff").read_text() == run.diff
    assert load_run("test-run", TASK.instance_id, tmp_path / "runs") == run


def test_nothing_hidden_reaches_the_model(evaluated: list[str], tmp_path: Path) -> None:
    model = ScriptedModel(
        replies=[
            ai(call("bash", "c1", command="cat mod.py && git log -p")),
            ai(call("bash", "c2", command="echo 'x = 2' > mod.py && git diff")),
            ai(call("submit", "c3")),
        ],
        seen=[],
    )
    run = solve(TASK, CONFIG, model=model, runs_dir=None)
    # Every request, not just the first: tool results come from the sandbox.
    for request in model.seen:
        sent = "\n".join(str(m.content) for m in request)
        for hidden in (TASK.gold_patch, TASK.test_patch, TASK.eval_script, "test_new"):
            assert hidden not in sent
    assert not run.flagged


def test_a_leak_through_the_sandbox_is_flagged(
    evaluated: list[str], tmp_path: Path, repo: Path
) -> None:
    fix = "return decode_with_fallback(chunk, encoding, errors='replace')"
    task = TASK.model_copy(
        update={"gold_patch": f"diff --git a/mod.py b/mod.py\n+    {fix}\n"}
    )
    leak = repo / "leak.txt"  # not committed: stands in for a newer release on disk
    leak.write_text(f"    {fix}\n")
    model = ScriptedModel(
        replies=[
            ai(call("bash", "c1", command="cat leak.txt")),
            ai(call("submit", "c2")),
        ],
        seen=[],
    )
    run = solve(task, CONFIG, model=model, runs_dir=None)
    assert run.hidden_text_seen == [fix] and run.flagged


def test_capture_diff_keeps_git_warnings_out(repo: Path) -> None:
    git(
        repo, "config", "core.autocrlf", "true"
    )  # makes `git add` warn about line endings
    (repo / "mod.py").write_text("x = 2\n")
    diff = capture_diff(LocalShell(repo))
    assert diff.startswith("diff --git a/mod.py b/mod.py")
    assert "warning" not in diff


def test_capture_diff_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="failed to capture the diff"):
        capture_diff(LocalShell(tmp_path))  # not a git repo


def test_an_unknown_context_fails_before_the_sandbox_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoSandbox:
        @classmethod
        def start(cls, task: Task) -> None:
            raise AssertionError("the sandbox should not start")

    monkeypatch.setattr("agent.solve.Sandbox", NoSandbox)
    with pytest.raises(UnknownStrategyError):
        solve(TASK, CONFIG.model_copy(update={"context": "nope"}), runs_dir=None)


def test_the_strategys_context_is_in_the_first_message(
    evaluated: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[tuple[object, object, int]] = []

    class Fixed:
        name = "fixed"

        def build(self, task: AgentTask, repo: RepoView, budget_chars: int) -> Context:
            received.append((task, repo, budget_chars))
            return Context(strategy=self.name, text="CHOSEN_CODE")

    monkeypatch.setitem(STRATEGIES, "fixed", Fixed())
    model = ScriptedModel(replies=[ai(call("submit", "c1"))], seen=[])
    config = CONFIG.model_copy(update={"context": "fixed", "context_budget_chars": 99})
    run = solve(TASK, config, model=model, runs_dir=None)
    ((task, repo, budget),) = received
    # The strategy sees the agent's view of the task only, never the Task.
    assert type(task) is AgentTask and isinstance(repo, RepoView) and budget == 99
    assert "CHOSEN_CODE" in str(model.seen[0][1].content)
    assert run.config.context == "fixed"
