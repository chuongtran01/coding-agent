import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.graph import ToolCall
from agent.harness.dataset import Task
from agent.isolation import (
    candidate_lines,
    hidden_lines,
    hidden_text_seen,
    package_names,
    package_paths_outside_testbed,
)

from .shell import LocalShell

FIX = "        return decode_with_fallback(chunk, encoding, errors='replace')"
TEST = "    assert all(isinstance(chunk, str) for chunk in chunks_without_encoding)"
OLD = "    legacy_helper_that_the_base_repo_already_has(argument_one)"
GOLD = f"""diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,4 @@
 def f(chunk, encoding):
-    return chunk
+{FIX}
+{OLD}
+    short = 1
"""
TEST_PATCH = f"""diff --git a/tests/test_mod.py b/tests/test_mod.py
--- a/tests/test_mod.py
+++ b/tests/test_mod.py
@@ -1 +1,2 @@
 import pkg
+{TEST}
"""
ISSUE_LINE = "    print(the_line_quoted_in_the_issue_itself_is_fine)"


def task(**overrides: str) -> Task:
    fields = dict(
        instance_id="psf__requests-0000",
        repo="psf/requests",
        base_commit="a" * 40,
        problem_statement=f"It breaks:\n{ISSUE_LINE}\n",
        FAIL_TO_PASS=(),
        PASS_TO_PASS=(),
        gold_patch=GOLD + f"+{ISSUE_LINE}\n",
        test_patch=TEST_PATCH,
        version="2.0",
        image="img",
        eval_type="pass_and_fail",
        log_parser="parse_log_pytest",
        eval_script="git apply -v - <<'EOF_114329324912'\n...\nEOF_114329324912\n",
    )
    fields.update(overrides)
    return Task(**fields)  # type: ignore[arg-type]


def call(name: str, output: str, **args: str) -> ToolCall:
    return ToolCall(
        turn=1,
        name=name,
        args=args,
        output=output,
        is_error=False,
        truncated=False,
        exit_code=None,
        runtime_s=0.0,
    )


def test_candidates_are_long_new_lines_and_eval_markers() -> None:
    lines = candidate_lines(task())
    assert FIX.strip() in lines and TEST.strip() in lines and OLD.strip() in lines
    assert "EOF_114329324912" in lines
    assert "short = 1" not in lines  # too short to be distinctive
    assert ISSUE_LINE.strip() not in lines  # the agent is meant to see the issue


def test_hidden_lines_drop_what_the_repo_or_its_history_has(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "old.py").write_text(f"{OLD}\n")  # in an old commit only
    git("add", "-A")
    git("commit", "-qm", "old")
    (tmp_path / "old.py").unlink()
    (tmp_path / "mod.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-qm", "base")

    hidden = hidden_lines(task(), LocalShell(tmp_path))
    assert OLD.strip() not in hidden  # reachable through `git log -p`
    assert FIX.strip() in hidden and TEST.strip() in hidden


def test_hidden_text_counts_only_what_the_model_received_first() -> None:
    hidden = [FIX.strip(), TEST.strip()]
    messages = [
        SystemMessage("system"),
        HumanMessage("the issue"),
        AIMessage("", tool_calls=[{"name": "edit", "args": {"new": FIX}, "id": "1"}]),
        ToolMessage(f"Edited.\n{FIX}", tool_call_id="1"),  # its own fix echoed back
        AIMessage("", tool_calls=[{"name": "read", "args": {"path": "x"}, "id": "2"}]),
        ToolMessage(f"leaked:\n{TEST}", tool_call_id="2"),  # received, never written
    ]
    assert hidden_text_seen(hidden, messages) == [TEST.strip()]


def test_package_names() -> None:
    assert package_names(task(repo="pytest-dev/pytest")) == {"pytest", "_pytest"}


@pytest.mark.parametrize(
    ("path", "flagged"),
    [
        ("/opt/miniconda3/lib/python3.11/site-packages/requests/models.py", True),
        (
            "/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/pip/_vendor/"
            "requests/models.py",
            True,
        ),
        (
            "/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/requests/api.py",
            False,
        ),
        ("/testbed/requests/models.py", False),
        ("/opt/miniconda3/lib/python3.11/site-packages/urllib3/util.py", False),
    ],
)
def test_package_paths_outside_testbed(path: str, flagged: bool) -> None:
    found = package_paths_outside_testbed(task(), [call("read", "", path=path)])
    assert found == ([path] if flagged else [])


def test_package_paths_found_in_output_but_not_in_relative_paths_or_urls() -> None:
    leaked = "/opt/miniconda3/lib/python3.11/site-packages/requests/utils.py"
    calls = [
        call("bash", f"{leaked}:12:def f():", command="grep -rn 'def f' / "),
        call(
            "bash",
            "src/requests/models.py https://github.com/psf/requests",
            command="ls",
        ),
    ]
    assert package_paths_outside_testbed(task(), calls) == [leaked]
