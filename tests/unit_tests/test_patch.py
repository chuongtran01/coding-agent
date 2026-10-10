from typing import Any, cast

from agent.harness.patch import apply_patch
from agent.harness.sandbox import ExecResult, Sandbox

DIFF = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b"


class FakeSandbox:
    """Records calls; `git apply` exits with `apply_exit_code`."""

    def __init__(self, apply_exit_code: int = 0, output: str = "") -> None:
        self.apply_exit_code = apply_exit_code
        self.output = output
        self.files: dict[str, str] = {}
        self.commands: list[str] = []

    def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    def exec(self, cmd: str, timeout: float = 300) -> ExecResult:
        self.commands.append(cmd)
        is_apply = cmd.startswith("git apply")
        return ExecResult(
            exit_code=self.apply_exit_code if is_apply else 0,
            output=self.output if is_apply else "",
            runtime_s=0.1,
            timed_out=False,
        )


def _apply(fake: FakeSandbox, diff: str) -> Any:
    return apply_patch(cast(Sandbox, fake), diff)


def test_empty_diff_is_noop() -> None:
    fake = FakeSandbox()
    result = _apply(fake, "  \n")
    assert result.applied and result.error is None
    assert fake.commands == []


def test_writes_diff_outside_repo_with_final_newline_and_removes_it() -> None:
    fake = FakeSandbox()
    result = _apply(fake, DIFF)
    assert result.applied
    [(path, content)] = fake.files.items()
    assert not path.startswith("/testbed")
    assert content == DIFF + "\n"
    assert fake.commands == [f"git apply --verbose {path}", f"rm -f {path}"]


def test_failure_returns_git_error() -> None:
    fake = FakeSandbox(apply_exit_code=1, output="error: patch failed: x.py:1\n")
    result = _apply(fake, DIFF)
    assert not result.applied
    assert result.error == "error: patch failed: x.py:1"
    assert fake.commands[-1].startswith("rm -f")
