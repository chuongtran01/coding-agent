"""Apply a unified diff to the repo inside a sandbox."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from agent.harness.sandbox import Sandbox

# Patch files live outside /testbed so they never show up in the repo's own diff.
_PATCH_DIR = "/tmp/agent_patches"
_APPLY_TIMEOUT_S = 60


class ApplyResult(BaseModel):
    """Whether a patch applied, and git's error output if it did not."""

    applied: bool
    error: str | None = None


def apply_patch(sandbox: Sandbox, diff: str) -> ApplyResult:
    """Apply `diff` to the repo with `git apply`, all or nothing.

    An empty diff is a no-op that counts as applied: there is nothing to reject,
    and the tests decide whether doing nothing resolves the task (it never should).
    On failure the working tree is left unchanged and `error` holds git's message.
    """
    if not diff.strip():
        return ApplyResult(applied=True)
    if not diff.endswith("\n"):
        diff += "\n"  # git apply reports "corrupt patch" without a final newline

    path = f"{_PATCH_DIR}/{uuid.uuid4().hex}.diff"
    sandbox.write_file(path, diff)
    try:
        result = sandbox.exec(f"git apply --verbose {path}", timeout=_APPLY_TIMEOUT_S)
    finally:
        sandbox.exec(f"rm -f {path}", timeout=_APPLY_TIMEOUT_S)

    if result.exit_code == 0:
        return ApplyResult(applied=True)
    if result.timed_out:
        return ApplyResult(
            applied=False, error=f"git apply timed out after {_APPLY_TIMEOUT_S}s"
        )
    return ApplyResult(applied=False, error=result.output.strip() or "git apply failed")
