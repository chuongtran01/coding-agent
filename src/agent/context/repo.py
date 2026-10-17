"""Read-only access to the task's repo, for context strategies.

Strategies run before the agent, with the repo at its base commit, so what they
read is the code the agent starts from. `RepoView` only runs reading commands
(`git ls-files`, `cat`, `git grep`); it never writes to the repo.
"""

from __future__ import annotations

import posixpath
import shlex
from collections.abc import Sequence

from pydantic import BaseModel

from agent.tools import MAX_LINE_CHARS, Shell

REPO_TIMEOUT_S = 120
_ERR = "/tmp/.agent-repo.err"  # outside the repo, so it never reaches the diff


class RepoError(Exception):
    """A read failed: a missing file, a bad path, or a git error."""


class GrepMatch(BaseModel):
    """One line `git grep` matched."""

    path: str
    line: int
    text: str  # cut to MAX_LINE_CHARS, like the agent's `read`


class RepoView:
    """The repo in the sandbox, at its base commit, for strategies to read."""

    def __init__(self, shell: Shell) -> None:
        """Wrap the sandbox's shell, whose commands run in the repo."""
        self._shell = shell
        self._python_files: list[str] | None = None

    def python_files(self) -> list[str]:
        """Return the tracked `.py` files, as paths relative to the repo, sorted."""
        if self._python_files is None:
            out = self._run("git ls-files -z -- '*.py'")
            self._python_files = sorted(p for p in out.split("\0") if p)
        return list(self._python_files)

    def read(self, path: str) -> str:
        """Return the text of a file, given as a path relative to the repo."""
        return self._run(f"cat -- {shlex.quote(_relative(path))}")

    def grep(
        self,
        patterns: Sequence[str],
        *,
        regex: bool = False,
        word: bool = False,
        ignore_case: bool = False,
        pathspec: str = "*.py",
    ) -> list[GrepMatch]:
        """Return the lines of tracked files that match any of `patterns`.

        Patterns are fixed strings unless `regex` (extended regex); `word` matches
        whole words only. Binary files are skipped. No match is an empty list.
        """
        if not patterns:
            return []
        flags = ["--null", "-n", "-I", "-E" if regex else "-F"]
        flags += ["-w"] * word + ["-i"] * ignore_case
        args = " ".join(f"-e {shlex.quote(p)}" for p in patterns)
        out = self._run(
            f"git grep {' '.join(flags)} {args} -- {shlex.quote(pathspec)}",
            max_ok_status=1,  # 1 is "no match"
        )
        matches = []
        # Not splitlines(): it also splits on form feeds, which old source files hold.
        for row in out.split("\n"):
            if not row:
                continue
            path, line, text = row.split("\0", 2)
            matches.append(
                GrepMatch(path=path, line=int(line), text=text[:MAX_LINE_CHARS])
            )
        return matches

    def _run(self, cmd: str, *, max_ok_status: int = 0) -> str:
        # stderr goes to a file, so the output is the command's stdout alone.
        result = self._shell.exec(
            f"{cmd} 2> {_ERR}; status=$?; "
            f"if [ $status -gt {max_ok_status} ]; then cat {_ERR}; fi; "
            f"rm -f {_ERR}; exit $status",
            timeout=REPO_TIMEOUT_S,
        )
        if result.timed_out or result.exit_code > max_ok_status:
            raise RepoError(f"`{cmd}` failed ({result.exit_code}):\n{result.output}")
        return result.output


def _relative(path: str) -> str:
    """Return `path` normalised, refusing anything that leaves the repo."""
    normal = posixpath.normpath(path)
    if posixpath.isabs(normal) or normal == ".." or normal.startswith("../"):
        raise RepoError(f"not a path inside the repo: {path!r}")
    return normal
