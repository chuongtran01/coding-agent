"""Read-only access to the task's repo, for context strategies."""

from __future__ import annotations

from agent.tools import Shell


class RepoView:
    """The repo in the sandbox, at its base commit, for strategies to read."""

    def __init__(self, shell: Shell) -> None:
        """Wrap the sandbox's shell; strategies must not write through it."""
        self._shell = shell
