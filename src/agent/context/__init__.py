"""Context strategies: code chosen from the issue and the repo before the agent starts.

A strategy runs in the sandbox while the repo is at its base commit, sees only the
`AgentTask` and a read-only `RepoView`, and returns a `Context` whose text goes in
the first user message. Everything else (system prompt, tools, loop, limits) stays
the same, so strategies can be compared on the same tasks.
"""

from agent.context.base import Context, ContextItem, ContextStrategy
from agent.context.repo import GrepMatch, RepoError, RepoView
from agent.context.strategies import (
    STRATEGIES,
    UnknownStrategyError,
    build_context,
    get_strategy,
)

__all__ = [
    "STRATEGIES",
    "Context",
    "ContextItem",
    "ContextStrategy",
    "GrepMatch",
    "RepoError",
    "RepoView",
    "UnknownStrategyError",
    "build_context",
    "get_strategy",
]
