from pathlib import Path

import pytest

from agent.context import (
    STRATEGIES,
    Context,
    RepoView,
    UnknownStrategyError,
    build_context,
    get_strategy,
)
from agent.harness.dataset import AgentTask

from .shell import LocalShell

TASK = AgentTask(
    instance_id="psf__requests-0000",
    repo="psf/requests",
    base_commit="a" * 40,
    problem_statement="Something is broken.",
)


def test_registry_is_keyed_by_name() -> None:
    assert "none" in STRATEGIES
    for name, strategy in STRATEGIES.items():
        assert strategy.name == name
        assert get_strategy(name) is strategy


def test_unknown_strategy_lists_the_known_ones() -> None:
    with pytest.raises(UnknownStrategyError, match="'nope'; choose from none"):
        get_strategy("nope")


def test_none_is_empty(tmp_path: Path) -> None:
    repo = RepoView(LocalShell(tmp_path))
    context = get_strategy("none").build(TASK, repo, 24_000)
    assert context == Context(strategy="none", text="", items=[])


def test_build_context_times_the_strategy(tmp_path: Path) -> None:
    repo = RepoView(LocalShell(tmp_path))
    context = build_context(get_strategy("none"), TASK, repo, 24_000)
    assert context.strategy == "none" and context.build_s >= 0
