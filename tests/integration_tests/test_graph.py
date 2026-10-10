import pytest

from agent.graph import State, graph

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_simple_passthrough() -> None:
    res = await graph.ainvoke(State(changeme="some_val"))
    assert res is not None
