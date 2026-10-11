from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END

from agent.config import AgentConfig
from agent.graph import _END, NUDGE, AgentResult, build_graph, run_agent, turn_note
from agent.harness.dataset import AgentTask
from agent.tools import Toolbox

from .fakes import ScriptedModel, ai, call
from .shell import LocalShell

TASK = AgentTask(
    instance_id="psf__requests-0000",
    repo="psf/requests",
    base_commit="a" * 40,
    problem_statement="Something is broken.",
)
CONFIG = AgentConfig(model="anthropic:claude-sonnet-5-5", max_turns=5)


@pytest.fixture
def toolbox(tmp_path: Path) -> Toolbox:
    (tmp_path / "mod.py").write_text("x = 1\n")
    return Toolbox(LocalShell(tmp_path), bash_timeout_s=5, workdir=str(tmp_path))


def run(
    toolbox: Toolbox, replies: list[Any], **config: Any
) -> tuple[AgentResult, ScriptedModel]:
    model = ScriptedModel(replies=replies, seen=[])
    result = run_agent(TASK, model, toolbox, CONFIG.model_copy(update=config))
    return result, model


def test_runs_tools_until_submit(toolbox: Toolbox, tmp_path: Path) -> None:
    result, model = run(
        toolbox,
        [
            ai(call("read", "c1", path="mod.py")),
            ai(call("edit", "c2", path="mod.py", old="x = 1", new="x = 2")),
            ai(call("submit", "c3")),
        ],
    )
    assert result.stop_reason == "submitted"
    assert result.turns == 3
    assert (tmp_path / "mod.py").read_text() == "x = 2\n"
    assert [c.name for c in result.tool_calls] == ["read", "edit", "submit"]
    assert [c.turn for c in result.tool_calls] == [1, 2, 3]
    # The first call sees the system prompt and the issue.
    first = model.seen[0]
    assert isinstance(first[0], SystemMessage) and isinstance(first[1], HumanMessage)
    assert "Something is broken." in str(first[1].content)


def test_counts_tokens_and_cost(toolbox: Toolbox) -> None:
    result, _ = run(
        toolbox, [ai(call("bash", "c1", command="true")), ai(call("submit", "c2"))]
    )
    assert result.input_tokens == 200 and result.output_tokens == 20
    # Sonnet 5.5: 200 input at $2/M + 20 output at $10/M.
    assert result.cost_usd == pytest.approx((200 * 2.0 + 20 * 10.0) / 1e6)


def test_parallel_calls_answered_together_in_order(toolbox: Toolbox) -> None:
    result, model = run(
        toolbox,
        [
            ai(
                call("bash", "a", command="echo one"),
                call("bash", "b", command="echo two"),
            ),
            ai(call("submit", "c")),
        ],
    )
    second = model.seen[1]
    results = [m for m in second if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in results] == ["a", "b"]
    assert isinstance(second[-3], AIMessage)  # both results follow the one reply
    assert "one" in str(results[0].content) and "two" in str(results[1].content)


def test_submit_runs_after_other_calls_in_the_same_reply(
    toolbox: Toolbox, tmp_path: Path
) -> None:
    result, _ = run(
        toolbox,
        [
            ai(
                call("submit", "s"),
                call("write", "w", path="new.py", content="y = 1\n"),
            )
        ],
    )
    assert result.stop_reason == "submitted"
    assert (tmp_path / "new.py").read_text() == "y = 1\n"


def test_tool_errors_go_back_to_the_model(toolbox: Toolbox) -> None:
    result, model = run(
        toolbox,
        [
            ai(call("edit", "c1", path="mod.py", old="nope", new="x")),
            ai(call("submit", "c2")),
        ],
    )
    error = model.seen[1][-1]
    assert isinstance(error, ToolMessage) and error.status == "error"
    assert "not found" in str(error.content)
    assert result.tool_calls[0].is_error


def test_invalid_tool_call_gets_an_error_result(toolbox: Toolbox) -> None:
    broken = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "edit",
                "args": '{"path": "mod.py", "old": ',
                "id": "bad",
                "error": "Unterminated string",
                "type": "invalid_tool_call",
            }
        ],
    )
    result, model = run(toolbox, [broken, ai(call("submit", "c2"))])
    error = model.seen[1][-1]
    assert isinstance(error, ToolMessage) and error.tool_call_id == "bad"
    assert error.status == "error" and "could not parse" in str(error.content)
    assert result.stop_reason == "submitted"


def test_nudges_once_then_stops(toolbox: Toolbox) -> None:
    result, model = run(toolbox, [ai(text="I think it's fixed."), ai(text="Done.")])
    assert result.stop_reason == "no_tool_calls"
    assert result.turns == 2
    assert model.seen[1][-1] == HumanMessage(NUDGE)


def test_nudge_can_get_the_run_going_again(toolbox: Toolbox) -> None:
    result, _ = run(toolbox, [ai(text="Let me think."), ai(call("submit", "c1"))])
    assert result.stop_reason == "submitted"


def test_a_tool_call_resets_the_nudge(toolbox: Toolbox) -> None:
    result, model = run(
        toolbox,
        [
            ai(text="Let me think."),  # nudged
            ai(call("bash", "c1", command="true")),
            ai(text="The fix looks good."),  # nudged again: a new stall
            ai(text="Done."),  # second silent reply in a row: stop
        ],
        max_turns=10,
    )
    assert result.stop_reason == "no_tool_calls"
    assert result.turns == 4
    nudges = [m for m in result.messages if m == HumanMessage(NUDGE)]
    assert len(nudges) == 2


def test_max_turns(toolbox: Toolbox) -> None:
    replies = [ai(call("bash", f"c{i}", command="true")) for i in range(10)]
    result, _ = run(toolbox, replies, max_turns=3)
    assert result.stop_reason == "max_turns"
    assert result.turns == 3
    assert len(result.tool_calls) == 3  # the last turn's calls still ran


def test_max_tokens(toolbox: Toolbox) -> None:
    replies = [
        ai(call("bash", f"c{i}", command="true"), tokens=1000) for i in range(10)
    ]
    result, _ = run(toolbox, replies, max_tokens=2500)
    assert result.stop_reason == "max_tokens"
    assert result.turns == 3  # 1010, 2020, 3030: stops once over budget


def test_max_wall_clock(toolbox: Toolbox) -> None:
    replies = [ai(call("bash", f"c{i}", command="sleep 0.3")) for i in range(10)]
    result, _ = run(toolbox, replies, max_wall_clock_s=0.5)
    assert result.stop_reason == "max_wall_clock"
    assert result.turns == 2


def test_model_error_stops_the_run(toolbox: Toolbox) -> None:
    result, _ = run(
        toolbox, [ai(call("bash", "c1", command="true")), RuntimeError("API down")]
    )
    assert result.stop_reason == "error"
    assert result.error == "RuntimeError: API down"
    assert result.turns == 1


def test_messages_are_only_appended(toolbox: Toolbox) -> None:
    result, model = run(
        toolbox,
        [
            ai(call("bash", "c1", command="true")),
            ai(call("bash", "c2", command="true")),
            ai(call("submit", "c3")),
        ],
    )
    # Every request extends the previous one, so the cached prefix stays valid.
    for before, after in zip(model.seen, model.seen[1:]):
        assert after[: len(before)] == before
    assert result.messages[: len(model.seen[-1])] == model.seen[-1]


def test_end_alias_matches_langgraph() -> None:
    assert _END == END


def test_edges_come_from_the_router_return_types(toolbox: Toolbox) -> None:
    graph = build_graph(ScriptedModel(replies=[]), toolbox, CONFIG).get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert {
        ("model", "tools"),
        ("model", "nudge"),
        ("model", "give_up"),
        ("model", "__end__"),
        ("tools", "model"),
        ("tools", "__end__"),
    } <= edges


def test_result_round_trips_through_json(toolbox: Toolbox) -> None:
    result, _ = run(
        toolbox, [ai(call("bash", "c1", command="echo hi")), ai(call("submit", "c2"))]
    )
    loaded = AgentResult.model_validate_json(result.model_dump_json())
    assert [type(m) for m in loaded.messages] == [type(m) for m in result.messages]
    assert loaded.messages == result.messages
    assert loaded.tool_calls == result.tool_calls


def test_turn_note_counts_down_and_warns_near_the_end() -> None:
    assert turn_note(3, 30) == "[Turn 3 of 30 used; 27 left.]"
    assert turn_note(25, 30).endswith(
        "Make your change now if it is not in place, then call submit."
    )
    assert turn_note(30, 30).startswith("[Turn 30 of 30 used; 0 left.]")


def test_last_tool_result_of_each_step_carries_the_turn_note(toolbox: Toolbox) -> None:
    result, model = run(
        toolbox,
        [
            ai(
                call("bash", "a", command="echo one"),
                call("bash", "b", command="echo two"),
            ),
            ai(call("submit", "c")),
        ],
    )
    first, second = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert "[Turn" not in str(first.content)  # one note per step, not per call
    assert str(second.content).endswith(turn_note(1, CONFIG.max_turns))
    # The records keep what the tool returned, without the note.
    assert "[Turn" not in result.tool_calls[1].output
    # No note after submit: the run is over.
    assert "[Turn" not in str(result.messages[-1].content)
