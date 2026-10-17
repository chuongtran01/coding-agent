"""The agent loop: a model node and a tool node, run until submit or a limit.

    model ──tool calls──► tools ──► model ...
      │ no tool calls: nudge once, then stop if the next reply has none either
      └ submit, or a limit (turns, tokens, wall clock), or a model error: stop

Limits are checked before each model call, so a run stops cleanly between turns
and the caller can still capture the diff. Messages are only ever appended, never
edited, which keeps provider prompt caches and thinking blocks valid.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from agent.config import AgentConfig
from agent.context import Context
from agent.harness.dataset import AgentTask
from agent.model import cost_usd
from agent.prompts import initial_messages
from agent.tools import Toolbox

StopReason = Literal[
    "submitted", "no_tool_calls", "max_turns", "max_tokens", "max_wall_clock", "error"
]

NUDGE = (
    "You replied without calling a tool. Keep working with the tools, or call "
    "submit if the fix is in place."
)
MAX_NUDGES = 1  # per stall: a turn with tool calls resets the count
LOW_TURNS_LEFT = 5  # from here the turn note tells the agent to finish

# LangGraph's END, typed as a Literal so the routers' return types can include it
# (END itself is a plain str). LangGraph reads those return types to build the
# conditional edges.
_END: Literal["__end__"] = "__end__"


class ToolCall(BaseModel):
    """One tool call as the agent made it and as the model saw the result."""

    turn: int  # the model turn that made the call, 1-based
    name: str
    args: dict[str, Any]
    output: str  # what the model saw, after truncation
    is_error: bool
    truncated: bool
    exit_code: int | None
    runtime_s: float


class AgentResult(BaseModel):
    """Everything one agent run produced, before the diff is captured and scored."""

    stop_reason: StopReason
    error: str | None  # set when stop_reason is "error"
    turns: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    messages: list[AnyMessage]
    tool_calls: list[ToolCall]
    runtime_s: float


class State(TypedDict):
    """Graph state. Lists are appended to; the counters hold running totals."""

    messages: Annotated[list[AnyMessage], operator.add]
    tool_calls: Annotated[list[ToolCall], operator.add]
    turns: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    nudges_since_tool_call: int
    started: float  # time.monotonic() at the start of the run
    stop_reason: StopReason | None
    error: str | None


class StateUpdate(TypedDict, total=False):
    """What a node returns: the State keys it changes, all optional.

    Mirrors State minus `started`, which no node changes. Keep the two in step;
    a key or type that is not here is a mypy error in the node that returns it.
    """

    messages: list[AnyMessage]
    tool_calls: list[ToolCall]
    turns: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    nudges_since_tool_call: int
    stop_reason: StopReason
    error: str


def build_graph(
    model: BaseChatModel, toolbox: Toolbox, config: AgentConfig
) -> CompiledStateGraph[State, None, State, State]:
    """Compile the loop for one run, with `model` already built from `config`."""
    bound = model.bind_tools(toolbox.tools())

    def call_model(state: State) -> StateUpdate:
        reason = _limit_hit(state, config)
        if reason:
            return {"stop_reason": reason}
        try:
            reply = bound.invoke(state["messages"])
        except Exception as e:  # the SDKs have already retried rate limits and 5xx
            return {"stop_reason": "error", "error": f"{type(e).__name__}: {e}"}
        usage = reply.usage_metadata or {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        return {
            "messages": [reply],
            "turns": state["turns"] + 1,
            "input_tokens": state["input_tokens"] + usage["input_tokens"],
            "output_tokens": state["output_tokens"] + usage["output_tokens"],
            "cost_usd": state["cost_usd"] + cost_usd(config.model, usage),
        }

    def run_tools(state: State) -> StateUpdate:
        reply = state["messages"][-1]
        assert isinstance(reply, AIMessage)
        messages: list[AnyMessage] = []
        records: list[ToolCall] = []
        submitted = False
        # Run calls in order, all in this step: every call needs a result, and
        # parallel calls' results must come back together. A submit next to other
        # calls takes effect after them.
        for call in reply.tool_calls:
            result = toolbox.run(call["name"], call["args"])
            submitted = submitted or (call["name"] == "submit" and not result.is_error)
            messages.append(_tool_message(call["id"], result.output, result.is_error))
            records.append(
                ToolCall(
                    turn=state["turns"],
                    name=call["name"],
                    args=call["args"],
                    **result.model_dump(),
                )
            )
        # Calls whose arguments were not valid JSON still need a result, so the
        # model sees what went wrong and can try again.
        for bad in reply.invalid_tool_calls:
            output = (
                f"Error: could not parse the arguments for {bad['name']}: "
                f"{bad['error'] or 'invalid JSON'}. Call the tool again with valid "
                "JSON arguments."
            )
            messages.append(_tool_message(bad["id"], output, is_error=True))
            records.append(
                ToolCall(
                    turn=state["turns"],
                    name=bad["name"] or "",
                    args={"raw": bad["args"]},
                    output=output,
                    is_error=True,
                    truncated=False,
                    exit_code=None,
                    runtime_s=0.0,
                )
            )
        # The model cannot see the turn limit otherwise. One note per step, on the
        # last result; the ToolCall records keep the tools' own output.
        if messages and not submitted:
            last = messages[-1]
            last.content = (
                f"{last.content}\n\n{turn_note(state['turns'], config.max_turns)}"
            )
        update: StateUpdate = {
            "messages": messages,
            "tool_calls": records,
            "nudges_since_tool_call": 0,
        }
        if submitted:
            update["stop_reason"] = "submitted"
        return update

    def nudge(state: State) -> StateUpdate:
        return {
            "messages": [HumanMessage(NUDGE)],
            "nudges_since_tool_call": state["nudges_since_tool_call"] + 1,
        }

    def give_up(state: State) -> StateUpdate:
        return {"stop_reason": "no_tool_calls"}

    def after_model(state: State) -> Literal["tools", "nudge", "give_up", "__end__"]:
        if state["stop_reason"]:
            return _END
        reply = state["messages"][-1]
        assert isinstance(reply, AIMessage)
        if reply.tool_calls or reply.invalid_tool_calls:
            return "tools"
        if state["nudges_since_tool_call"] < MAX_NUDGES:
            return "nudge"
        return "give_up"

    def after_tools(state: State) -> Literal["model", "__end__"]:
        return _END if state["stop_reason"] else "model"

    graph: StateGraph[State, None, State, State] = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("nudge", nudge)
    graph.add_node("give_up", give_up)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model)
    graph.add_conditional_edges("tools", after_tools)
    graph.add_edge("nudge", "model")
    graph.add_edge("give_up", END)
    return graph.compile(name="agent")


def run_agent(
    task: AgentTask,
    model: BaseChatModel,
    toolbox: Toolbox,
    config: AgentConfig,
    context: Context | None = None,
) -> AgentResult:
    """Run the loop on `task` until it stops; never raises for model or tool errors.

    `context` is a strategy's code for the first message; None adds nothing.
    """
    started = time.monotonic()
    initial: State = {
        "messages": initial_messages(task, context),
        "tool_calls": [],
        "turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "nudges_since_tool_call": 0,
        "started": started,
        "stop_reason": None,
        "error": None,
    }
    # Each turn takes up to four graph steps (model, tools or nudge, ...); the turn
    # limit stops the run well before LangGraph's own recursion limit would.
    # invoke() is typed as returning a plain dict; the graph's output is a State.
    final = cast(
        State,
        build_graph(model, toolbox, config).invoke(
            initial, {"recursion_limit": 4 * config.max_turns + 10}
        ),
    )
    # Every path to END sets a stop reason; if one does not, that is a bug here.
    if final["stop_reason"] is None:
        raise AssertionError("agent loop ended without a stop reason")
    return AgentResult(
        stop_reason=final["stop_reason"],
        error=final["error"],
        turns=final["turns"],
        input_tokens=final["input_tokens"],
        output_tokens=final["output_tokens"],
        cost_usd=final["cost_usd"],
        messages=final["messages"],
        tool_calls=final["tool_calls"],
        runtime_s=time.monotonic() - started,
    )


def turn_note(turns_used: int, max_turns: int) -> str:
    """Return the note that tells the agent how many turns it has left."""
    left = max(max_turns - turns_used, 0)
    note = f"[Turn {turns_used} of {max_turns} used; {left} left.]"
    if left <= LOW_TURNS_LEFT:
        note += " Make your change now if it is not in place, then call submit."
    return note


def _limit_hit(state: State, config: AgentConfig) -> StopReason | None:
    if state["turns"] >= config.max_turns:
        return "max_turns"
    if state["input_tokens"] + state["output_tokens"] >= config.max_tokens:
        return "max_tokens"
    if time.monotonic() - state["started"] >= config.max_wall_clock_s:
        return "max_wall_clock"
    return None


def _tool_message(call_id: str | None, output: str, is_error: bool) -> ToolMessage:
    return ToolMessage(
        content=output,
        tool_call_id=call_id or "",
        status="error" if is_error else "success",
    )
