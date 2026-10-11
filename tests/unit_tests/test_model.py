import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages.ai import UsageMetadata
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.config import AgentConfig
from agent.model import MAX_OUTPUT_TOKENS, cost_usd, make_model, provider_of


@tool
def bash(command: str) -> str:
    """Run a command."""
    return ""


@pytest.fixture(autouse=True)
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.model.load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("GOOGLE_API_KEY", "test")


def test_provider_of() -> None:
    assert provider_of("anthropic:claude-sonnet-5-5") == "anthropic"
    for bad in ("claude-sonnet-5-5", ":x", "anthropic:"):
        with pytest.raises(ValueError):
            provider_of(bad)


def test_make_model_gemini() -> None:
    model = make_model(AgentConfig(model="google_genai:gemini-3.8-flash"))
    assert isinstance(model, ChatGoogleGenerativeAI)


def test_make_model_claude_caches_after_bind_tools() -> None:
    model = make_model(AgentConfig(model="anthropic:claude-sonnet-5-5"))
    assert isinstance(model, ChatAnthropic)
    bound = model.bind_tools([bash])
    payload = model._get_request_payload([("user", "hi")], **bound.kwargs)  # type: ignore[attr-defined]
    assert payload["cache_control"] == {"type": "ephemeral"}
    assert payload["max_tokens"] == MAX_OUTPUT_TOKENS
    assert "temperature" not in payload


def test_make_model_needs_only_the_used_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    make_model(AgentConfig(model="google_genai:gemini-3.8-flash"))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_model(AgentConfig(model="anthropic:claude-sonnet-5-5"))


def test_make_model_rejects_unpriced_model() -> None:
    with pytest.raises(ValueError, match="no price"):
        make_model(AgentConfig(model="anthropic:claude-unknown"))


def test_cost_counts_cache_reads_and_writes_separately() -> None:
    usage = UsageMetadata(
        input_tokens=1_000_000,
        output_tokens=100_000,
        total_tokens=1_100_000,
        input_token_details={"cache_read": 600_000, "cache_creation": 100_000},
    )
    # 300k uncached * 2.0 + 600k * 0.20 + 100k * 2.50 + 100k out * 10.0
    assert cost_usd("anthropic:claude-sonnet-5-5", usage) == pytest.approx(
        0.6 + 0.12 + 0.25 + 1.0
    )


def test_cost_uses_the_long_prompt_tier() -> None:
    def usage(n: int) -> UsageMetadata:
        return UsageMetadata(input_tokens=n, output_tokens=0, total_tokens=n)

    model = "anthropic:claude-haiku-5-5"
    assert cost_usd(model, usage(100_000)) == pytest.approx(0.01)
    assert cost_usd(model, usage(200_000)) == pytest.approx(0.10)
