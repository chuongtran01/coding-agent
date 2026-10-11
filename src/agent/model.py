"""Build the chat model from config, and price its token usage.

Every provider difference lives here. The rest of the agent sees a LangChain
`BaseChatModel` and `UsageMetadata`, so switching models is a config change.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel

from agent.config import AgentConfig

MAX_OUTPUT_TOKENS = 16_000  # per model turn; the run budget is AgentConfig.max_tokens
MAX_RETRIES = 6  # SDK retries with backoff on rate limits and 5xx

_API_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "google_genai": "GOOGLE_API_KEY"}


class Price(BaseModel):
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float  # Gemini's implicit caching has no write charge: same as input


# Prices as of 2026-10. Tiers are (max prompt tokens, price), checked in order.
# A model's prompt size is the input tokens of one turn, cached tokens included.
_PRICES: dict[str, list[tuple[int | None, Price]]] = {
    # Launch price through 2026-12-31; it doubles from 2027-01-01.
    "google_genai:gemini-3.8-flash": [
        (None, Price(input=0.75, output=3.75, cache_read=0.075, cache_write=0.75)),
    ],
    "anthropic:claude-haiku-5-5": [
        (100_000, Price(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125)),
        (None, Price(input=0.50, output=2.50, cache_read=0.05, cache_write=0.625)),
    ],
    "anthropic:claude-sonnet-5-5": [
        (None, Price(input=2.0, output=10.0, cache_read=0.20, cache_write=2.50)),
    ],
    "anthropic:claude-opus-5-5": [
        (None, Price(input=4.0, output=20.0, cache_read=0.20, cache_write=5.0)),
    ],
}


def provider_of(model: str) -> str:
    """Return the provider half of a `"provider:model"` string."""
    provider, sep, name = model.partition(":")
    if not sep or not provider or not name:
        raise ValueError(f"model must be 'provider:model', got {model!r}")
    return provider


def make_model(config: AgentConfig) -> BaseChatModel:
    """Build the chat model named by `config.model`.

    Reads API keys from `.env`; only the chosen provider's key is required.
    Claude gets automatic prompt caching (a breakpoint on the last block of each
    request, so every turn reads the previous turn's prefix from cache). Recent
    Gemini models cache implicitly and need nothing.
    """
    load_dotenv()
    provider = provider_of(config.model)
    if config.model not in _PRICES:
        raise ValueError(f"no price for {config.model!r}; add it to _PRICES")
    key = _API_KEYS.get(provider)
    if key and not os.environ.get(key):
        raise RuntimeError(f"{key} is not set; add it to .env")

    kwargs: dict[str, Any] = {}
    if provider == "anthropic":
        # model_kwargs survive bind_tools(); a bind(cache_control=...) would not.
        kwargs["model_kwargs"] = {"cache_control": {"type": "ephemeral"}}
    elif provider == "google_genai":
        # Gemini hides its reasoning unless asked; the summaries make trajectories
        # explain why the agent did what it did.
        kwargs["include_thoughts"] = True
    model: BaseChatModel = init_chat_model(
        config.model,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_retries=MAX_RETRIES,
        **kwargs,
    )
    return model


def cost_usd(model: str, usage: UsageMetadata) -> float:
    """Price one model turn's usage.

    LangChain's `input_tokens` includes cached tokens for both providers, so the
    uncached part is what remains after cache reads and writes.
    """
    details = usage.get("input_token_details", {})
    cache_read = details.get("cache_read") or 0
    cache_write = (details.get("cache_creation") or 0) + sum(
        details.get(k) or 0  # type: ignore[misc]
        for k in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
    )
    uncached = usage["input_tokens"] - cache_read - cache_write
    price = _price(model, usage["input_tokens"])
    return (
        uncached * price.input
        + cache_read * price.cache_read
        + cache_write * price.cache_write
        + usage["output_tokens"] * price.output
    ) / 1_000_000


def _price(model: str, prompt_tokens: int) -> Price:
    for max_prompt, price in _PRICES[model]:
        if max_prompt is None or prompt_tokens <= max_prompt:
            return price
    raise AssertionError("price tiers must end with an open tier")
