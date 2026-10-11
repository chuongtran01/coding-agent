"""Configuration for one agent run."""

from __future__ import annotations

from pydantic import BaseModel

DEFAULT_MODEL = "google_genai:gemini-2.5-flash"


class AgentConfig(BaseModel):
    """Model and limits for one agent run; recorded with every run."""

    model: str = DEFAULT_MODEL  # "provider:model", passed to init_chat_model
    max_turns: int = 30
    max_tokens: int = 1_000_000  # total over the run, input + output
    max_wall_clock_s: float = 1800
    bash_timeout_s: float = 120
