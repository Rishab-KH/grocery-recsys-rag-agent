"""
Central configuration for the RAG agent stack.

Loads environment variables from .env and exposes a pre-configured
LLM client and LangSmith tracing settings.

Usage:
    from rag_agent.config import get_llm, settings
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env from project root (works regardless of where Python is invoked from)
load_dotenv()


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example to .env and fill in your credentials."
        )
    return value


class Settings:
    """Validated environment settings. Raises on first missing required key."""

    # OpenAI
    openai_api_key: str = property(lambda self: _require("OPENAI_API_KEY"))

    # LangSmith — optional; tracing is disabled if not set
    langsmith_tracing: bool  = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langsmith_project: str   = os.getenv("LANGCHAIN_PROJECT", "instacart-recsys")
    langsmith_endpoint: str  = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


settings = Settings()


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """
    Return a configured ChatOpenAI instance.

    LangSmith tracing is picked up automatically via LANGCHAIN_* env vars
    — no extra code needed.

    Args:
        model:       OpenAI model name (default: gpt-4o-mini for cost efficiency).
        temperature: Sampling temperature (default: 0.0 for deterministic output).
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=_require("OPENAI_API_KEY"),
    )
