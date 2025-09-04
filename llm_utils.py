"""LLM factory utilities for creating LangChain chat models with pluggable providers.

Supports provider-prefixed model strings like:
- "openai/gpt-4o"
- "anthropic/claude-3-opus-20240229"
- "groq/llama3-70b-8192"
- "google/gemini-1.5-pro"

If no provider prefix is given (e.g., "gpt-4"), the provider is chosen by:
1) LLM_PROVIDER env var (openai|anthropic|groq|google)
2) heuristic based on model name (defaults to openai)

All returned objects are LangChain chat models.
"""
from __future__ import annotations

import os
from typing import Any, Tuple


def _parse_provider_and_model(model: str) -> Tuple[str, str]:
    # Provider can be given as "provider/model"
    if "/" in model:
        provider, raw = model.split("/", 1)
        return provider.lower(), raw

    # Else use env override or heuristics
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    m = model.lower()
    if provider not in {"openai", "anthropic", "groq", "google"}:
        # Heuristic fallbacks
        if m.startswith("gpt") or m.startswith("o-"):
            provider = "openai"
        elif "claude" in m:
            provider = "anthropic"
        elif any(x in m for x in ["llama", "mixtral", "gemma"]):
            # Could be Groq or Google; default to Groq for llama/mixtral, Google for gemma
            provider = "groq" if any(x in m for x in ["llama", "mixtral"]) else "google"
        elif "gemini" in m:
            provider = "google"
        else:
            provider = "openai"

    return provider, model


def _apply_param_aliases_for_google(kwargs: dict) -> dict:
    # ChatGoogleGenerativeAI uses max_output_tokens instead of max_tokens
    if "max_tokens" in kwargs and "max_output_tokens" not in kwargs:
        kwargs = dict(kwargs)
        kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
    return kwargs


def _supports_temperature(provider: str, model: str) -> bool:
    """Return False for models where temperature must not be set.

    Currently, OpenAI reasoning models (o1, o3, o4 families) require default temperature
    and may reject explicit temperature values, even 1.0.
    """
    m = model.lower()
    if provider == "openai":
        # Disallow for o1*, o3*, o4* families
        if m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
            return False
    return True


def _sanitize_kwargs_for_model(provider: str, model: str, kwargs: dict) -> dict:
    if not _supports_temperature(provider, model):
        if "temperature" in kwargs:
            k = dict(kwargs)
            k.pop("temperature", None)
            return k
    return kwargs


def _call_with_model_or_model_name(ctor, model: str, kwargs: dict):
    # LangChain provider classes sometimes expect model or model_name depending on version
    try:
        return ctor(model=model, **kwargs)
    except TypeError:
        return ctor(model_name=model, **kwargs)


def make_chat_model(model: str, **kwargs: Any):
    """Create a LangChain chat model for the given provider/model string.

    model can be prefixed with a provider (openai|anthropic|groq|google), e.g. "anthropic/claude-3".
    If no prefix, LLM_PROVIDER env var or heuristics choose a provider.
    """
    provider, raw_model = _parse_provider_and_model(model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        # Support base_url override via env if needed
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_OPENAI_BASE_URL")
        k = dict(kwargs)
        if base_url:
            k.setdefault("base_url", base_url)
        k = _sanitize_kwargs_for_model(provider, raw_model, k)
        return _call_with_model_or_model_name(ChatOpenAI, raw_model, k)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "langchain-anthropic is not installed. Please `pip install langchain-anthropic`."
            ) from e
        return _call_with_model_or_model_name(ChatAnthropic, raw_model, kwargs)

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "langchain-groq is not installed. Please `pip install langchain-groq`."
            ) from e
        return _call_with_model_or_model_name(ChatGroq, raw_model, kwargs)

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "langchain-google-genai is not installed. Please `pip install langchain-google-genai`."
            ) from e
        k = _apply_param_aliases_for_google(kwargs)
        return _call_with_model_or_model_name(ChatGoogleGenerativeAI, raw_model, k)

    raise ValueError(f"Unsupported provider: {provider}")


# Backwards-compatible alias used in the codebase
def make_chat_openai(model: str, **kwargs: Any):
    """Alias to the generic factory, so the rest of the app can be provider-agnostic."""
    return make_chat_model(model, **kwargs)

