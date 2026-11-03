"""
llm_client.py
=============
Thin wrapper around the Gemini API (``google-genai`` SDK) with a model fallback chain.

Why this exists: Gemini model ids are retired on a fixed schedule, and a retired id answers with a
404. Instead of a hardcoded id inside the report generator, every LLM call goes through
:class:`LLMClient`, which tries the configured model first, then each fallback, and raises
:class:`LLMUnavailableError` only if all of them fail. Callers (notice generator, analytics
insights) catch that error and degrade to a deterministic template, so an LLM outage never blocks
the enforcement pipeline.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from src.config import (
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODELS,
    GEMINI_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
    get_logger,
)

logger = get_logger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when no configured Gemini model could produce a response."""


class MissingApiKeyError(LLMUnavailableError, ValueError):
    """Raised when ``GEMINI_API_KEY`` is not configured (also a ValueError, for callers that expect one)."""


class LLMClient:
    """Gemini text-generation client with a fallback chain of model ids."""

    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        models: Optional[Sequence[str]] = None,
        client: Any = None,
        temperature: float = LLM_TEMPERATURE,
        timeout_s: int = LLM_TIMEOUT_S,
    ) -> None:
        """Create the client; the network SDK is only imported on first use.

        Args:
            api_key: Gemini API key (``GEMINI_API_KEY`` env var by default).
            models: Ordered model ids to try. Defaults to the configured model plus fallbacks.
            client: A pre-built ``google.genai.Client`` (or a test double with the same
                ``models.generate_content`` method).
            temperature: Sampling temperature; kept low so notices stay factual.
            timeout_s: Per-request timeout in seconds.
        """
        self.api_key = api_key
        self.models: List[str] = list(models) if models else [GEMINI_MODEL_NAME, *GEMINI_FALLBACK_MODELS]
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._client = client

    @property
    def available(self) -> bool:
        """True if a client was injected or an API key is configured."""
        return self._client is not None or bool(self.api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise MissingApiKeyError("GEMINI_API_KEY is not set. Add it to your .env file (see .env.example).")
            from google import genai  # imported lazily: heavy, and unused when the LLM is off

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> Tuple[str, str]:
        """Generate text, trying each model in the chain until one answers.

        Args:
            prompt: The user prompt.
            system_instruction: Optional system prompt.

        Returns:
            ``(text, model_id)`` for the first model that returned non-empty text.

        Raises:
            LLMUnavailableError: If no key is configured or every model failed.
        """
        client = self._get_client()
        errors: List[str] = []
        for model in self.models:
            try:
                config: Any = {"temperature": self.temperature}
                if system_instruction:
                    config["system_instruction"] = system_instruction
                try:  # typed config when the real SDK is present; plain dict works for test doubles
                    from google.genai import types

                    config = types.GenerateContentConfig(**config)
                except ImportError:
                    pass
                response = client.models.generate_content(model=model, contents=prompt, config=config)
                text = (getattr(response, "text", None) or "").strip()
                if not text:
                    raise ValueError("empty response")
                return text, model
            except LLMUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001 - 404 retired model, 429 quota, 5xx, safety block...
                errors.append(f"{model}: {type(exc).__name__}: {exc}")
                logger.warning("LLM model %s failed (%s); trying next fallback", model, exc)
        raise LLMUnavailableError("All Gemini models failed: " + " | ".join(errors))
