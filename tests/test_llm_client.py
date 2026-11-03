"""Tests for src/llm_client.py - model fallback chain and failure modes (no network)."""

import pytest

from src.llm_client import LLMClient, LLMUnavailableError


def test_uses_first_model_when_it_works(fake_genai):
    fake = fake_genai(["hello"])
    client = LLMClient(models=["m1", "m2"], client=fake)
    assert client.generate("hi") == ("hello", "m1")
    assert fake.calls == ["m1"]


def test_falls_back_to_next_model_when_first_is_retired(fake_genai):
    fake = fake_genai([RuntimeError("404 model no longer available"), "ok from m2"])
    text, model = LLMClient(models=["old", "new"], client=fake).generate("hi")
    assert (text, model) == ("ok from m2", "new")
    assert fake.calls == ["old", "new"]


def test_raises_when_every_model_fails_and_lists_each_error(fake_genai):
    fake = fake_genai([RuntimeError("quota"), RuntimeError("404")])
    with pytest.raises(LLMUnavailableError) as exc:
        LLMClient(models=["a", "b"], client=fake).generate("hi")
    assert "a:" in str(exc.value) and "b:" in str(exc.value)


def test_empty_response_counts_as_failure(fake_genai):
    fake = fake_genai(["   ", "real answer"])
    assert LLMClient(models=["a", "b"], client=fake).generate("hi")[1] == "b"


def test_missing_api_key_raises_clear_error():
    client = LLMClient(api_key="", models=["a"])
    assert client.available is False
    with pytest.raises(LLMUnavailableError, match="GEMINI_API_KEY"):
        client.generate("hi")


def test_available_true_with_key_or_injected_client(fake_genai):
    assert LLMClient(api_key="k", models=["a"]).available is True
    assert LLMClient(api_key="", models=["a"], client=fake_genai([])).available is True
