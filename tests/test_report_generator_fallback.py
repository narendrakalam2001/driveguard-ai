"""Merged ReportGenerator + LLMClient behaviour: retired-model fallback and total LLM outage."""

import pytest

from src.llm_client import LLMUnavailableError
from src.report_generator import ReportGenerator
from tests.test_report_generator_a2 import GOOD_NOTICE, NoRag, StubClient, _record  # reuse the a2 fixtures


class RetiredThenWorks:
    """First model id answers 404 (retired), the second answers with a good notice."""

    def __init__(self):
        self.calls = []
        outer = self

        class _Models:
            def generate_content(self, model, contents, config=None):
                outer.calls.append(model)
                if model == "retired-model":
                    raise RuntimeError("404 NOT_FOUND: model is no longer available")

                class _R:
                    text = GOOD_NOTICE

                return _R()

        self.models = _Models()


def _gen(client, model="retired-model"):
    gen = ReportGenerator(rag_engine=NoRag(), api_key="k", model_name=model, client=client)
    gen._rag_unavailable = True
    return gen


def test_retired_model_falls_through_to_configured_fallback():
    client = RetiredThenWorks()
    notice = _gen(client).generate(_record(), "TRK_007")
    assert client.calls[0] == "retired-model" and len(client.calls) == 2
    assert notice.model_used != "retired-model" and "Section 183" in notice.notice_text


def test_all_models_failing_raises_so_the_api_can_use_the_template():
    with pytest.raises(LLMUnavailableError):
        _gen(StubClient(error=RuntimeError("503"))).generate(_record(), "TRK_007")


def test_template_notice_is_still_available_when_llm_is_down():
    notice = _gen(StubClient(error=RuntimeError("503"))).generate_fallback_notice(_record(), "TRK_007")
    assert notice.model_used == "fallback_template" and "MH12AB1234" in notice.notice_text
