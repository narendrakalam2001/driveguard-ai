"""Tests for src/insights.py - facts come from stats, LLM only rephrases, rules are the safe fallback."""

from src.insights import build_facts, generate_insights, rule_based_insight
from src.llm_client import LLMClient

SUMMARY = {"total_violations": 40}
DETAILED = {
    "peak_window": {"start_hour": 17, "end_hour": 19},
    "severity_counts": {"WARNING": 10, "VIOLATION": 20, "SEVERE": 10},
    "vehicle_share": [
        {"vehicle_type": "Truck", "severe_share_pct": 60.0, "traffic_share_pct": 12.0},
        {"vehicle_type": "Car", "severe_share_pct": 40.0, "traffic_share_pct": 70.0},
    ],
    "heatmap": [[0] * 24 for _ in range(7)],
}


def test_too_few_violations_gives_no_facts():
    assert build_facts({"total_violations": 2}, DETAILED) == []
    assert "Not enough" in rule_based_insight({"total_violations": 2}, DETAILED)


def test_facts_contain_peak_and_over_represented_vehicle():
    text = " ".join(build_facts(SUMMARY, DETAILED))
    assert "5 PM and 7 PM" in text
    assert "Trucks account for 60% of SEVERE violations but only 12%" in text
    assert "Cars account" not in text  # under-represented types are not called out


def test_rules_path_when_llm_not_configured():
    out = generate_insights(SUMMARY, DETAILED, llm=LLMClient(api_key="", models=["m"]))
    assert out["source"] == "rules" and out["model"] is None


def test_llm_path_returns_model_text(fake_genai):
    llm = LLMClient(models=["m1"], client=fake_genai(["Peak is 5-7 PM. Add patrols."]))
    out = generate_insights(SUMMARY, DETAILED, llm=llm)
    assert out == {"text": "Peak is 5-7 PM. Add patrols.", "source": "llm", "model": "m1"}


def test_llm_failure_degrades_to_rules(fake_genai):
    llm = LLMClient(models=["m1"], client=fake_genai([RuntimeError("down")]))
    assert generate_insights(SUMMARY, DETAILED, llm=llm)["source"] == "rules"


def test_use_llm_false_never_calls_llm(fake_genai):
    fake = fake_genai([])
    out = generate_insights(SUMMARY, DETAILED, llm=LLMClient(models=["m"], client=fake), use_llm=False)
    assert out["source"] == "rules" and fake.calls == []
