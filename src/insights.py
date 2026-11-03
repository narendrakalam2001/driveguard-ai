"""
insights.py
===========
Plain-language analytics insights for the dashboard ("Peak violation hours...", "Trucks account for
34% of severe violations...").

Every number in the text comes from the database, never from the model: the rule-based writer builds
sentences straight from the statistics, and the optional Gemini path is only asked to *rephrase* the
facts it is given. If the LLM is unavailable the rule-based text is returned, so the panel never
shows an error to a control-room operator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import get_logger
from src.llm_client import LLMClient, LLMUnavailableError

logger = get_logger(__name__)

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MIN_VIOLATIONS_FOR_INSIGHTS = 5


def _fmt_hour(h: int) -> str:
    suffix = "AM" if h < 12 else "PM"
    return f"{(h % 12) or 12} {suffix}"


def build_facts(summary: Dict[str, Any], detailed: Dict[str, Any]) -> List[str]:
    """Turn analytics dicts into a list of short, verified factual statements.

    Args:
        summary: Output of ``AlertSystem.get_analytics_summary``.
        detailed: Output of ``AlertSystem.get_detailed_analytics``.

    Returns:
        Fact sentences; empty if there are too few violations to say anything meaningful.
    """
    total = summary.get("total_violations", 0)
    if total < _MIN_VIOLATIONS_FOR_INSIGHTS:
        return []
    facts: List[str] = [f"{total} violations are logged in total."]

    peak = detailed.get("peak_window")
    if peak:
        facts.append(f"Violations peak between {_fmt_hour(peak['start_hour'])} and {_fmt_hour(peak['end_hour'])} (local time).")

    sev = detailed.get("severity_counts") or {}
    if sev:
        top = max(sev, key=sev.get)
        facts.append(f"{round(100 * sev[top] / total)}% of violations are {top}.")
        if sev.get("SEVERE"):
            facts.append(f"{sev['SEVERE']} violations are SEVERE (more than 50% over the limit).")

    for row in detailed.get("vehicle_share", []):
        traffic, severe = row.get("traffic_share_pct"), row["severe_share_pct"]
        if traffic and severe >= 15 and severe >= 1.5 * traffic:
            facts.append(
                f"{row['vehicle_type']}s account for {severe:.0f}% of SEVERE violations but only {traffic:.0f}% of tracked traffic."
            )

    heat = detailed.get("heatmap") or []
    if heat and any(any(r) for r in heat):
        day_totals = [sum(r) for r in heat]
        d = max(range(7), key=lambda i: day_totals[i])
        facts.append(f"{_DAYS[d]} has the most violations ({day_totals[d]}).")
    return facts


def rule_based_insight(summary: Dict[str, Any], detailed: Dict[str, Any]) -> str:
    """Write an insight paragraph directly from the statistics (no LLM).

    Args:
        summary: Output of ``AlertSystem.get_analytics_summary``.
        detailed: Output of ``AlertSystem.get_detailed_analytics``.

    Returns:
        A short paragraph, or a message saying more data is needed.
    """
    facts = build_facts(summary, detailed)
    if not facts:
        return "Not enough violations logged yet to draw reliable conclusions (need at least 5)."
    text = " ".join(facts)
    if detailed.get("peak_window"):
        text += " Consider scheduling additional enforcement around the peak window."
    return text


def generate_insights(
    summary: Dict[str, Any], detailed: Dict[str, Any], llm: Optional[LLMClient] = None, use_llm: bool = True
) -> Dict[str, Optional[str]]:
    """Produce the dashboard insight text, preferring the LLM and degrading to rules.

    Args:
        summary: Output of ``AlertSystem.get_analytics_summary``.
        detailed: Output of ``AlertSystem.get_detailed_analytics``.
        llm: LLM client; a default one is created if None.
        use_llm: Set False to force the rule-based writer.

    Returns:
        ``{"text", "source", "model"}`` where ``source`` is ``"llm"`` or ``"rules"``.
    """
    facts = build_facts(summary, detailed)
    rules_text = rule_based_insight(summary, detailed)
    if not facts or not use_llm:
        return {"text": rules_text, "source": "rules", "model": None}

    client = llm or LLMClient()
    if not client.available:
        return {"text": rules_text, "source": "rules", "model": None}
    prompt = (
        "You are a traffic-enforcement analyst writing 3-4 sentences for a control-room dashboard.\n"
        "Use ONLY the verified facts below. Do not add numbers, causes or facts that are not listed. "
        "End with one concrete enforcement recommendation. Plain text, no markdown.\n\nFACTS:\n- "
        + "\n- ".join(facts)
    )
    try:
        text, model = client.generate(prompt)
        return {"text": text, "source": "llm", "model": model}
    except LLMUnavailableError as exc:
        logger.warning("Insight LLM unavailable, using rules: %s", exc)
        return {"text": rules_text, "source": "rules", "model": None}
