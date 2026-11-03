"""
report_generator.py
=====================
Generates an official-style traffic violation notice from a classified
violation event.

Grounding: the legal section and fine amounts come from exact lookups in the knowledge base
(:class:`src.rag_engine.StructuredKnowledge`), never from whatever a vector search ranks first;
:class:`src.rag_engine.RAGEngine` (if the Chroma store has been built) only adds explanatory
context. The LLM (Gemini, via ``google-genai`` and :class:`src.llm_client.LLMClient`, which falls back across model ids) writes the prose, and its output is
checked: a notice that cites a section or fine amount that was not supplied, or that drops the
plate or the speed, is rejected and replaced by the deterministic template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.config import GEMINI_API_KEY, GEMINI_FALLBACK_MODELS, GEMINI_MODEL_NAME, get_logger
from src.llm_client import LLMClient
from src.rag_engine import FineInfo, RAGEngine, StructuredKnowledge
from src.violation_classifier import ViolationRecord

logger = get_logger(__name__)

_RUPEE_AMOUNT_RE = re.compile(r"(?:₹|Rs\.?|INR)\s*([\d,]+)")
FALLBACK_MODEL_NAME = "fallback_template"

_SYSTEM_INSTRUCTION = """You are DriveGuard AI's official notice-drafting assistant for Indian \
traffic enforcement. You write formal, factual violation notices for automated CCTV \
speed-enforcement systems.

Rules you must follow:
1. Use ONLY the legal section numbers and fine amounts given in the provided context. \
Never invent a section number or a fine amount that isn't in the context.
2. State facts plainly: vehicle, speed, limit, location, time.
3. Keep the tone formal and neutral — this is an official government-style notice, not an \
accusatory or emotional document.
4. Always include the standard closing line about reporting to the RTO within 30 days.
5. If the plate could not be reliably read, say so explicitly instead of stating a plate number.
6. Do not include markdown formatting — output plain text suitable for printing on a notice.
7. State the fine exactly as given under "Applicable Fine" and mention that amounts are indicative \
and the state-notified amount applies.
"""

_PROMPT_TEMPLATE = """Generate an official traffic violation notice using the following data.

VIOLATION DATA:
- Vehicle ID: {vehicle_id}
- Plate Number: {plate_number}
- Vehicle Type: {vehicle_type}
- Detected Speed: {detected_speed} km/h
- Speed Limit: {speed_limit} km/h
- Excess: {excess_speed} km/h ({excess_percent}% over limit)
- Severity: {severity}
- Location: {location}
- Timestamp: {timestamp}
- Evidence Reference: {evidence_frame}
- Weather: {weather}
- Road Condition: {road_condition}
- Repeat Offender: {is_repeat_offender}

LEGAL CONTEXT (use only this):
{context_block}

Write the notice now."""


@dataclass
class GeneratedNotice:
    """The LLM's output plus the metadata needed to audit it.

    Attributes:
        violation_id: The source violation's UUID.
        notice_text: The generated notice, plain text.
        cited_context: The knowledge-base chunks actually given to the LLM,
            for auditability (what grounded this notice).
        model_used: Name of the LLM model that generated the text, or ``fallback_template``.
    """

    violation_id: str
    notice_text: str
    cited_context: List[str]
    model_used: str


class ReportGenerator:
    """Generates grounded violation notices via Gemini + knowledge-base context."""

    def __init__(
        self,
        rag_engine: Optional[RAGEngine] = None,
        api_key: str = GEMINI_API_KEY,
        model_name: str = GEMINI_MODEL_NAME,
        knowledge: Optional[StructuredKnowledge] = None,
        client=None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        """Configure the knowledge base and the Gemini client.

        Args:
            rag_engine: Optional pre-built :class:`RAGEngine` for extra semantic context. If
                None, one is tried lazily and silently skipped when the Chroma store is absent.
            api_key: Gemini API key (``GEMINI_API_KEY`` env var by default).
            model_name: Gemini model id (``GEMINI_MODEL_NAME`` env var by default).
            knowledge: Pre-built :class:`StructuredKnowledge`; loaded from disk if None.
            client: Pre-built ``google.genai.Client`` (tests); created lazily from ``api_key``.
            llm_client: Pre-built :class:`LLMClient` (overrides ``client``/``api_key``/``model_name``).
        """
        self.api_key = api_key
        self.model_name = model_name
        self._rag_engine = rag_engine
        self._rag_unavailable = False
        self._knowledge = knowledge
        models = list(dict.fromkeys([model_name, *GEMINI_FALLBACK_MODELS]))  # preferred first, no duplicates
        self._llm = llm_client or LLMClient(api_key=api_key, models=models, client=client)

    # ------------------------------------------------------------------ context
    def _get_knowledge(self) -> StructuredKnowledge:
        if self._knowledge is None:
            self._knowledge = StructuredKnowledge()
        return self._knowledge

    def _extra_context(self, violation_type: str) -> List[str]:
        """Semantic FAQ context from Chroma; empty (never an error) if the store is not built."""
        if self._rag_unavailable:
            return []
        try:
            if self._rag_engine is None:
                self._rag_engine = RAGEngine()
            chunks = self._rag_engine.retrieve(violation_type.replace("_", " ") + " penalty", top_k=1, doc_type="faq")
            return [c.content for c in chunks]
        except Exception as exc:  # store not ingested, embeddings model unavailable, ...
            self._rag_unavailable = True
            logger.info(f"Semantic context unavailable, using exact knowledge only: {exc}")
            return []

    def build_context(
        self, violation_type: str, severity: str, is_repeat_offender: bool, include_semantic: bool = True
    ) -> tuple[List[str], Optional[FineInfo]]:
        """Collect the legal context for a notice.

        Args:
            violation_type: Key into the fine schedule.
            severity: WARNING / VIOLATION / SEVERE.
            is_repeat_offender: Chooses the first-offence or subsequent-offence fine range.
            include_semantic: Also ask the Chroma store for FAQ context (slow the first time:
                it loads the embedding model). The template fallback turns this off.

        Returns:
            ``(context_lines, fine)``. ``context_lines`` always starts with the exact applicable
            fine and the section text(s); ``fine`` is None if the schedule has no such violation.
        """
        knowledge = self._get_knowledge()
        fine = knowledge.get_fine(violation_type)
        lines: List[str] = []
        if fine:
            lines.append(
                f"Applicable Fine ({fine.section}): {fine.applicable(is_repeat_offender)} "
                f"[{'subsequent' if is_repeat_offender else 'first'} offense]. "
                f"Full schedule: {fine.summary()}. {fine.notes} {fine.disclaimer}".strip()
            )
        for section in knowledge.get_sections(violation_type, severity):
            lines.append(f"{section['section']} - {section['title']}: {section['summary']}")
        if include_semantic:
            lines.extend(self._extra_context(violation_type))
        return lines, fine

    # ------------------------------------------------------------------ LLM
    def generate(
        self,
        record: ViolationRecord,
        vehicle_id: str,
        weather: str = "Clear",
        road_condition: str = "Dry",
        is_repeat_offender: bool = False,
        evidence_frame_ref: Optional[str] = None,
        violation_type: str = "SPEEDING",
    ) -> GeneratedNotice:
        """Generate a violation notice for one classified record.

        Args:
            record: A classified :class:`ViolationRecord`.
            vehicle_id: Short internal vehicle identifier (e.g. "TRK_007").
            weather: Weather at time of violation.
            road_condition: Road surface condition at time of violation.
            is_repeat_offender: Whether the plate has prior logged violations.
            evidence_frame_ref: Reference string/filename for the evidence frame.
            violation_type: Key into the fine schedule (the pipeline detects "SPEEDING").

        Returns:
            A :class:`GeneratedNotice`. If the LLM output fails validation the deterministic
            template is returned instead (``model_used == "fallback_template"``).

        Raises:
            LLMUnavailableError: If no key is configured (also a ValueError) or every Gemini model
                failed; callers fall back to the template.
        """
        context_lines, fine = self.build_context(violation_type, record.severity, is_repeat_offender)
        prompt = _PROMPT_TEMPLATE.format(
            vehicle_id=vehicle_id,
            plate_number=record.plate_number or "UNREADABLE — pending manual verification",
            vehicle_type=record.vehicle_type,
            detected_speed=f"{record.detected_speed_kmph:g}",
            speed_limit=record.speed_limit_kmph,
            excess_speed=f"{record.excess_speed_kmph:g}",
            excess_percent=record.excess_percent,
            severity=record.severity,
            location=record.location,
            timestamp=record.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            evidence_frame=evidence_frame_ref or "N/A",
            weather=weather,
            road_condition=road_condition,
            is_repeat_offender="Yes" if is_repeat_offender else "No",
            context_block="\n".join(context_lines) if context_lines else "No specific context retrieved.",
        )

        notice_text, model_used = self._llm.generate(prompt, system_instruction=_SYSTEM_INSTRUCTION)

        problem = self.validate_notice(notice_text, record, context_lines)
        if problem:
            logger.warning(f"LLM notice rejected for violation_id={record.violation_id}: {problem}")
            fallback = self.generate_fallback_notice(record, vehicle_id, is_repeat_offender, violation_type)
            fallback.cited_context = context_lines
            return fallback

        logger.info(f"Generated notice for violation_id={record.violation_id} chars={len(notice_text)}")
        return GeneratedNotice(
            violation_id=record.violation_id,
            notice_text=notice_text,
            cited_context=context_lines,
            model_used=model_used,
        )

    @staticmethod
    def validate_notice(text: str, record: ViolationRecord, context_lines: List[str]) -> Optional[str]:
        """Check an LLM notice against the facts and context it was given.

        Returns:
            None if the notice is acceptable, otherwise a short reason it was rejected.
        """
        if len(text) < 80:
            return "notice is empty or too short"
        context = "\n".join(context_lines)

        allowed_sections = StructuredKnowledge.section_numbers(context)
        cited_sections = StructuredKnowledge.section_numbers(text)
        if cited_sections - allowed_sections:
            return f"cites sections not in the context: {sorted(cited_sections - allowed_sections)}"

        allowed_amounts = {a.replace(",", "") for a in _RUPEE_AMOUNT_RE.findall(context)}
        cited_amounts = {a.replace(",", "") for a in _RUPEE_AMOUNT_RE.findall(text)}
        if cited_amounts - allowed_amounts:
            return f"states fine amounts not in the context: {sorted(cited_amounts - allowed_amounts)}"

        if record.plate_number and record.plate_number.replace(" ", "") not in text.replace(" ", ""):
            return "plate number missing from the notice"
        speed = f"{record.detected_speed_kmph:g}"
        if speed not in text and f"{record.detected_speed_kmph:.0f}" not in text:
            return "detected speed missing from the notice"
        return None

    def generate_fallback_notice(
        self,
        record: ViolationRecord,
        vehicle_id: str,
        is_repeat_offender: bool = False,
        violation_type: str = "SPEEDING",
    ) -> GeneratedNotice:
        """Deterministic, template-based notice used if the LLM call fails or is unavailable.

        Ensures the pipeline never blocks on an LLM outage; the section and fine come from the
        same exact lookup as the LLM path, so the legal content is identical.

        Args:
            record: A classified :class:`ViolationRecord`.
            vehicle_id: Short internal vehicle identifier.
            is_repeat_offender: Whether the plate has prior logged violations.
            violation_type: Key into the fine schedule.

        Returns:
            A :class:`GeneratedNotice` produced without calling the LLM.
        """
        try:
            context_lines, fine = self.build_context(
                violation_type, record.severity, is_repeat_offender, include_semantic=False
            )
        except Exception as exc:  # knowledge files missing: still produce a notice
            logger.warning(f"Knowledge base unavailable for fallback notice: {exc}")
            context_lines, fine = [], None

        plate = record.plate_number or "UNREADABLE — pending manual verification"
        when = record.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        section = fine.section if fine else "Section 183"
        paragraphs = [
            "TRAFFIC VIOLATION NOTICE",
            f"This is to inform that vehicle {plate} ({record.vehicle_type}) was detected traveling at "
            f"{record.detected_speed_kmph:g} km/h in a {record.speed_limit_kmph} km/h zone on {record.location} "
            f"at {when}.",
            f"This constitutes a violation of {section} of the Motor Vehicles Act, 1988 (Driving at excessive speed).",
        ]
        if fine:
            paragraphs.append(
                f"Applicable Fine: {fine.summary()}. Amounts are indicative; the state-notified amount applies."
            )
        paragraphs += [
            f"Vehicle ID: {vehicle_id}. The vehicle exceeded the speed limit by {record.excess_percent}% "
            f"({record.excess_speed_kmph:g} km/h over limit). This is classified as {record.severity}. "
            f"Repeat offender: {'Yes' if is_repeat_offender else 'No'}.",
            "Action Required: Report to nearest RTO within 30 days or a challan will be issued.",
            "[This notice was auto-generated from a fixed template - LLM drafting was unavailable or its output was rejected.]",
        ]
        return GeneratedNotice(
            violation_id=record.violation_id,
            notice_text="\n\n".join(paragraphs),
            cited_context=context_lines,
            model_used=FALLBACK_MODEL_NAME,
        )
