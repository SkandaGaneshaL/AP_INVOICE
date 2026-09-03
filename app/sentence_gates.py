"""Small, dependency-free gates for the active LLM-first sentence path."""

from __future__ import annotations

from .models import EvaluationResult, RuleGenerationContext
from .sentence_validators import validate_sentence


def evaluate_sentence_gates(sentence: str, context: RuleGenerationContext) -> EvaluationResult:
    """Validate a generated sentence without loading extraction/GEPA modules."""
    try:
        validate_sentence(sentence, {"old_value": context.old_value, "new_value": context.new_value})
    except ValueError as exc:
        return EvaluationResult(score=0.0, confidence="failed", feedback=str(exc),
                                candidate_status="rejected", promotion_eligible=False,
                                schema_valid=False)
    packet = context.feedback_packet
    supported = packet is None or bool(packet.evidence)
    return EvaluationResult(
        score=1.0 if supported else 0.0,
        confidence="normal" if supported else "failed",
        feedback=("Sentence passed compact safety and evidence gates." if supported
                  else "No supporting evidence was available."),
        field_match=True,
        evidence_supported=supported,
        schema_valid=True,
        candidate_status="evaluated" if supported else "rejected",
        promotion_eligible=supported,
    )
