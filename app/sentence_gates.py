"""Small, dependency-free gates for the active LLM-first sentence path."""

from __future__ import annotations

from .models import EvaluationResult, RuleGenerationContext
from .sentence_validators import validate_sentence


def evaluate_sentence_gates(sentence: str, context: RuleGenerationContext) -> EvaluationResult:
    """Validate a generated sentence without loading retired modules."""
    try:
        validate_sentence(sentence, {"old_value": context.old_value, "new_value": context.new_value})
    except ValueError as exc:
        return EvaluationResult(score=0.0, confidence="failed", feedback=str(exc),
                                candidate_status="rejected", promotion_eligible=False,
                                schema_valid=False)
    packet = context.feedback_packet
    # Evidence is an independent diagnostic. Missing text-layer evidence must
    # not turn a behaviorally valid correction into a rejected candidate.
    evidence_supported = None if packet is None else bool(packet.evidence)
    evidence_status = "supported" if evidence_supported else "unavailable"
    if packet is not None and not packet.evidence and packet.competing_evidence:
        evidence_status = "ambiguous"
    return EvaluationResult(
        score=1.0,
        confidence="normal",
        feedback=f"Sentence passed compact safety gates; evidence status is {evidence_status}.",
        field_match=True,
        evidence_supported=evidence_supported,
        schema_valid=True,
        candidate_status="evaluated",
        promotion_eligible=True,
    )
