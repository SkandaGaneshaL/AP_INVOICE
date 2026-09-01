from __future__ import annotations

from typing import Any, Mapping, Sequence

from .evaluation import EvaluationTrace, ExtractionEvaluator
from .models import DemonstrationExample, EvaluationResult, RuleFeedbackPacket, RuleGenerationContext, UsageSummary


def serialize_context_example(context: RuleGenerationContext) -> dict[str, Any]:
    """Serialize a GEPA example without document bytes or model-only objects."""
    payload = context.model_dump(mode="json", exclude={"document_bytes"})
    payload["_example_type"] = "current"
    payload["document_available"] = bool(context.document_bytes)
    if context.feedback_packet:
        packet = context.feedback_packet
        payload["feedback_packet"] = packet.model_dump(mode="json")
    return payload


def _normalise_packet(value: Any) -> RuleFeedbackPacket | None:
    if value is None:
        return None
    if isinstance(value, RuleFeedbackPacket):
        return value.model_copy(deep=True)
    return RuleFeedbackPacket.model_validate(value)


def _normalise_demonstrations(value: Any) -> list[DemonstrationExample]:
    result: list[DemonstrationExample] = []
    for item in value or []:
        result.append(item if isinstance(item, DemonstrationExample)
                      else DemonstrationExample.model_validate(item))
    return result


def deserialize_context_example(
    base_context: RuleGenerationContext,
    example: Mapping[str, Any],
) -> RuleGenerationContext:
    """Rebuild typed nested models and isolate the base context."""
    payload = base_context.model_dump(mode="python")
    for key, value in example.items():
        if key.startswith("_") or key in {"document_bytes", "document_available"}:
            continue
        if key in payload:
            payload[key] = value

    packet = _normalise_packet(payload.get("feedback_packet"))
    if packet:
        packet.historical_examples = _normalise_demonstrations(packet.historical_examples)
    payload["feedback_packet"] = packet
    payload["historical_examples"] = [dict(item) if isinstance(item, Mapping) else item
                                       for item in payload.get("historical_examples", [])]

    # Historical metadata is not executable against the current PDF. Only the
    # explicitly serialized current example may use the base document bytes.
    is_current = example.get("_example_type") == "current" and example.get("document_available", False)
    rebuilt = RuleGenerationContext.model_validate(payload)
    if is_current:
        rebuilt.document_bytes = base_context.document_bytes
    return rebuilt


class ExtractionRuleGEPAAdapter:
    """Standalone GEPA adapter for evolving one extraction instruction component."""

    def __init__(self, context: RuleGenerationContext, evaluator: ExtractionEvaluator):
        self.context = context
        self.evaluator = evaluator
        self.usage = UsageSummary()
        # GEPA 0.1.x inspects this optional hook directly.  We use the
        # injected reflection_strategy instead, so the hook must exist and
        # explicitly be None.
        self.propose_new_texts = None

    def _context_for(self, example: dict[str, Any]) -> RuleGenerationContext:
        if not isinstance(example, Mapping):
            raise TypeError("GEPA example must be a mapping")
        if example.get("field_key") not in {None, self.context.field_key}:
            return self.context.model_copy(deep=True)
        return deserialize_context_example(self.context, example)

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch
        instruction = candidate["extraction_instruction"]
        outputs, scores, traces = [], [], []
        for example in batch:
            try:
                context = self._context_for(example)
                result, trace = self.evaluator.evaluate(context, instruction)
            except Exception as exc:
                feedback = f"The GEPA example could not be deserialized or evaluated: {type(exc).__name__}."
                result = EvaluationResult(score=0.0, confidence="failed", feedback=feedback)
                trace = EvaluationTrace(
                    field_path=self.context.field_path,
                    instruction=instruction,
                    extracted_value=None,
                    expected_value=self.context.new_value,
                    score=0.0,
                    feedback=feedback,
                    format_valid=False,
                    field_key=self.context.field_key,
                )
            outputs.append({"field_key": trace.field_key, "extracted_value": trace.extracted_value,
                            "actual_page": trace.actual_page, "evidence_supported": trace.evidence_supported,
                            "feedback": trace.feedback,
                            "usage": trace.usage.model_dump() if trace.usage else None})
            if trace.usage:
                self.usage.add_summary(trace.usage)
            scores.append(float(result.score or 0.0))
            traces.append(trace)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traces if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = []
        for trace in eval_batch.trajectories or []:
            records.append({
                "Inputs": {"field_key": trace.field_key, "field_path": trace.field_path,
                           "evidence_snippets": trace.evidence_snippets,
                           "competing_snippets": trace.competing_snippets},
                "Generated Outputs": {"extracted_value": trace.extracted_value, "Page": trace.actual_page},
                "Expected Outputs": {"value": trace.expected_value, "Page": trace.expected_page},
                "Feedback": trace.feedback,
                "score": trace.score,
                "evidence_supported": trace.evidence_supported,
                "contradiction": trace.contradiction,
            })
        return {component: records for component in components_to_update}
