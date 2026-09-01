import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from .models import ProviderGenerationResult, RuleGenerationContext, UsageSummary


@dataclass
class GeneratedRuleCandidate:
    sentence: str | None
    strategy: str
    request_id: str | None = None
    response_format: str | None = None
    evaluation: Any = None
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_candidate_id: str | None = None
    prompt_hash: str | None = None
    status: str = "generated"
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: UsageSummary = field(default_factory=UsageSummary)


def prompt_hash(context: RuleGenerationContext) -> str:
    payload = {
        "field_key": context.field_key,
        "field_path": context.field_path,
        "short_rule": context.short_rule,
        "detailed_rule": context.detailed_rule,
        "old_value": context.old_value,
        "new_value": context.new_value,
        "feedback_packet": context.feedback_packet.model_dump(exclude={"original_field_node", "corrected_field_node"})
        if context.feedback_packet else None,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


class GenerativeRuleGenerator:
    name = "generative"

    def __init__(self, generator):
        self.generator = generator

    def generate_candidate(self, context: RuleGenerationContext) -> GeneratedRuleCandidate:
        try:
            if hasattr(self.generator, "generate_with_metadata"):
                result = self.generator.generate_with_metadata(context=context)
                if isinstance(result, ProviderGenerationResult):
                    return GeneratedRuleCandidate(
                        sentence=result.sentence,
                        strategy=self.name,
                        request_id=result.request_id,
                        response_format=result.response_format,
                        prompt_hash=prompt_hash(context),
                        usage=result.usage,
                        metadata={"model": result.model, "attempts": result.attempts,
                                  "finish_reason": result.finish_reason,
                                  "usage_location": result.usage_location,
                                  "reasoning_effort_requested": result.reasoning_effort_requested,
                                  "reasoning_effort_effective": result.reasoning_effort_effective,
                                  "reasoning_supported": result.reasoning_supported,
                                  "visible_reasoning": result.visible_reasoning,
                                  "decision_summary": result.decision_summary,
                                  "reason": result.reason or result.decision_summary,
                                  "reasoning_summary_available": result.reasoning_summary_available,
                                  "reasoning_mode": result.reasoning_mode,
                                  "reasoning_parameter_sent": result.reasoning_parameter_sent,
                                  "verbosity_parameter_sent": result.verbosity_parameter_sent,
                                  "usage_diagnostics": result.usage_diagnostics},
                    )
            result = self.generator.generate(context=context)
        except TypeError as exc:
            if "context" not in str(exc) and "unexpected keyword" not in str(exc):
                raise
            result = self.generator.generate(
                field_key=context.field_key,
                display_label=context.display_label,
                short_rule=context.short_rule,
                detailed_rule=context.detailed_rule,
                old_value=context.old_value,
                new_value=context.new_value,
                field_path=context.field_path,
                invoice_payload=context.invoice_payload,
                final_response=context.final_response,
                historical_examples=context.historical_examples,
            )
        sentence, request_id, response_format = result if len(result) == 3 else (*result, None)
        return GeneratedRuleCandidate(
            sentence=sentence,
            strategy=self.name,
            request_id=request_id,
            response_format=response_format,
            prompt_hash=prompt_hash(context),
            usage=UsageSummary(calls=1, unknown_calls=1),
        )
