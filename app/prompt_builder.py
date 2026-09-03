from __future__ import annotations

import json
from .models import RuleGenerationContext


class RulePromptBuilder:
    """Canonical prompt/context builder shared by every rule strategy."""

    @staticmethod
    def feedback(context: RuleGenerationContext) -> str:
        packet = context.feedback_packet
        if packet:
            return (
                f"Observed correction: {packet.observed_correction}\n"
                f"Generalized extraction intent: {packet.inferred_intent}\n"
                f"Positive label categories: {sorted({item.label for item in packet.evidence if item.label})}\n"
                f"Negative label categories: {sorted({item.label for item in packet.competing_evidence if item.label})}\n"
                f"Constraints: {packet.constraints[:4]}"
            )
        return (
            f"Previous extracted value: {context.old_value}\n"
            f"Corrected value: {context.new_value}\n"
            f"Required behavior: extract the explicitly identified {context.display_label or context.field_key}.\n"
            f"Generalization constraint: do not hard-code {context.new_value} as the universal value."
        )

    @classmethod
    def normal_payload(cls, context: RuleGenerationContext) -> dict:
        from .sentence_payload import build_sentence_payload
        return build_sentence_payload(context)

    @classmethod
    def seed(cls, context: RuleGenerationContext) -> str:
        return cls.json_text(cls.normal_payload(context))

    @staticmethod
    def json_text(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
