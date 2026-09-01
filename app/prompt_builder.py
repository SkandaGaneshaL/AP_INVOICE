from __future__ import annotations

import json
from .models import RuleGenerationContext


class RulePromptBuilder:
    """Canonical prompt/context builder shared by every rule strategy."""

    @staticmethod
    def feedback(context: RuleGenerationContext) -> str:
        packet = context.feedback_packet
        if packet:
            evidence = "\n".join(
                f"- Page {item.page}: {item.snippet} (label={item.label or 'unknown'}, raw_value={item.raw_value}, "
                f"normalized_value={item.normalized_value}, transformation={item.transformation or 'none'}, "
                f"confidence={item.confidence:.2f})"
                for item in packet.evidence
            ) or "- No deterministic source evidence was found."
            competing = "\n".join(
                f"- Page {item.page}: {item.snippet} (label={item.label or 'unknown'})"
                for item in packet.competing_evidence
            ) or "- No competing occurrence was found."
            examples = "\n".join(
                f"- {example.input_evidence} -> {example.expected_output}; lesson: {example.rule_lesson}"
                for example in packet.historical_examples
            ) or "- No historical demonstrations are available."
            return (
                f"Observed correction: {packet.observed_correction}\n"
                f"Evidence found in the current invoice:\n{evidence}\n"
                f"Competing evidence:\n{competing}\n"
                f"Generalized extraction intent: {packet.inferred_intent}\n"
                f"Constraints:\n" + "\n".join(f"- {item}" for item in packet.constraints) +
                f"\nHistorical demonstrations:\n{examples}"
            )
        return (
            f"Previous extracted value: {context.old_value}\n"
            f"Corrected value: {context.new_value}\n"
            f"Required behavior: extract the explicitly identified {context.display_label or context.field_key}.\n"
            f"Generalization constraint: do not hard-code {context.new_value} as the universal value."
        )

    @classmethod
    def normal_payload(cls, context: RuleGenerationContext) -> dict:
        return {
            "field_key": context.field_key,
            "display_label": context.display_label,
            "short_rule": context.short_rule,
            "existing_rules": context.detailed_rule,
            "field_path": context.field_path,
            "previous_value": context.old_value,
            "corrected_value": context.new_value,
            "feedback": cls.feedback(context),
            "invoice_payload": context.invoice_payload,
            "corrected_response": context.final_response,
            "feedback_packet": context.feedback_packet.model_dump() if context.feedback_packet else None,
            "normalization_mode": context.normalization_mode,
            "instruction_precedence": "The current user correction controls this candidate; do not apply conflicting older rules.",
        }

    @classmethod
    def seed(cls, context: RuleGenerationContext) -> str:
        parts = [
            context.short_rule,
            *context.detailed_rule,
            cls.feedback(context),
            "Write reusable extraction behavior; preserve existing rules and do not memorize this invoice.",
        ]
        return "\n".join(parts).strip()

    @classmethod
    def gepa_seed(cls, context: RuleGenerationContext) -> str:
        """Return one compact instruction for GEPA's text component."""
        if context.field_key.casefold() == "invoicenumber":
            if context.normalization_mode == "preserve_prefix":
                return "Extract the invoice number next to the explicit invoice-number label and retain all meaningful leading alphanumeric characters exactly as shown."
            if context.normalization_mode == "remove_prefix":
                return "Extract the invoice number next to the explicit invoice-number label and remove leading alphabetic characters only when the remaining portion is a valid numeric invoice number."
        packet = context.feedback_packet
        candidates = [
            context.detailed_rule[0] if context.detailed_rule else "",
            context.short_rule,
            packet.inferred_intent if packet else "",
        ]
        for candidate in candidates:
            text = " ".join(str(candidate).strip().split())
            if text and text[-1] not in ".!?":
                text += "."
            if text and text.count(".") <= 1 and "\n" not in text:
                return text
        return f"Extract the value for {context.display_label or context.field_key} from its explicit invoice label."

    @staticmethod
    def json_text(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
