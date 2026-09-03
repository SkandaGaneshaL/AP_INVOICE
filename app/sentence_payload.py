from __future__ import annotations

from typing import Any
from .correction_kind import CorrectionKind, classify_correction


def _gist(rules: list[str]) -> str:
    return " ".join(" ".join(str(rule).split()) for rule in rules[:2])[:500]


def build_sentence_payload(context: Any) -> dict[str, Any]:
    packet = getattr(context, "feedback_packet", None)
    evidence = list(getattr(packet, "evidence", []) or [])
    competing = list(getattr(packet, "competing_evidence", []) or [])
    positive = list(dict.fromkeys([str(item.label) for item in evidence if getattr(item, "label", None)]))[:4]
    negative = list(dict.fromkeys([str(item.label) for item in competing if getattr(item, "label", None)]))[:4]
    kind = classify_correction(context.old_value, context.new_value, labels=positive, competing_labels=negative)
    return {
        "field_key": context.field_key,
        "display_label": context.display_label,
        "short_rule": context.short_rule,
        "existing_detailed_rule_gist": _gist(context.detailed_rule),
        "old_value": context.old_value,
        "new_value": context.new_value,
        "correction_kind": kind.value,
        "positive_labels": positive,
        "negative_labels": negative,
        "null_policy": context.current_program.null_policy if context.current_program else "labeled_empty_to_null",
        "supplier_scope": "this_supplier_only",
    }
