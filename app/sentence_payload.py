from __future__ import annotations

from typing import Any
import re
from .correction_kind import CorrectionKind, classify_correction


def build_policy_gist(rule_or_rules: Any) -> str:
    """Compress rule policy while retaining selection/transform/null signals."""
    if hasattr(rule_or_rules, "SHORT_RULE"):
        short = getattr(rule_or_rules, "SHORT_RULE", "")
        rules = getattr(rule_or_rules, "DETAILED_RULE", []) or []
    else:
        short = ""
        rules = rule_or_rules or []
    fragments = [" ".join(str(item).split()) for item in [short, *rules] if str(item).strip()]
    # Keep behavior-bearing clauses instead of blindly taking the first two
    # bullets, and cap the dynamic portion for predictable input cost.
    ranked = [item for item in fragments if re.search(
        r"label|select|extract|use|prefer|normalize|format|convert|remove|trim|empty|null|preserve|ignore|exclude",
        item, re.I
    )]
    selected = list(dict.fromkeys(ranked or fragments))[:4]
    return " ".join(selected)[:300]


def build_sentence_payload(context: Any) -> dict[str, Any]:
    packet = getattr(context, "feedback_packet", None)
    evidence = list(getattr(packet, "evidence", []) or [])
    positive = list(dict.fromkeys([str(item.label) for item in evidence if getattr(item, "label", None)]))[:4]
    kind = classify_correction(context.old_value, context.new_value)
    old = context.old_value
    new = context.new_value
    observed = observe_change(old, new)
    return {
        "sentence_payload_version": "correction-delta-v1",
        "task": "write_one_extraction_rule_sentence",
        "field_key": context.field_key,
        "display_label": context.display_label,
        "supplier_scope": "this_supplier_only",
        "existing_behavior_one_line": build_policy_gist(context),
        "delta": {"old_value": old, "new_value": new, "observed_change": observed,
                  "do_not_restate": "Do not repeat either invoice value."},
        "correction_kind": kind.value,
        "positive_labels": positive,
        "null_policy": (
            context.current_program.get("null_policy", "labeled_empty_to_null")
            if isinstance(context.current_program, dict)
            else "labeled_empty_to_null"
        ),
        "constraints": ["One reusable imperative sentence.", "Do not repeat either invoice value.",
                        "Do not add fallback hops.", "If already covered, return noop=true."],
    }


def observe_change(old: Any, new: Any) -> str:
    if new is None or str(new).strip().casefold() in {"", "null", "none", "n/a", "not present"}:
        return "value replaced with null"
    if old == new:
        return "no meaningful correction"
    if old is None:
        return "value introduced"
    a, b = str(old), str(new)
    if a.casefold() == b.casefold() and a != b:
        return "literal formatting preserved with case change"
    if "".join(a.split()) == "".join(b.split()):
        return "whitespace normalized"
    if re.fullmatch(r"[A-Za-z]+[- ]?\d+", a) and re.fullmatch(r"\d+", b):
        return "leading identifier prefix removed"
    if re.sub(r"(?:,?\s+(?:inc|llc|ltd|limited|corp|corporation|co)\.?$)", "", a, flags=re.I).casefold() == b.casefold():
        return "legal entity suffix removed"
    if re.fullmatch(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", a) and re.fullmatch(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", b):
        return "date order or date formatting changed"
    if re.sub(r"[^A-Za-z0-9]", "", a).casefold() == re.sub(r"[^A-Za-z0-9]", "", b).casefold():
        return "punctuation canonicalized"
    return "value replaced"
