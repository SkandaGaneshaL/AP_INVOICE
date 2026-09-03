from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field


class RuleMergeResult(BaseModel):
    updated_detailed_rule: list[str] = Field(default_factory=list, max_length=6)
    dropped_bullets: list[str] = Field(default_factory=list, max_length=6)
    conflict_resolved: bool = False


def merge_rule_sentences(existing: list[str], new_sentence: str, *, max_rules: int = 6) -> RuleMergeResult:
    """Merge compatible prose without creating duplicate or contradictory bullets.

    The active application does not call a second model for an ordinary merge.
    This bounded merger is the safe fallback and intentionally treats explicit
    negation as a conflict with a positive clause about the same concept.
    """
    values: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    incoming = " ".join(str(new_sentence).split()).strip()
    incoming_key = incoming.casefold()
    incoming_terms = _concept_terms(incoming)
    for item in [*existing, incoming]:
        text = " ".join(str(item).split()).strip()
        key = text.casefold()
        if not text or key in seen:
            if text and key == incoming_key:
                dropped.append(text)
            continue
        if text != incoming and _conflicts(text, incoming_terms):
            dropped.append(text)
            continue
        values.append(text)
        seen.add(key)
    overflow = values[:-max_rules] if len(values) > max_rules else []
    if overflow:
        dropped.extend(overflow)
    return RuleMergeResult(updated_detailed_rule=values[-max_rules:], dropped_bullets=dropped,
                           conflict_resolved=bool(dropped))


def _concept_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z][a-z-]+", text.casefold())
            if term not in {"the", "and", "from", "only", "with", "when", "this", "field"}}


def _conflicts(existing: str, incoming_terms: set[str]) -> bool:
    existing_terms = _concept_terms(existing)
    shared = existing_terms & incoming_terms
    if not shared:
        return False
    action_terms = {"extract", "retain", "preserve", "normalize", "convert", "use", "select", "ignore", "exclude", "infer"}
    if not (existing_terms & incoming_terms & action_terms):
        return False
    existing_negative = bool(re.search(r"\b(do not|don't|never|exclude|ignore|not)\b", existing, re.I))
    incoming_negative = bool(re.search(r"\b(do not|don't|never|exclude|ignore|not)\b", " ".join(incoming_terms), re.I))
    return existing_negative != incoming_negative


def build_merge_payload(context: Any, new_sentence: str, correction_kind: str) -> dict[str, Any]:
    return {
        "field_key": context.field_key,
        "display_label": context.display_label,
        "short_rule": context.short_rule,
        "detailed_rule": [" ".join(str(x).split()) for x in context.detailed_rule[:6]],
        "new_sentence": new_sentence,
        "correction_kind": correction_kind,
        "merge_policy": ["Keep behavior generic.", "Rewrite contradictions.", "Do not append duplicates.",
                          "Return at most six one-sentence bullets."],
    }
