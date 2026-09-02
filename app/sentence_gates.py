from __future__ import annotations

import re
from typing import Any


def validate_sentence_for_program(sentence: str, program: Any, *, old_value: Any = None, new_value: Any = None,
                                  existing_text: str = "") -> tuple[bool, list[str]]:
    reasons: list[str] = []
    text = (sentence or "").strip()
    if not text or "{" in text or "}" in text or "```" in text:
        reasons.append("empty_or_unstructured_sentence")
    if old_value is not None and str(old_value).strip() and str(old_value).casefold() in text.casefold():
        reasons.append("old_value_memorized")
    if new_value is not None and str(new_value).strip() and str(new_value).casefold() in text.casefold():
        reasons.append("new_value_memorized")
    ops = {item.op for item in getattr(program, "transform", [])}
    lowered = text.casefold()
    if ops.intersection({"numeric_thousands_canonicalize", "money_canonicalize", "parse_money"}) and not any(
        token in lowered for token in ("number", "separator", "comma", "currency", "grouping")
    ):
        reasons.append("money_transform_not_expressed")
    if ops.intersection({"identifier_strip_leading_alpha", "strip_leading_alpha_token"}) and not any(
        token in lowered for token in ("leading", "alphabetic", "prefix")
    ):
        reasons.append("identifier_transform_not_expressed")
    if "date_format_repattern" in ops or "parse_date" in ops:
        target = getattr(getattr(program, "format", None), "target_pattern", None)
        if target and target.casefold() not in lowered:
            reasons.append("date_target_pattern_missing")
    return not reasons, reasons
