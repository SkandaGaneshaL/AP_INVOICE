from __future__ import annotations

from enum import Enum
from typing import Any
import re


class CorrectionKind(str, Enum):
    LABEL_DISAMBIGUATION = "label_disambiguation"
    SECTION_PRIOR = "section_prior"
    NULL_POLICY = "null_policy"
    PRESERVE_LITERAL = "preserve_literal"
    FORMAT_POLICY = "format_policy"
    LIST_ALIGNMENT = "list_alignment"


def classify_correction(old_value: Any, new_value: Any, *, labels: list[str] | None = None,
                        competing_labels: list[str] | None = None) -> CorrectionKind:
    if new_value is None or str(new_value).strip().casefold() in {"", "null", "none", "n/a", "not present"}:
        return CorrectionKind.NULL_POLICY
    if labels and competing_labels:
        return CorrectionKind.LABEL_DISAMBIGUATION
    if isinstance(old_value, list) or isinstance(new_value, list):
        return CorrectionKind.LIST_ALIGNMENT
    old, new = str(old_value or ""), str(new_value or "")
    if old.casefold() == new.casefold() and old != new:
        return CorrectionKind.PRESERVE_LITERAL
    if re.fullmatch(r"[\d\s,.'$€£₹-]+", old) and re.fullmatch(r"[\d\s,.'$€£₹-]+", new):
        return CorrectionKind.FORMAT_POLICY
    if re.fullmatch(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", old) and re.fullmatch(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", new):
        return CorrectionKind.FORMAT_POLICY
    return CorrectionKind.PRESERVE_LITERAL
