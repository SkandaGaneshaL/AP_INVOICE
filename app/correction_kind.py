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
    IDENTIFIER_PREFIX = "identifier_prefix_removed"
    DATE_FORMAT = "date_format_changed"
    LEGAL_SUFFIX = "legal_suffix_removed"
    WHITESPACE = "whitespace_normalized"
    PUNCTUATION = "punctuation_canonicalized"
    VALUE_REPLACEMENT = "value_replacement"
    NOOP = "no_op"


def classify_correction(old_value: Any, new_value: Any, *, labels: list[str] | None = None,
                        competing_labels: list[str] | None = None) -> CorrectionKind:
    # Classification is intentionally based on the value delta. Labels and
    # competing evidence are contextual metadata, never field-specific logic.
    if old_value == new_value:
        return CorrectionKind.NOOP
    if new_value is None or str(new_value).strip().casefold() in {"", "null", "none", "n/a", "not present"}:
        return CorrectionKind.NULL_POLICY
    if isinstance(old_value, list) or isinstance(new_value, list):
        return CorrectionKind.LIST_ALIGNMENT
    old, new = str(old_value or ""), str(new_value or "")
    if re.fullmatch(r"[A-Za-z]+(?:[- ]+)[A-Za-z0-9-]+", old) and re.fullmatch(r"[A-Za-z0-9-]+", new):
        if old.replace(" ", "-").casefold().endswith(new.casefold()) or old.split()[-1].casefold() == new.casefold():
            return CorrectionKind.IDENTIFIER_PREFIX
    if re.sub(r"(?:,?\s+(?:inc|llc|ltd|limited|corp|corporation|co)\.?$)", "", old, flags=re.I).casefold() == new.casefold():
        return CorrectionKind.LEGAL_SUFFIX
    if old.casefold() == new.casefold() and old != new:
        return CorrectionKind.PUNCTUATION
    if "".join(old.split()) == "".join(new.split()):
        return CorrectionKind.WHITESPACE
    if re.fullmatch(r"[\d\s,.'$€£₹-]+", old) and re.fullmatch(r"[\d\s,.'$€£₹-]+", new):
        return CorrectionKind.FORMAT_POLICY
    if re.fullmatch(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", old) and re.fullmatch(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", new):
        return CorrectionKind.DATE_FORMAT
    if old and new and old != new:
        return CorrectionKind.VALUE_REPLACEMENT
    return CorrectionKind.PRESERVE_LITERAL
