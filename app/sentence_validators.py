from __future__ import annotations

import re
from typing import Any


# ``when`` is valid rule language (for example, "extract the value when it is
# adjacent to its label").  Only reject explicit alternate-source/fallback
# instructions here.
_HOPS = re.compile(r"\b(otherwise|fallback|fall back|then use|instead|if not found|if missing|if absent|if empty|search for|nearest heading|adjacent to the date|regex|or another field|alternate source|alternative source)\b", re.I)


def validate_sentence(sentence: Any, payload: dict[str, Any]) -> str:
    if not isinstance(sentence, str):
        raise ValueError("sentence must be text")
    value = " ".join(sentence.strip().split()).strip('`"')
    if not value or len(value.split()) > 40 or any(token in value for token in ("{", "}", "```")):
        raise ValueError("sentence failed length or format gate")
    if len(re.findall(r"[.!?](?:\s|$)", value)) > 1 or _HOPS.search(value):
        raise ValueError("sentence contains multiple sentences or fallback hops")
    for key in (payload.get("old_value"), payload.get("new_value")):
        if key is not None and len(str(key)) >= 3 and str(key).casefold() in value.casefold():
            raise ValueError("sentence hard-codes the correction value")
    configured = payload.get("configured_field_keys", [])
    for key in [*payload.get("unrelated_field_keys", []), *configured]:
        if key and re.search(rf"\b{re.escape(str(key))}\b", value, re.I):
            raise ValueError("sentence references an unrelated field")
    existing = str(payload.get("existing_behavior_one_line", "")).strip()
    if existing and value.casefold() == existing.casefold():
        raise ValueError("sentence merely restates the existing rule")
    _validate_correction_behavior(value, payload)
    return value


def _date_pattern(value: Any) -> str | None:
    match = re.fullmatch(r"(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})", str(value or "").strip())
    if not match:
        return None
    parts = [match.group(index) for index in range(1, 4)]
    if len(parts[0]) == 4:
        return "YYYY/MM/DD"
    if len(parts[2]) == 4:
        return "DD/MM/YYYY" if int(parts[1]) > 12 else "MM/DD/YYYY"
    return None


def _validate_correction_behavior(sentence: str, payload: dict[str, Any]) -> None:
    """Require the generated sentence to express the observed value delta."""
    kind = str(payload.get("correction_kind") or "").casefold()
    if kind in {"", "no_op", "noop"}:
        return
    text = sentence.casefold()
    if "identifier_prefix" in kind or "prefix" in kind:
        if not re.search(r"\b(prefix|identifier|numeric core|non-core|leading)\b", text):
            raise ValueError("sentence did not encode identifier-prefix correction")
    elif "legal_suffix" in kind or "suffix" in kind:
        if not re.search(r"\b(suffix|legal entity|company suffix|corporate)\b", text):
            raise ValueError("sentence did not encode legal-suffix correction")
    elif "date" in kind:
        if not re.search(r"\b(date|format|reformat|order)\b", text):
            raise ValueError("sentence did not encode date correction")
        old_pattern = _date_pattern((payload.get("delta") or {}).get("old_value", payload.get("old_value")))
        new_pattern = _date_pattern((payload.get("delta") or {}).get("new_value", payload.get("new_value")))
        if old_pattern and new_pattern and old_pattern != new_pattern and not re.search(
            r"\b(format|reformat|order|repattern|convert|canonical)\b", text
        ):
            raise ValueError("sentence did not encode date correction")
        if old_pattern and new_pattern and old_pattern != new_pattern and old_pattern.casefold() in text:
            raise ValueError("sentence preserves the obsolete date format")
    elif "whitespace" in kind:
        if not re.search(r"\b(whitespace|space|trim|collapse|canonical)\b", text):
            raise ValueError("sentence did not encode whitespace correction")
    elif "punctuation" in kind or "canonical" in kind:
        if not re.search(r"\b(punctuation|canonical|separator|normalize|format)\b", text):
            raise ValueError("sentence did not encode punctuation correction")
    elif "null" in kind:
        if not re.search(r"\b(null|empty|absent|missing)\b", text):
            raise ValueError("sentence did not encode null correction")


def assemble_local_sentence(payload: dict[str, Any]) -> str:
    """Build a generic sentence from validated correction metadata, without a second model call."""
    label = str(payload.get("display_label") or payload.get("field_key") or "field")
    kind = str(payload.get("correction_kind") or payload.get("delta", {}).get("observed_change") or "value behavior")
    if "prefix" in kind:
        return f"Extract the value from the explicitly labeled {label} field and remove non-core identifier prefixes."
    if "suffix" in kind:
        return f"Extract the value from the explicitly labeled {label} field and remove legal entity suffixes."
    if "date" in kind:
        return f"Extract the value from the explicitly labeled {label} field and apply the configured date format."
    if "null" in kind:
        return f"Extract the value from the explicitly labeled {label} field and return null when the labeled value is absent."
    if "whitespace" in kind or "format" in kind or "canonical" in kind:
        return f"Extract the value from the explicitly labeled {label} field and apply the configured canonical format."
    return f"Extract the value from the explicitly labeled {label} field according to the active correction policy."
