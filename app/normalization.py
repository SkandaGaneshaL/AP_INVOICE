from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizationPolicy:
    field_key: str
    mode: str = "none"


def infer_policy(field_key: str, old_value: Any, new_value: Any) -> NormalizationPolicy:
    """Infer a generic policy from the typed edit, never from a field name."""
    old, new = str(old_value or ""), str(new_value or "")
    if re.fullmatch(r"[A-Za-z]+(?:[- ]+)[A-Za-z0-9-]+", old) and re.fullmatch(r"[A-Za-z0-9-]+", new) and old.split()[-1] == new:
        return NormalizationPolicy(field_key, "strip_leading_alpha_token")
    if re.fullmatch(r"[\d\s,.'$€£₹-]+", old) and re.fullmatch(r"[\d\s,.'$€£₹-]+", new):
        return NormalizationPolicy(field_key, "parse_money")
    return NormalizationPolicy(field_key)


def apply_policy(field_key: str, value: Any, mode: str) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if mode in {"remove_prefix", "strip_leading_alpha_token"} and re.search(r"\d+$", value):
        return re.sub(r"^[A-Za-z][A-Za-z0-9 _-]*?(?=\d+$)", "", value).strip()
    if mode == "parse_money":
        return re.sub(r"[, ]", "", value)
    return value
