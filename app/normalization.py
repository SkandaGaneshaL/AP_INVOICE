from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from .transform_induction import induce_transforms
from .type_inference import infer_type, InferredType


@dataclass(frozen=True)
class NormalizationPolicy:
    field_key: str
    mode: str = "none"


def infer_policy(field_key: str, old_value: Any, new_value: Any) -> NormalizationPolicy:
    """Infer a generic policy from the typed edit, never from a field name."""
    inferred = infer_type(old_value, new_value, field_key)
    candidates = induce_transforms(old_value, new_value, inferred)
    if candidates:
        op = candidates[0].program.transform[0].op
        if op == "strip_leading_alpha_token":
            return NormalizationPolicy(field_key, "strip_leading_alpha_token")
        if op == "parse_date":
            return NormalizationPolicy(field_key, "parse_date")
        if op == "parse_money":
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
