from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizationPolicy:
    field_key: str
    mode: str = "none"


def infer_policy(field_key: str, old_value: Any, new_value: Any) -> NormalizationPolicy:
    """Infer only the transformation explicitly represented by the correction."""
    if field_key.casefold() != "invoicenumber":
        return NormalizationPolicy(field_key)
    old = str(old_value or "").strip()
    new = str(new_value or "").strip()
    if re.fullmatch(r"[A-Za-z]+\d+", old) and re.fullmatch(r"\d+", new):
        if old[len(old) - len(new):] == new:
            return NormalizationPolicy(field_key, "remove_prefix")
    if re.fullmatch(r"\d+", old) and re.fullmatch(r"[A-Za-z]+\d+", new):
        if new[len(new) - len(old):] == old:
            return NormalizationPolicy(field_key, "preserve_prefix")
    return NormalizationPolicy(field_key)


def apply_policy(field_key: str, value: Any, mode: str) -> Any:
    if field_key.casefold() != "invoicenumber" or not isinstance(value, str):
        return value
    value = value.strip()
    if mode == "remove_prefix" and re.fullmatch(r"[A-Za-z]+\d+", value):
        return re.sub(r"^[A-Za-z]+(?=\d+$)", "", value)
    return value
