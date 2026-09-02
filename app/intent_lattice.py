from __future__ import annotations

from enum import Enum
from typing import Any
from .type_inference import InferredType


class FailureType(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    NORMALIZATION_ERROR = "NORMALIZATION_ERROR"
    USER_PREFERENCE_FORMATTING = "USER_PREFERENCE_FORMATTING"
    VALUE_BOUNDARY_ERROR = "VALUE_BOUNDARY_ERROR"
    LABEL_SELECTION_ERROR = "LABEL_SELECTION_ERROR"
    WRONG_FIELD_ERROR = "WRONG_FIELD_ERROR"
    OCR_ERROR = "OCR_ERROR"
    MULTI_CANDIDATE_CONFLICT = "MULTI_CANDIDATE_CONFLICT"
    NULL_POLICY_ERROR = "NULL_POLICY_ERROR"
    UNSUPPORTED_CORRECTION = "UNSUPPORTED_CORRECTION"


def classify_intent(old_value: Any, new_value: Any, inferred_type: InferredType, evidence_hits=None, competing_hits=None) -> FailureType:
    if old_value == new_value:
        return FailureType.EXACT_MATCH
    if new_value is None or str(new_value).casefold() in {"", "null", "none", "n/a"}:
        return FailureType.NULL_POLICY_ERROR
    if evidence_hits and competing_hits:
        return FailureType.MULTI_CANDIDATE_CONFLICT
    if inferred_type in {InferredType.IDENTIFIER, InferredType.DATE, InferredType.MONEY, InferredType.ADDRESS}:
        return FailureType.NORMALIZATION_ERROR
    return FailureType.USER_PREFERENCE_FORMATTING

