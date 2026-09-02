from __future__ import annotations

from typing import Any
from .operators import OperatorCandidate
from .type_inference import InferredType


def pack_rule_context(*, field_key: str, display_label: str, old_value: Any, new_value: Any,
                      inferred_type: InferredType, edit_script: str, candidates: list[OperatorCandidate],
                      evidence_hits: list[dict[str, Any]], competing_hits: list[dict[str, Any]],
                      current_program: dict[str, Any] | None = None, historical_examples=None) -> dict[str, Any]:
    return {
        "field_key": field_key,
        "display_label": display_label,
        "correction": {"old": old_value, "new": new_value, "edit_script": edit_script, "inferred_type": inferred_type.value},
        "candidates": [candidate.model_dump() for candidate in candidates[:7]],
        "evidence_hits": [{k: item.get(k) for k in ("page", "source_label", "evidence_text", "confidence", "raw_value")} for item in evidence_hits[:3]],
        "competing_hits": [{k: item.get(k) for k in ("page", "source_label", "evidence_text", "confidence")} for item in competing_hits[:2]],
        "current_program": current_program or {},
        "historical_examples": (historical_examples or [])[:2],
        "output_schema": {"program": {"transform": [{"op": "identity", "when": None}]}, "sentence": "one imperative sentence"},
    }

