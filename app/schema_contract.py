from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ExtractionField(BaseModel):
    value: Any = None
    raw_value: str | None = None
    canonical_value: Any = None
    page: int | None = None
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    source_label: str | None = None
    evidence_text: str | None = None
    confidence: float = 0.0
    absent_reason: Literal["not_present", "labeled_empty", "unreadable", "not_applicable"] | None = None


def normalize_field(value: Any) -> dict[str, Any]:
    """Return the public extraction representation for one scalar field.

    OCI/Gemini may return provenance attributes alongside ``value`` and
    ``Page``.  Those attributes are useful to provider-specific pipelines but
    are intentionally not part of the public extraction contract.  Projecting
    here also prevents unknown fields from leaking into Streamlit or into the
    corrected JSON sent back to the update API.
    """
    if value is None:
        return {"value": None, "Page": None}
    if isinstance(value, dict):
        extracted_value = value.get("value")
        if isinstance(extracted_value, str) and extracted_value.strip() in {"Not present", "No value", "N/A"}:
            extracted_value = None
        return {"value": extracted_value, "Page": value.get("Page")}
    return {"value": value, "Page": None}
