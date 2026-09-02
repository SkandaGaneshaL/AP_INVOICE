from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class SelectOp(BaseModel):
    label_aliases: list[str] = Field(default_factory=list)
    spatial: str = "right_of_or_same_cell"
    section_prior: str | None = None


class DisambiguateOp(BaseModel):
    prefer: list[str] = Field(default_factory=list)
    ignore: list[str] = Field(default_factory=list)


class TransformOp(BaseModel):
    op: str = "identity"
    when: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class FieldProgram(BaseModel):
    select: SelectOp = Field(default_factory=SelectOp)
    disambiguate: DisambiguateOp = Field(default_factory=DisambiguateOp)
    transform: list[TransformOp] = Field(default_factory=lambda: [TransformOp()])
    null_policy: str = "labeled_empty_to_null"
    section_prior: str | None = None
    sentence: str | None = None
    type: str = "TEXT"


class CandidateValue(BaseModel):
    value: Any = None
    raw_value: str | None = None
    canonical_value: Any = None
    page: int | None = None
    bbox: list[float] | None = None
    source_label: str | None = None
    evidence_text: str | None = None
    confidence: float = 0.0


class EvidenceHit(CandidateValue):
    score: float = 0.0
    match_type: str = "exact"


class CompetingHit(EvidenceHit):
    reason: str = "competing_candidate"


class OperatorCandidate(BaseModel):
    program: FieldProgram
    score_cost: float = 0.0
    mismatches: int = 0
    rationale: str = ""
    explanation: str = ""
    mdl_cost: float = 0.0
    matched: bool = False
    failure_type: str = "unknown"
    examples_used: int = 0
    risk_flags: list[str] = Field(default_factory=list)


class CompiledRule(BaseModel):
    program: FieldProgram
    sentence: str


class BindResult(BaseModel):
    status: Literal["unique", "none", "ambiguous", "invalid"]
    selected_candidate: CandidateValue | None = None
    transformed_value: Any = None
    evidence: list[EvidenceHit] = Field(default_factory=list)
    competing_candidates: list[CompetingHit] = Field(default_factory=list)
    debug_trace: list[str] = Field(default_factory=list)
    raw_value: Any = None
    canonical_value: Any = None
    confidence: float = 0.0


class EvidenceBundle(BaseModel):
    evidence_hits: list[EvidenceHit] = Field(default_factory=list)
    competing_hits: list[CompetingHit] = Field(default_factory=list)
    selected_edge: dict[str, Any] | None = None
    competing_edge: dict[str, Any] | None = None
    section: str | None = None
    bbox_relation: str | None = None
    score: float = 0.0
    debug_trace: list[str] = Field(default_factory=list)


class CorrectionExample(BaseModel):
    field_key: str = ""
    field_type: str = "TEXT"
    failure_type: str = "unknown"
    old_value: Any = None
    new_value: Any = None
    edit_script: list[dict[str, Any]] = Field(default_factory=list)
    selected_operator: str | None = None
    program_before: dict[str, Any] = Field(default_factory=dict)
    program_after: dict[str, Any] = Field(default_factory=dict)
    evidence_signature: str | None = None
    layout_signature: str | None = None
    label_text: str = ""
    promoted: bool = False


class FieldContext(BaseModel):
    field_key: str = ""
    display_label: str = ""
    old_value: Any = None
    new_value: Any = None
    field_type: str = "TEXT"
    section_prior: str | None = None
    label_aliases: list[str] = Field(default_factory=list)
