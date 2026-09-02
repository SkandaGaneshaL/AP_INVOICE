from __future__ import annotations

from typing import Any

from .layout_graph import LayoutGraph, find_kv_candidates
from .operators import BindResult, CandidateValue, CompetingHit, EvidenceHit, FieldContext, FieldProgram
from .transform_induction import apply_program


def bind_field(field_program: FieldProgram, graph_or_candidates: LayoutGraph | list[CandidateValue],
               field_context: FieldContext | dict[str, Any] | None = None) -> BindResult:
    """Bind a field program to layout candidates; retain the legacy list adapter."""
    if isinstance(graph_or_candidates, LayoutGraph):
        candidates = find_kv_candidates(graph_or_candidates, field_program,
                                        section_prior=field_program.section_prior or field_program.select.section_prior)
    else:
        candidates = graph_or_candidates
    selected: list[CandidateValue] = []
    ignored: list[CompetingHit] = []
    for candidate in candidates:
        if field_program.select.label_aliases and candidate.source_label and not any(
                alias.casefold() in candidate.source_label.casefold() for alias in field_program.select.label_aliases):
            ignored.append(CompetingHit.model_validate({**candidate.model_dump(), "reason": "label_not_preferred"}))
            continue
        if field_program.disambiguate.ignore and candidate.source_label and any(
                token.casefold() in candidate.source_label.casefold() for token in field_program.disambiguate.ignore):
            ignored.append(CompetingHit.model_validate({**candidate.model_dump(), "reason": "configured_ignore"}))
            continue
        selected.append(candidate)
    if field_program.disambiguate.prefer:
        preferred = [item for item in selected if item.source_label and any(
            token.casefold() in item.source_label.casefold() for token in field_program.disambiguate.prefer)]
        if preferred:
            selected = preferred
    if len(selected) != 1:
        status = "none" if not selected else "ambiguous"
        return BindResult(status=status, evidence=[EvidenceHit.model_validate(item.model_dump()) for item in selected],
                          competing_candidates=ignored, debug_trace=["label/spatial selection", "candidate count=%d" % len(selected)])
    candidate = selected[0]
    raw = candidate.raw_value if candidate.raw_value is not None else candidate.value
    value = candidate.canonical_value if candidate.canonical_value is not None else candidate.value
    transformed = apply_program(field_program, value)
    return BindResult(status="unique", selected_candidate=candidate, transformed_value=transformed,
                      raw_value=raw, canonical_value=transformed,
                      evidence=[EvidenceHit.model_validate(candidate.model_dump())], competing_candidates=ignored,
                      confidence=candidate.confidence, debug_trace=[item.op for item in field_program.transform])
