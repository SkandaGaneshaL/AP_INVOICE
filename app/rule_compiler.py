from __future__ import annotations

from pydantic import Field
from .operators import CompiledRule, CorrectionExample, EvidenceBundle, FieldProgram, OperatorCandidate, TransformOp


def compile_program(program: FieldProgram) -> CompiledRule:
    ops = [item.op for item in program.transform if item.op != "identity"]
    action = " and ".join(ops) if ops else "identity"
    label = program.select.label_aliases[0] if program.select.label_aliases else "the explicitly labeled field"
    sentence = program.sentence or f"Extract the value from {label} and apply {action} normalization when applicable."
    return CompiledRule(program=program, sentence=sentence)


def merge_program(existing: FieldProgram | None, candidate: FieldProgram) -> CompiledRule:
    if existing is None:
        return compile_program(candidate)
    merged = existing.model_copy(deep=True)
    existing_ops = {item.op for item in merged.transform}
    for op in candidate.transform:
        if op.op not in existing_ops:
            merged.transform.append(op)
    if candidate.select.label_aliases:
        merged.select.label_aliases = list(dict.fromkeys(merged.select.label_aliases + candidate.select.label_aliases))
    return compile_program(merged)


class RuleCompileResult(CompiledRule):
    program_before: FieldProgram | None = None
    selected_operator: str | None = None
    audit_diff: dict = Field(default_factory=dict)
    conflict_status: str = "none"
    promoted_candidate: bool = False
    requires_llm_binding: bool = False
    reasons: list[str] = Field(default_factory=list)


def compile_rule_update(current_program: FieldProgram | None, correction: CorrectionExample,
                        operator_candidates: list[OperatorCandidate], evidence_bundle: EvidenceBundle,
                        history: list[CorrectionExample] | None = None) -> RuleCompileResult:
    """Merge the lowest-cost consistent operator into a structured program."""
    del evidence_bundle, history
    selected = next((item for item in sorted(operator_candidates, key=lambda item: item.mdl_cost or item.score_cost) if item.matched), None)
    before = current_program.model_copy(deep=True) if current_program else None
    if selected is None:
        return RuleCompileResult(program=current_program or FieldProgram(), sentence=(current_program.sentence if current_program else ""),
                                 program_before=before, requires_llm_binding=True, reasons=["no deterministic operator matched"])
    after = (current_program or FieldProgram()).model_copy(deep=True)
    after.type = selected.program.type or after.type
    existing = {item.op for item in after.transform}
    for transform in selected.program.transform:
        if transform.op != "identity" and transform.op not in existing:
            after.transform.append(transform)
    if correction.label_text and correction.label_text not in after.select.label_aliases:
        after.select.label_aliases.append(correction.label_text)
    after.sentence = None
    compiled = compile_program(after)
    return RuleCompileResult(program=compiled.program, sentence=compiled.sentence, program_before=before,
                             selected_operator=selected.program.transform[0].op,
                             audit_diff={"program_before": before.model_dump() if before else None,
                                         "program_after": compiled.program.model_dump(),
                                         "added_operator": selected.program.transform[0].op},
                             conflict_status="none", promoted_candidate=True,
                             requires_llm_binding=False, reasons=[selected.rationale])
