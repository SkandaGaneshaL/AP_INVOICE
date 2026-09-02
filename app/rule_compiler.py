from __future__ import annotations

from pydantic import Field
from .operators import CompiledRule, CorrectionExample, EvidenceBundle, FieldProgram, OperatorCandidate, TransformOp
from .sentence_templates import render_sentence


def compile_program(program: FieldProgram) -> CompiledRule:
    sentence = program.sentence or render_sentence(program)
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
    promotion_eligible: bool = False
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
        if transform.op == "identity":
            continue
        # Canonicalization and date-format operators are exclusive slots; a
        # new correction replaces an incompatible prior transform instead of
        # creating contradictory prose.
        exclusive = {
            "date_format_repattern", "parse_date",
            "numeric_thousands_canonicalize", "money_canonicalize", "parse_money",
            "identifier_strip_leading_alpha", "strip_leading_alpha_token",
        }
        if transform.op in exclusive:
            after.transform = [item for item in after.transform if item.op not in exclusive]
        if transform.op not in {item.op for item in after.transform}:
            after.transform.append(transform)
        existing.add(transform.op)
    if correction.label_text and correction.label_text not in after.select.label_aliases:
        after.select.label_aliases.append(correction.label_text)
    if selected.program.format.target_pattern:
        after.format.target_pattern = selected.program.format.target_pattern
    for transform in selected.program.transform:
        if transform.args.get("target_pattern"):
            after.format.target_pattern = transform.args["target_pattern"]
    after.sentence = None
    compiled = compile_program(after)
    return RuleCompileResult(program=compiled.program, sentence=compiled.sentence, program_before=before,
                             selected_operator=selected.program.transform[0].op,
                             audit_diff={"program_before": before.model_dump() if before else None,
                                         "program_after": compiled.program.model_dump(),
                                         "added_operator": selected.program.transform[0].op},
                             conflict_status="none", promoted_candidate=True, promotion_eligible=True,
                             requires_llm_binding=False, reasons=[selected.rationale])
