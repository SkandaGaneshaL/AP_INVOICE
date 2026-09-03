from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import EvaluationResult, RuleFeedbackPacket, RuleGenerationContext, UsageSummary
from .normalization import apply_policy, infer_policy
from .candidate_binder import bind_field
from .layout_graph import LayoutGraph
from .operators import CorrectionExample, FieldProgram
from .transform_induction import apply_program
from .sentence_validators import validate_sentence


class ExtractionExecutor(Protocol):
    def extract(self, *, document_bytes: bytes, mime_type: str, field_key: str,
                field_path: str, instruction: str, rules: list[str]) -> Any: ...


@dataclass
class EvaluationTrace:
    field_path: str
    instruction: str
    extracted_value: Any
    expected_value: Any
    score: float
    feedback: str
    format_valid: bool = True
    contradiction: bool = False
    expected_page: int | None = None
    actual_page: int | None = None
    match: bool = False
    field_key: str = ""
    evidence_snippets: list[str] = field(default_factory=list)
    competing_snippets: list[str] = field(default_factory=list)
    evidence_supported: bool = False
    unaffected_preserved: bool = True
    null_behavior_valid: bool = True
    reusable_wording: bool = True
    schema_valid: bool = True
    usage: UsageSummary | None = None


def evaluate_candidate_program(graph: LayoutGraph, current_programs: dict[str, FieldProgram], field_key: str,
                               candidate_program: FieldProgram, correction: CorrectionExample) -> EvaluationResult:
    """Evaluate a candidate against the current layout without calling an LLM."""
    del current_programs
    context = {"field_key": field_key, "old_value": correction.old_value, "new_value": correction.new_value}
    bound = bind_field(candidate_program, graph, context)
    expected = correction.new_value
    match = _same(bound.transformed_value, expected)
    evidence_supported = bound.status == "unique" and bool(bound.evidence)
    score = 1.0 if match and evidence_supported else .0
    return EvaluationResult(score=score, feedback="Counterfactual program matched the corrected value and layout evidence." if score else "Counterfactual program did not produce a unique supported match.",
                             confidence="normal" if score else "failed", field_match=match,
                             evidence_supported=evidence_supported, schema_valid=bound.status == "unique",
                             candidate_status="accepted_with_transformation" if score else "rejected",
                             promotion_eligible=score >= .9, expected_value=expected,
                             actual_value=bound.transformed_value, canonical_actual_value=bound.canonical_value,
                             transformation_expected=(candidate_program.transform[0].op if candidate_program.transform else None))


def evaluate_program_counterfactual(program: FieldProgram, raw_evidence_value: Any,
                                    corrected_value: Any, evidence_hit: Any = None,
                                    competing_hits: list[Any] | None = None,
                                    existing_programs: dict[str, FieldProgram] | None = None) -> EvaluationResult:
    """Evaluate the executable program without extraction-model calls."""
    del existing_programs
    actual = apply_program(program, raw_evidence_value)
    match = _same(actual, corrected_value)
    supported = evidence_hit is not None
    # Distinct order/shipment references remain audit evidence and do not
    # invalidate a correctly labelled match. Only a competitor supporting the
    # same corrected value is a promotion blocker.
    competing = any(
        _norm(getattr(hit, "normalized_value", None) or getattr(hit, "value", None))
        == _norm(corrected_value)
        for hit in (competing_hits or [])
    )


def evaluate_sentence_gates(sentence: str, context: RuleGenerationContext) -> EvaluationResult:
    """Cheap text/evidence gate; avoids a second PDF extraction per candidate."""
    try:
        payload = {"old_value": context.old_value, "new_value": context.new_value}
        validate_sentence(sentence, payload)
    except ValueError as exc:
        return EvaluationResult(score=0.0, confidence="failed", feedback=str(exc),
                                 candidate_status="rejected", promotion_eligible=False, schema_valid=False)
    packet = context.feedback_packet
    # Legacy callers without a document packet are still valid unit-test and
    # API integrations; document-backed production candidates require actual
    # evidence support.
    supported = packet is None or bool(packet.evidence)
    return EvaluationResult(score=1.0 if supported else 0.0, confidence="normal" if supported else "failed",
                            feedback="Sentence passed compact safety and evidence gates." if supported else "No supporting evidence was available.",
                            field_match=True, evidence_supported=supported, schema_valid=True,
                            candidate_status="evaluated" if supported else "rejected",
                            promotion_eligible=supported)
    score = 1.0 if match and supported else 0.0
    return EvaluationResult(
        score=score,
        feedback=("Counterfactual program matched the corrected value and selected evidence."
                  if score else "Counterfactual program did not match supported evidence."),
        confidence="normal" if score else "failed",
        field_match=match,
        evidence_supported=supported,
        schema_valid=True,
        candidate_status="accepted_with_transformation" if score else "rejected",
        promotion_eligible=bool(match and supported and not competing),
        expected_value=corrected_value,
        actual_value=actual,
        canonical_actual_value=actual,
        transformation_expected=(program.transform[0].op if program.transform else None),
    )


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _value(node: Any) -> Any:
    return node.get("value") if isinstance(node, dict) else node


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list):
            return False
        return [_norm(_value(x)) for x in left] == [_norm(_value(x)) for x in right]
    if left is None or right is None:
        return left is None and right is None
    return _norm(left) == _norm(right)


def canonicalize(field_key: str, value: Any, context: RuleGenerationContext) -> Any:
    """Canonicalize only the actual value under the explicit correction policy."""
    return apply_policy(field_key, value, context.normalization_mode)


class ExtractionEvaluator:
    """Evaluate the actual extraction result and produce GEPA diagnostics."""

    def __init__(self, executor: ExtractionExecutor | None = None):
        self.executor = executor

    def evaluate(self, context: RuleGenerationContext, instruction: str) -> tuple[EvaluationResult, EvaluationTrace]:
        if context.normalization_mode == "none":
            context.normalization_mode = infer_policy(
                context.field_key, context.old_value, context.new_value
            ).mode
        packet = context.feedback_packet
        if isinstance(packet, dict):
            try:
                packet = RuleFeedbackPacket.model_validate(packet)
                context.feedback_packet = packet
            except Exception:
                feedback = "The GEPA example contained an invalid feedback packet and could not be evaluated."
                return EvaluationResult(score=0.0, confidence="failed", feedback=feedback), EvaluationTrace(
                    context.field_path, instruction, None, context.new_value, 0.0, feedback,
                    format_valid=False, field_key=context.field_key, schema_valid=False)
        evidence = packet.evidence if packet else []
        competing = packet.competing_evidence if packet else []
        snippets = [x.snippet for x in evidence if x.snippet]
        competing_snippets = [x.snippet for x in competing if x.snippet]
        expected_node = context.final_response.get(context.field_key, {})
        expected_value = _value(expected_node)
        expected_page = expected_node.get("Page") if isinstance(expected_node, dict) else None
        if self.executor is None:
            feedback = "No extraction executor is configured; GEPA cannot claim extraction improvement."
            return EvaluationResult(score=None, confidence="unavailable", feedback=feedback), EvaluationTrace(
                context.field_path, instruction, None, expected_value, 0.0, feedback,
                field_key=context.field_key, evidence_snippets=snippets, competing_snippets=competing_snippets)
        try:
            if not context.document_bytes:
                raise ValueError("document bytes are missing from the evaluation context")
            full = getattr(self.executor, "extract_full", None)
            usage = UsageSummary()
            full_with_metadata = getattr(self.executor, "extract_full_with_metadata", None)
            if callable(full_with_metadata):
                actual_document, diagnostics = full_with_metadata(
                    document_bytes=context.document_bytes, mime_type=context.mime_type,
                    instruction=instruction, normalization_mode=context.normalization_mode,
                )
                usage = UsageSummary.model_validate(diagnostics.get("usage", {}))
                actual = actual_document.get(context.field_key, {"value": None, "Page": None})
                full = None
            if callable(full):
                try:
                    actual_document = full(document_bytes=context.document_bytes, mime_type=context.mime_type,
                                           instruction=instruction, normalization_mode=context.normalization_mode)
                except TypeError as exc:
                    if "normalization_mode" not in str(exc):
                        raise
                    actual_document = full(document_bytes=context.document_bytes, mime_type=context.mime_type,
                                           instruction=instruction)
                actual = actual_document.get(context.field_key, {"value": None, "Page": None})
            else:
                actual_document = None
                try:
                    actual = self.executor.extract(document_bytes=context.document_bytes, mime_type=context.mime_type,
                        field_key=context.field_key, field_path=context.field_path, instruction=instruction,
                        rules=context.detailed_rule, normalization_mode=context.normalization_mode)
                except TypeError as exc:
                    if "normalization_mode" not in str(exc):
                        raise
                    actual = self.executor.extract(document_bytes=context.document_bytes, mime_type=context.mime_type,
                        field_key=context.field_key, field_path=context.field_path, instruction=instruction,
                        rules=context.detailed_rule)
            schema_valid = isinstance(actual, dict) and "value" in actual and "Page" in actual
            actual_value = _value(actual)
            canonical_actual = canonicalize(context.field_key, actual_value, context)
            # The corrected value is the user's ground truth. Never normalize it
            # using the old production rule or the candidate's policy.
            canonical_expected = expected_value
            actual_page = actual.get("Page") if isinstance(actual, dict) else None
            match = _same(canonical_actual, canonical_expected)
            page_match = expected_page is None or actual_page == expected_page
            evidence_supported = bool(match and evidence and any(
                _same(canonical_actual, item.normalized_value if item.normalized_value is not None else item.value)
                and (actual_page is None or item.page == actual_page)
                for item in evidence))
            null_behavior = expected_value is not None or actual_value is None
            unaffected = True
            if isinstance(actual_document, dict):
                for key, expected in context.final_response.items():
                    if key != context.field_key and key in actual_document and not _same(
                        _value(actual_document[key]), _value(expected)):
                        unaffected = False
                        break
            lower = instruction.casefold()
            new_text = _norm(context.new_value)
            hardcoded = bool(new_text and (f"always use {new_text}" in lower or
                                           f"the value is {new_text}" in lower or
                                           f"use {new_text} as" in lower))
            reusable = not hardcoded
            contradiction = hardcoded or (bool(competing) and not evidence_supported and any(
                _norm(item.value) == new_text for item in competing))
            hard_gate = bool(match and schema_valid and null_behavior)
            score = (0.25 * match + 0.15 * page_match + 0.15 * evidence_supported +
                     0.15 * unaffected + 0.10 * null_behavior + 0.05 * schema_valid +
                     0.05 * (not contradiction) + 0.10 * reusable)
            parts = []
            if not match:
                parts.append(f"Expected {expected_value!r}, but extracted {actual_value!r}.")
            if not page_match:
                parts.append(f"Expected source page {expected_page}, got {actual_page}.")
            if not evidence_supported and evidence:
                parts.append("Prefer a value supported by the matched field label evidence.")
            if competing and not evidence_supported:
                parts.append("Competing evidence exists; ignore unrelated reference or conversion occurrences.")
            if not unaffected:
                parts.append("The candidate changed an unaffected field; preserve existing extraction behavior.")
            if contradiction:
                parts.append("Do not hard-code the corrected invoice value or contradict the evidence.")
            feedback = " ".join(parts) or "Matched the corrected value and supporting document evidence while preserving unaffected fields."
            result = EvaluationResult(
                score=score, feedback=feedback, confidence="normal",
                field_match=match, evidence_supported=evidence_supported,
                schema_valid=schema_valid, candidate_status="accepted" if hard_gate else "rejected",
                promotion_eligible=hard_gate and evidence_supported and not contradiction,
                expected_value=expected_value, actual_value=actual_value,
                canonical_actual_value=canonical_actual,
                transformation_expected=(evidence[0].transformation if evidence else None),
            )
            trace = EvaluationTrace(context.field_path, instruction, actual_value, expected_value, score, feedback,
                format_valid=schema_valid, contradiction=contradiction, expected_page=expected_page,
                actual_page=actual_page, match=match, field_key=context.field_key,
                evidence_snippets=snippets, competing_snippets=competing_snippets,
                evidence_supported=evidence_supported, unaffected_preserved=unaffected,
                null_behavior_valid=null_behavior, reusable_wording=reusable, schema_valid=schema_valid,
                usage=usage)
            return result, trace
        except Exception as exc:
            feedback = f"Extraction evaluation failed: {type(exc).__name__}: {str(exc)[:240]}"
            return EvaluationResult(score=0.0, confidence="failed", feedback=feedback), EvaluationTrace(
                context.field_path, instruction, None, expected_value, 0.0, feedback, format_valid=False,
                field_key=context.field_key, evidence_snippets=snippets, competing_snippets=competing_snippets,
                schema_valid=False)
