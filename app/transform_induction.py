from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterable

from .operators import CompetingHit, CorrectionExample, EvidenceHit, FieldProgram, OperatorCandidate, TransformOp
from .type_inference import FieldType, InferredType
from .type_inference import parse_date as parse_date_value, parse_number

_IDENTIFIER_CORE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_DATE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$")
_MONEY = re.compile(r"^[\s$€£₹A-Z]*[-+]?\d[\d\s,.' ]*[.,]\d{2}$", re.I)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _tokens(value: str) -> list[str]:
    return [part for part in re.split(r"[\s,;|]+", value.strip()) if part]


def _identifier_remainder(value: str) -> bool:
    return bool(_IDENTIFIER_CORE.fullmatch(value.strip())) and any(ch.isdigit() for ch in value)


def _strip_leading_alpha(value: str) -> str | None:
    match = re.match(r"^([A-Za-z]+)(?:[\s_-]+)(.+)$", value.strip())
    if match and _identifier_remainder(match.group(2)):
        return match.group(2).strip()
    return None


def _strip_trailing_alpha(value: str) -> str | None:
    match = re.match(r"^(.+?)(?:[-_\s]+)([A-Za-z]+)$", value.strip())
    return match.group(1).strip() if match and _identifier_remainder(match.group(1)) else None


def _parse_money(value: str) -> str:
    parsed = parse_number(value)
    return format(parsed, "f") if parsed is not None else re.sub(r"[^0-9.-]", "", value).replace(",", "")


def apply_program(program: FieldProgram, value: Any) -> Any:
    current = value
    for transform in program.transform:
        op = transform.op
        text = _text(current)
        if op == "identity":
            continue
        if op == "trim":
            current = text.strip()
        elif op == "collapse_ws":
            current = " ".join(text.split())
        elif op == "case_fold":
            current = text.casefold()
        elif op in {"strip_leading_alpha_token", "identifier_strip_leading_alpha"}:
            current = _strip_leading_alpha(text) or current
        elif op == "strip_trailing_alpha":
            current = _strip_trailing_alpha(text) or current
        elif op == "keep_leading_alpha_prefix":
            match = re.match(r"^([A-Za-z]+(?:[-_ ]?[A-Za-z]+)?)", text.strip())
            current = match.group(1) if match else current
        elif op == "split_token":
            index = int(transform.args.get("index", 0))
            parts = _tokens(text)
            current = parts[index] if 0 <= index < len(parts) else current
        elif op in {"parse_money", "money_canonicalize", "numeric_thousands_canonicalize"}:
            current = _parse_money(text)
        elif op in {"parse_date", "date_format_repattern"}:
            parsed = parse_date_value(text)
            if parsed:
                target = transform.args.get("target_pattern")
                if target:
                    output_format = {"DD/MM/YYYY": "%d/%m/%Y", "MM/DD/YYYY": "%m/%d/%Y", "YYYY-MM-DD": "%Y-%m-%d"}.get(target, "%d/%m/%Y")
                else:
                    output_format = "%d/%m/%Y"
                current = parsed[0].strftime(output_format)
        elif op == "join_address_block":
            current = " ".join(text.splitlines())
        elif op == "drop_label_echo":
            label = str(transform.args.get("label", ""))
            current = re.sub(r"^" + re.escape(label) + r"\s*[:#-]?\s*", "", text, flags=re.I).strip()
        elif op == "regex_capture":
            match = re.search(str(transform.args.get("pattern", "")), text)
            if match:
                current = match.group(int(transform.args.get("group", 1)))
        elif op == "null_if_unlabeled":
            current = None
    return current


def _candidate(op: str, when: str | None, cost: float, rationale: str, *, field_type: str, matched: bool = True, args=None) -> OperatorCandidate:
    program = FieldProgram(type=field_type, transform=[TransformOp(op=op, when=when, args=args or {})])
    return OperatorCandidate(program=program, score_cost=cost, mdl_cost=cost,
                             rationale=rationale, explanation=rationale, matched=matched)


def _history_mismatches(candidate: OperatorCandidate, history: Iterable[Any] | None) -> int:
    mismatches = 0
    for example in history or []:
        old = example.get("old_value") if isinstance(example, dict) else getattr(example, "old_value", None)
        expected = example.get("new_value") if isinstance(example, dict) else getattr(example, "new_value", None)
        if old is not None and _text(apply_program(candidate.program, old)).strip().casefold() != _text(expected).strip().casefold():
            mismatches += 1
    return mismatches


def induce_transform_candidates(
    field_type: FieldType | InferredType | str,
    old_value: Any,
    new_value: Any,
    evidence_hits: list[EvidenceHit] | None = None,
    competing_hits: list[CompetingHit] | None = None,
    history: list[CorrectionExample] | list[dict[str, Any]] | None = None,
    max_depth: int = 3,
    beam_size: int = 8,
) -> list[OperatorCandidate]:
    """Synthesize a bounded ranked DSL program from an input/output example."""
    evidence_hits = evidence_hits or []
    competing_hits = competing_hits or []
    max_depth = max(1, min(int(max_depth), 3))
    beam_size = max(1, min(int(beam_size), 8))
    kind = str(getattr(field_type, "value", field_type))
    old, new = _text(old_value), _text(new_value)
    candidates: list[OperatorCandidate] = []
    if old.strip() == new.strip():
        candidates.append(_candidate("identity", None, 1, "values are unchanged after trimming", field_type=kind))
    if old != old.strip() and old.strip() == new:
        candidates.append(_candidate("trim", "leading_or_trailing_whitespace", 1, "only surrounding whitespace changed", field_type=kind))
    if " ".join(old.split()) == new and old != new:
        candidates.append(_candidate("collapse_ws", "whitespace_only", 1, "only whitespace runs changed", field_type=kind))
    if old.casefold() == new.casefold() and old != new:
        candidates.append(_candidate("case_fold", "case_only", 1, "only character case changed", field_type=kind))
    if kind == FieldType.IDENTIFIER.value:
        if (_strip_leading_alpha(old) or "").casefold() == new.strip().casefold() and _strip_leading_alpha(old) is not None:
            candidates.append(_candidate("strip_leading_alpha_token", "remainder_is_numeric_or_identifier_core", 1,
                                         "the corrected value is the identifier core after a leading alphabetic token", field_type=kind))
            candidates.append(_candidate("identifier_strip_leading_alpha", "remainder_is_identifier_core", 1.05,
                                         "the corrected value is the identifier core after a leading alphabetic prefix", field_type=kind))
        if (_strip_trailing_alpha(old) or "").casefold() == new.strip().casefold() and _strip_trailing_alpha(old) is not None:
            candidates.append(_candidate("strip_trailing_alpha", "prefix_is_numeric_or_identifier_core", 1,
                                         "the corrected value removes a trailing alphabetic suffix", field_type=kind))
    for index, part in enumerate(_tokens(old)):
        if part.casefold() == new.strip().casefold() and len(_tokens(old)) > 1:
            candidates.append(_candidate("split_token", f"token_{index}", 2, f"the correction selects token {index}", field_type=kind, args={"index": index}))
    if kind == FieldType.DATE.value and _DATE.fullmatch(old.strip()) and _DATE.fullmatch(new.strip()) and old != new:
        parsed_old, parsed_new = parse_date_value(old), parse_date_value(new)
        if parsed_old and parsed_new and parsed_old[0] == parsed_new[0]:
            target = "MM/DD/YYYY" if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", new.strip()) and int(new.split("/")[0]) <= 12 else None
            candidates.append(_candidate("date_format_repattern", "same_calendar_date", 1, "the correction changes date representation", field_type=kind, args={"target_pattern": target} if target else {}))
    if kind in {FieldType.MONEY.value, "NUMBER"} and (_MONEY.fullmatch(old.strip()) or _MONEY.fullmatch(new.strip())) and _parse_money(old) == _parse_money(new):
        candidates.append(_candidate("numeric_thousands_canonicalize", "same_numeric_amount", 1, "the correction changes grouping separators or currency decoration", field_type=kind))
    if "\n" in old and " ".join(old.splitlines()) == new.strip():
        candidates.append(_candidate("join_address_block", "same_address_components", 1, "line breaks changed without changing address components", field_type=kind))
    if new.strip().casefold() in {"", "null", "none", "n/a"}:
        candidates.append(_candidate("null_if_unlabeled", "correction_is_null", 1, "the correction explicitly removes an unsupported value", field_type=kind))
    candidates.append(_candidate("identity", "preserve_corrected_value", 4, "preserve the corrected value without a known deterministic transform", field_type=kind, matched=False))
    for candidate in candidates:
        candidate.mismatches = _history_mismatches(candidate, history)
        candidate.mdl_cost = candidate.score_cost + 2.0 * candidate.mismatches
        candidate.score_cost = candidate.mdl_cost
        candidate.examples_used = len(history or [])
        if candidate.mismatches:
            candidate.risk_flags.append("history_mismatch")
    unique: dict[tuple[str, str], OperatorCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: (item.mdl_cost, item.program.transform[0].op)):
        key = (candidate.program.transform[0].op, str(candidate.program.transform[0].args))
        unique.setdefault(key, candidate)
    # The current DSL uses primitive candidates; the bounded beam contract is
    # still enforced here and leaves room for composed operators to be added.
    return list(unique.values())[: max(1, min(beam_size, 8))]


def induce_transforms(old_value: Any, new_value: Any, inferred_type: InferredType, history: list[dict[str, Any]] | None = None) -> list[OperatorCandidate]:
    return induce_transform_candidates(inferred_type, old_value, new_value, [], [], history=history)


def levenshtein_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def token_lcs(left: str, right: str) -> list[str]:
    a, b = _tokens(left), _tokens(right)
    table = [[[] for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            table[i + 1][j + 1] = table[i][j] + [a[i]] if a[i].casefold() == b[j].casefold() else max(table[i][j + 1], table[i + 1][j], key=len)
    return table[-1][-1]
