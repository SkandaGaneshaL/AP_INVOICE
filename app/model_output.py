import json
import re
from typing import Any


class ModelOutputError(ValueError):
    def __init__(self, reason: str, *, finish_reason: str | None = None, text_length: int = 0, prefix: str = "", diagnostics: dict[str, Any] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.finish_reason = finish_reason
        self.text_length = text_length
        self.prefix = prefix
        self.diagnostics = diagnostics or {}


def _strip_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def _embedded_json(text: str) -> str | None:
    start = text.find('{')
    while start >= 0:
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        value = json.loads(candidate)
                        if isinstance(value, dict) and isinstance(value.get("sentence"), str):
                            return value["sentence"]
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find('{', start + 1)
    return None


def _one_sentence(text: str) -> str:
    value = text.strip().strip('"`')
    if not value or "{" in value or "}" in value or "```" in value or value.startswith(("#", "Explanation:", "Here is")):
        raise ValueError("response is not a plain rule sentence")
    if len(re.findall(r"[.!?](?:\s|$)", value)) > 1:
        raise ValueError("response contains more than one sentence")
    return value


def parse_rule_response(text: Any, *, finish_reason: str | None = None) -> tuple[str, str]:
    if not isinstance(text, str):
        raise ModelOutputError("response content is not text", finish_reason=finish_reason)
    cleaned = _strip_fence(text.lstrip("\ufeff").strip())
    prefix = cleaned[:160].replace("\r", " ").replace("\n", " ")
    if not cleaned:
        raise ModelOutputError("response is empty", finish_reason=finish_reason, prefix=prefix)
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("sentence"), str):
            raise ValueError("JSON must be an object containing a string sentence")
        return _one_sentence(parsed["sentence"]), "structured_json"
    except (json.JSONDecodeError, ValueError):
        embedded = _embedded_json(cleaned)
        if embedded is not None:
            try:
                return _one_sentence(embedded), "structured_json"
            except ValueError:
                pass
        try:
            return _one_sentence(cleaned), "plain_sentence"
        except ValueError as exc:
            raise ModelOutputError("response was neither valid structured JSON nor one valid sentence",
                                   finish_reason=finish_reason, text_length=len(cleaned), prefix=prefix) from exc


def parse_rule_parts(parts: list[Any], *, finish_reason: str | None = None,
                     diagnostics: dict[str, Any] | None = None) -> tuple[str, str]:
    candidates = []
    for index, part in enumerate(parts):
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if not isinstance(text, str) or not text.strip():
            continue
        thought = bool(part.get("thought", False) if isinstance(part, dict) else getattr(part, "thought", False))
        candidates.append((index, text, thought))
    metadata = {"content_part_count": len(parts),
                "candidate_lengths": [len(x[1]) for x in candidates],
                "candidate_prefixes": [x[1][:80].replace("\n", " ") for x in candidates]}
    metadata.update(diagnostics or {})
    errors = []
    if str(finish_reason or "").upper() in {"MAX_TOKENS", "LENGTH"}:
        raise ModelOutputError("model output was truncated by the token limit", finish_reason=finish_reason,
                               text_length=sum(len(x[1]) for x in candidates),
                               prefix=metadata["candidate_prefixes"][0] if candidates else "",
                               diagnostics=metadata)
    for index, text, thought in reversed(candidates):
        if thought:
            continue
        try:
            return parse_rule_response(text, finish_reason=finish_reason)
        except ModelOutputError as exc:
            errors.append(f"part[{index}]: {exc.reason}")
    reason = "no valid final text part"
    if candidates and all(x[2] for x in candidates):
        reason = "only reasoning parts were returned"
    elif errors:
        reason = "all text parts failed rule-output validation"
    raise ModelOutputError(reason, finish_reason=finish_reason,
                           text_length=sum(len(x[1]) for x in candidates),
                           prefix=metadata["candidate_prefixes"][0] if candidates else "",
                           diagnostics={**metadata, "part_errors": errors})


_UNSAFE_SUMMARY_TERMS = re.compile(
    r"chain[- ]of[- ]thought|hidden reasoning|private reasoning|internal analysis|"
    r"system prompt|user prompt|tool trace|scratchpad|model thought",
    re.IGNORECASE,
)


def _safe_decision_summary(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    summary = " ".join(value.strip().split())
    if not summary or len(summary) > 320 or _UNSAFE_SUMMARY_TERMS.search(summary):
        return None
    if any(char in summary for char in "{}[]`"):
        return None
    if len(re.findall(r"[.!?](?:\s|$)", summary)) > 2:
        return None
    return summary


def parse_rule_response_with_summary(
    text: Any, *, finish_reason: str | None = None
) -> tuple[str, str, str | None]:
    """Parse a sentence and return the safe ``reason`` or its legacy alias."""
    sentence, format_used = parse_rule_response(text, finish_reason=finish_reason)
    summary = None
    if isinstance(text, str):
        cleaned = _strip_fence(text.lstrip("\ufeff").strip())
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                summary = _safe_decision_summary(parsed.get("reason"))
                if summary is None:
                    summary = _safe_decision_summary(parsed.get("decision_summary"))
        except json.JSONDecodeError:
            pass
    return sentence, format_used, summary


def parse_rule_parts_with_summary(
    parts: list[Any], *, finish_reason: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[str, str, str | None]:
    """Parse final content and ignore thought-marked parts."""
    candidates = []
    for index, part in enumerate(parts):
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if not isinstance(text, str) or not text.strip():
            continue
        thought = bool(part.get("thought", False) if isinstance(part, dict) else getattr(part, "thought", False))
        candidates.append((index, text, thought))
    if str(finish_reason or "").upper() in {"MAX_TOKENS", "LENGTH"}:
        raise ModelOutputError("model output was truncated by the token limit", finish_reason=finish_reason)
    errors = []
    for index, text, thought in reversed(candidates):
        if thought:
            continue
        try:
            return parse_rule_response_with_summary(text, finish_reason=finish_reason)
        except ModelOutputError as exc:
            errors.append(f"part[{index}]: {exc.reason}")
    reason = "no valid final text part"
    if candidates and all(item[2] for item in candidates):
        reason = "only reasoning parts were returned"
    elif errors:
        reason = "all text parts failed rule-output validation"
    raise ModelOutputError(reason, finish_reason=finish_reason,
                           diagnostics={**(diagnostics or {}), "part_errors": errors})


def parse_correction_intent_response(parts: list[Any], *, finish_reason: str | None = None):
    """Parse the bounded CorrectionIntent contract, ignoring private thought parts."""
    if str(finish_reason or "").upper() in {"MAX_TOKENS", "LENGTH"}:
        raise ModelOutputError("model output was truncated by the token limit", finish_reason=finish_reason)
    candidates = []
    for part in parts:
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        thought = bool(part.get("thought", False) if isinstance(part, dict) else getattr(part, "thought", False))
        if isinstance(text, str) and text.strip() and not thought:
            candidates.append(text.strip())
    for text in reversed(candidates):
        try:
            value = json.loads(_strip_fence(text))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(value, dict) or not isinstance(value.get("sentence"), str):
            continue
        try:
            from .models import CorrectionIntent
            allowed = {"noop", "behavior", "label_policy", "transform_policy", "null_policy", "scope", "sentence"}
            value = {key: item for key, item in value.items() if key in allowed}
            return CorrectionIntent.model_validate(value), "correction_intent_json"
        except Exception:
            continue
    # Backward-compatible fake providers may still return sentence-only JSON.
    sentence, format_used = parse_rule_parts(candidates and [{"text": candidates[-1]}] or [], finish_reason=finish_reason)
    from .models import CorrectionIntent
    return CorrectionIntent(sentence=sentence), format_used
