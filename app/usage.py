from __future__ import annotations

from typing import Any, Iterable

from .models import TokenUsage, UsageSummary

USAGE_NORMALIZER_VERSION = "provider-usage-v3"


def _get(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    result = getattr(value, name, None)
    if result is not None:
        return result
    # Some OCI SDK model versions expose a safe to_dict() representation
    # rather than all nested fields as normal attributes.  Read only the
    # requested allowlisted field; never retain the full serialized object.
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            serialized = to_dict()
        except Exception:
            return None
        return serialized.get(name) if isinstance(serialized, dict) else None
    return None


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _token_value(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _present_fields(value: Any, allowed: tuple[str, ...]) -> list[str]:
    return [name for name in allowed if _get(value, name) is not None]


def normalize_provider_usage(usage: Any, *, model: str | None = None,
                              request_id: str | None = None, call_type: str | None = None,
                              attempt: int = 1, usage_location: str | None = None) -> TokenUsage:
    """Normalize OCI-native and OpenAI-compatible usage shapes without guessing."""
    input_tokens = _token_value(first_not_none(_get(usage, "prompt_tokens"), _get(usage, "input_tokens")))
    output_tokens = _token_value(first_not_none(_get(usage, "completion_tokens"), _get(usage, "output_tokens")))
    prompt_details = _get(usage, "prompt_tokens_details")
    prompt_detail_name = "prompt_tokens_details"
    if prompt_details is None:
        prompt_details = _get(usage, "input_tokens_details")
        prompt_detail_name = "input_tokens_details"
    completion_details = _get(usage, "completion_tokens_details")
    completion_detail_name = "completion_tokens_details"
    if completion_details is None:
        completion_details = _get(usage, "output_tokens_details")
        completion_detail_name = "output_tokens_details"
    cached_tokens = _token_value(first_not_none(_get(prompt_details, "cached_tokens"), _get(prompt_details, "cache_read_input_tokens")))
    reasoning_sources = (
        ("reasoning_tokens", _get(usage, "reasoning_tokens")),
        (f"{completion_detail_name}.reasoning_tokens", _get(completion_details, "reasoning_tokens")),
        ("completion_tokens.reasoning_tokens", _get(_get(usage, "completion_tokens"), "reasoning_tokens")),
        ("output_tokens.reasoning_tokens", _get(_get(usage, "output_tokens"), "reasoning_tokens")),
    )
    reasoning_tokens_location = None
    reasoning_tokens = None
    reasoning_field_present = False
    for location, value in reasoning_sources:
        reasoning_field_present = reasoning_field_present or value is not None
        numeric = _token_value(value)
        if numeric is not None:
            reasoning_tokens_location = location
            reasoning_tokens = numeric
            break
    total_tokens = _token_value(_get(usage, "total_tokens"))
    usage_fields = _present_fields(usage, (
        "prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "total_tokens", "reasoning_tokens",
        "prompt_tokens_details", "completion_tokens_details", "input_tokens_details", "output_tokens_details",
    ))
    nested_details = _present_fields(prompt_details, ("cached_tokens", "cache_read_input_tokens"))
    nested_details += _present_fields(completion_details, ("reasoning_tokens",))
    nested_details += _present_fields(_get(usage, "completion_tokens"), ("reasoning_tokens",))
    nested_details += _present_fields(_get(usage, "output_tokens"), ("reasoning_tokens",))
    values = (
        input_tokens, output_tokens, total_tokens, cached_tokens, reasoning_tokens,
    )
    if reasoning_tokens is not None:
        reasoning_status = "reported"
    elif reasoning_field_present:
        reasoning_status = "unsupported_response_shape"
    else:
        reasoning_status = "provider_unavailable"
    output_semantics = (
        "may_include_reasoning_tokens"
        if model and "gpt-oss" in model.lower()
        else "provider_reported_output_total"
    )
    return TokenUsage(
        input_tokens=input_tokens, output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        reported=usage is not None and any(value is not None for value in values),
        model=model, request_id=request_id, call_type=call_type, attempt=attempt,
        usage_location=usage_location,
        reasoning_tokens_location=reasoning_tokens_location,
        reasoning_tokens_reported=reasoning_tokens is not None,
        reasoning_tokens_status=reasoning_status,
        usage_fields_present=usage_fields,
        nested_detail_fields_present=nested_details,
        output_tokens_semantics=output_semantics,
    )


def summarize_usage(usages: Iterable[TokenUsage | UsageSummary | None]) -> UsageSummary:
    result = UsageSummary()
    for usage in usages:
        if usage is not None:
            result.add_summary(usage) if isinstance(usage, UsageSummary) else result.add_call(usage)
    return result
