from types import SimpleNamespace

from app.models import TokenUsage, UsageSummary
from app.usage import normalize_provider_usage, summarize_usage


def test_normalizes_native_oci_usage_details():
    response_usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        prompt_tokens_details=SimpleNamespace(cached_tokens=30),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=4),
    )
    usage = normalize_provider_usage(response_usage, model="gemini", call_type="extraction")
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.cached_tokens == 30
    assert usage.reasoning_tokens == 4
    assert usage.total_tokens == 120
    assert usage.reported is True


def test_normalizes_openai_compatible_usage_and_preserves_unknown_categories():
    response_usage = {
        "input_tokens": 80,
        "output_tokens": 12,
        "total_tokens": 92,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    usage = normalize_provider_usage(response_usage)
    summary = summarize_usage([
        usage,
        TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12, reported=True),
    ])
    assert summary.input_tokens == 90
    assert summary.output_tokens == 14
    assert summary.cached_tokens is None
    assert summary.reasoning_tokens is None
    assert summary.total_tokens == 104


def test_unknown_usage_is_counted_without_fabricating_tokens():
    summary = UsageSummary()
    summary.add_call(TokenUsage(call_type="reflection"))
    assert summary.calls == 1
    assert summary.unknown_calls == 1
    assert summary.input_tokens is None
    assert summary.total_tokens is None


def test_normalizer_preserves_explicit_zero_values():
    usage = normalize_provider_usage({
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    })
    assert usage.reported is True
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.cached_tokens == 0
    assert usage.reasoning_tokens == 0
    assert usage.total_tokens == 0


def test_normalizer_reads_top_level_reasoning_tokens():
    usage = normalize_provider_usage({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "reasoning_tokens": 7,
    })
    assert usage.reasoning_tokens == 7
    assert usage.reasoning_tokens_location == "reasoning_tokens"
    assert usage.reasoning_tokens_status == "reported"
    assert usage.usage_fields_present == ["prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"]
    assert usage.output_tokens_semantics == "provider_reported_output_total"


def test_normalizer_reports_provider_unavailable_reasoning_details():
    usage = normalize_provider_usage({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    })
    assert usage.reasoning_tokens is None
    assert usage.reasoning_tokens_reported is False
    assert usage.reasoning_tokens_status == "provider_unavailable"
    summary = summarize_usage([usage, usage])
    assert summary.reasoning_tokens is None
    assert summary.reasoning_tokens_status == "provider_unavailable"


def test_normalizer_reads_reasoning_tokens_from_nested_output_details():
    usage = normalize_provider_usage({
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "output_tokens_details": {"reasoning_tokens": 3},
    })
    assert usage.reasoning_tokens == 3
    assert usage.reasoning_tokens_location == "output_tokens_details.reasoning_tokens"


def test_gpt_oss_output_usage_is_explicitly_marked_as_may_including_reasoning():
    usage = normalize_provider_usage({
        "prompt_tokens": 3344,
        "completion_tokens": 4296,
        "total_tokens": 7640,
    }, model="openai.gpt-oss-20b")
    assert usage.output_tokens_semantics == "may_include_reasoning_tokens"
    assert usage.reasoning_tokens is None
    assert usage.reasoning_tokens_status == "provider_unavailable"


def test_invalid_reasoning_token_shape_is_not_treated_as_zero():
    usage = normalize_provider_usage({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "completion_tokens_details": {"reasoning_tokens": "unknown"},
    })
    assert usage.reasoning_tokens is None
    assert usage.reasoning_tokens_status == "unsupported_response_shape"


def test_normalizer_reads_sdk_to_dict_nested_reasoning_details():
    class UsageObject:
        def to_dict(self):
            return {
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28,
                "output_tokens_details": {"reasoning_tokens": 3},
            }

    usage = normalize_provider_usage(UsageObject())
    assert usage.input_tokens == 20
    assert usage.output_tokens == 8
    assert usage.reasoning_tokens == 3
    assert usage.reasoning_tokens_status == "reported"


def test_summary_preserves_unsupported_reasoning_shape_status():
    usage = normalize_provider_usage({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "reasoning_tokens": "not-a-number",
    })
    summary = summarize_usage([usage])
    assert summary.reasoning_tokens is None
    assert summary.reasoning_tokens_status == "unsupported_response_shape"


def test_normalizer_supports_camel_case_and_top_level_cached_tokens():
    usage = normalize_provider_usage({
        "promptTokens": 100,
        "completionTokens": 25,
        "cachedTokens": 8,
        "totalTokens": 125,
        "completionTokensDetails": {"reasoningTokens": 7},
    }, model="openai.gpt-oss-20b")
    assert usage.input_tokens == 100
    assert usage.output_tokens == 25
    assert usage.cached_tokens == 8
    assert usage.reasoning_tokens == 7
    assert usage.reasoning_tokens_location == "completion_tokens_details.reasoning_tokens"
    assert usage.total_tokens == 125


def test_normalizer_supports_camel_case_nested_cached_tokens():
    usage = normalize_provider_usage({
        "inputTokens": 100,
        "outputTokens": 25,
        "totalTokens": 125,
        "inputTokensDetails": {"cachedTokens": 8},
        "outputTokensDetails": {"reasoningTokens": 0},
    })
    assert usage.cached_tokens == 8
    assert usage.reasoning_tokens == 0
