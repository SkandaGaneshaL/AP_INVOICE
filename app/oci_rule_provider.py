from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from .model_output import ModelOutputError, parse_rule_response
from .models import ProviderGenerationResult, RuleGenerationContext
from .oci_pdf_client import OciPdfClient
from .prompt_builder import RulePromptBuilder
from .usage import normalize_provider_usage, summarize_usage


class OciOpenAICompatibleRuleGenerator:
    """Text-only rule writer for OCI OpenAI-compatible models."""

    def __init__(self, client: Any = None, *, model_id: str = "openai.gpt-oss-20b"):
        self.model_id = model_id
        # Retained as a compatibility provider, but application-level output
        # limits are deliberately not sent to OCI. Only repair-request count
        # remains bounded operational configuration.
        self.output_retries = min(max(0, _int_env("OCI_MODEL_OUTPUT_REPAIR_ATTEMPTS", 1)), 1)
        self.client = client or OciPdfClient(model_id=model_id).client

    def generate_with_metadata(self, context: RuleGenerationContext, **_: Any) -> ProviderGenerationResult:
        prompt = RulePromptBuilder.normal_payload(context)
        system = (
            "You write invoice extraction rules. Return exactly one concise imperative sentence. "
            "Describe reusable extraction behavior and never hard-code the current invoice value. "
            "Return only the sentence, with no JSON, Markdown, explanation, reasoning, or preamble."
        )
        usages = []
        request_id = None
        finish_reason = None
        for attempt in range(self.output_retries + 1):
            user_payload = prompt if attempt == 0 else {
                "field_key": context.field_key,
                "existing_rule": context.short_rule,
                "correction": f"{context.old_value} -> {context.new_value}",
                "instruction": "Return exactly one concise reusable extraction-rule sentence.",
            }
            request = {
                "model": self.model_id,
                "store": False,
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": system}]},
                    {"role": "user", "content": [{"type": "input_text", "text": json.dumps(user_payload, ensure_ascii=False)}]},
                ],
                "temperature": 0.0,
            }
            try:
                response = self.client.responses.create(**request)
                request_id = _request_id(response)
                finish_reason = _get(response, "finish_reason")
                usages.append(normalize_provider_usage(
                    _get(response, "usage"), model=self.model_id,
                    request_id=request_id, call_type="rule_generation", attempt=attempt + 1,
                ))
                text = _output_text(response)
                sentence, response_format = parse_rule_response(text, finish_reason=finish_reason)
                return ProviderGenerationResult(
                    sentence=sentence, request_id=request_id, response_format=response_format,
                    usage=summarize_usage(usages), model=self.model_id,
                    finish_reason=finish_reason, attempts=len(usages),
                    usage_location="response.usage" if _get(response, "usage") is not None else None,
                )
            except ModelOutputError:
                if attempt >= self.output_retries:
                    raise
            except Exception as exc:
                if attempt >= self.output_retries or not _retryable(exc):
                    raise
                time.sleep((2**attempt) + random.uniform(0, 0.25))
        raise ModelOutputError("OCI rule-generation retries exhausted")

    def generate(self, context=None, **legacy_kwargs):
        if context is None:
            context = RuleGenerationContext.model_validate(legacy_kwargs)
        result = self.generate_with_metadata(context=context)
        return result.sentence, result.request_id, result.response_format or "plain_sentence"


def _output_text(response: Any) -> str:
    value = _get(response, "output_text")
    if value:
        return str(value).strip()
    output = _get(response, "output") or []
    parts = []
    for item in output:
        for content in (_get(item, "content") or []):
            text = _get(content, "text")
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def _request_id(response: Any) -> str | None:
    headers = _get(response, "headers")
    if isinstance(headers, dict):
        return headers.get("opc-request-id") or headers.get("x-request-id")
    return _get(response, "request_id")


def _get(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return value if value > 0 else default
    except ValueError:
        return default


def _retryable(exc: Exception) -> bool:
    return getattr(exc, "status", None) in {408, 429, 500, 502, 503}
