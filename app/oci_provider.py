import json
import logging
import os
from pathlib import Path
from typing import Any

import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails, GenericChatRequest, Message, OnDemandServingMode, DedicatedServingMode,
    TextContent, JsonSchemaResponseFormat, ResponseJsonSchema,
)
import time
import random
import threading
from dotenv import load_dotenv
from .model_output import ModelOutputError, parse_correction_intent_response, parse_rule_parts_with_summary
from .prompt_builder import RulePromptBuilder
from .sentence_payload import build_sentence_payload
from .sentence_validators import assemble_local_sentence, validate_sentence
from .models import ProviderGenerationResult
from .model_registry import (
    REASONING_EFFORT_ENUMS,
    RuleGenerationConfigurationError,
    RULE_GENERATION_MODELS,
    normalize_reasoning_effort,
    reasoning_supported,
    rule_generation_settings,
)
from .usage import normalize_provider_usage, summarize_usage

load_dotenv()
logger = logging.getLogger(__name__)


class OciNativeRuleGenerator:
    def __init__(self, client: Any = None, model_id: str | None = None,
                 *, region: str | None = None, compartment_id: str | None = None,
                 serving_mode: str | None = None, endpoint_id: str | None = None,
                 reasoning_effort: str | None = None, verbosity: str | None = None):
        self.model_id = model_id or os.getenv("OCI_RULE_GENERATION_MODEL_ID", "google.gemini-2.5-flash")
        model = next((item for item in RULE_GENERATION_MODELS.values() if item.model_id == self.model_id), None)
        settings = rule_generation_settings(model) if model else {}
        self.region = region or settings.get("region") or "ap-hyderabad-1"
        self.endpoint_config = settings.get("endpoint")
        self.compartment_id = compartment_id or settings.get("compartment_id")
        if not self.compartment_id:
            raise RuntimeError("A compartment for the selected sentence-generation model must be configured")
        self.serving_mode = (serving_mode or settings.get("serving_mode") or "on_demand").lower()
        self.endpoint_id = endpoint_id or settings.get("endpoint_id")
        self.reasoning_effort_requested = normalize_reasoning_effort(reasoning_effort)
        self.reasoning_supported = reasoning_supported(self.model_id)
        self.reasoning_effort_effective = (
            REASONING_EFFORT_ENUMS[self.reasoning_effort_requested]
            if self.reasoning_supported else None
        )
        # Kept for compatibility/observability only.  The active OCI Chat
        # endpoint rejects this field with a type-deserialization error, so
        # it is intentionally not placed on GenericChatRequest.
        self.verbosity = (verbosity or os.getenv("OCI_RULE_GENERATION_VERBOSITY", "low")).strip().upper()
        self._external_client = client is not None
        self._call_count = 0
        self._call_count_lock = threading.Lock()
        self._validate_serving_mode()
        # Output length is managed by OCI/model defaults. We only bound the
        # number of repair requests; no application token budget is sent.
        self.model_output_retries = 0
        if client:
            self.client = client
        else:
            signer = None
            auth = os.getenv("OCI_AUTH", "config").lower()
            if auth == "instance_principal":
                signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
                self.client = GenerativeAiInferenceClient({}, signer=signer, service_endpoint=self.endpoint)
            elif auth == "resource_principal":
                signer = oci.auth.signers.get_resource_principals_signer()
                self.client = GenerativeAiInferenceClient({}, signer=signer, service_endpoint=self.endpoint)
            else:
                config_path = os.getenv("OCI_CONFIG_FILE", ".oci/config")
                config = oci.config.from_file(config_path, os.getenv("OCI_PROFILE", "DEFAULT"))
                # Keep request signing aligned with the selected model's
                # endpoint even when the local profile has a stale region.
                config["region"] = self.region
                self.client = GenerativeAiInferenceClient(config, service_endpoint=self.endpoint)

    @property
    def endpoint(self) -> str:
        endpoint = self.endpoint_config or f"https://inference.generativeai.{self.region}.oci.oraclecloud.com"
        return endpoint.rstrip("/").removesuffix("/openai/v1")

    def _validate_serving_mode(self):
        if self.serving_mode not in {"on_demand", "dedicated"}:
            raise RuntimeError("OCI_RULE_GENERATION_SERVING_MODE must be 'on_demand' or 'dedicated'")
        if self.serving_mode == "dedicated" and not self.endpoint_id and not self._external_client:
            raise RuleGenerationConfigurationError(
                "MODEL_ENDPOINT_CONFIGURATION_ERROR",
                f"{self.model_id} sentence generation requires a dedicated endpoint ID",
                model=self.model_id,
                region=self.region,
            )

    def _serving_mode(self):
        # Injected clients are used by deterministic tests and callers that
        # already own transport configuration; do not require an endpoint
        # merely to construct their request object.
        if self.serving_mode == "dedicated" and self.endpoint_id:
            return DedicatedServingMode(endpoint_id=self.endpoint_id)
        return OnDemandServingMode(model_id=self.model_id)

    def generate_with_metadata(self, context=None, **legacy_kwargs) -> ProviderGenerationResult:
        """Generate a rule from the full context, retaining legacy callers."""
        from .models import RuleGenerationContext
        if context is None:
            context = RuleGenerationContext(
                field_key=legacy_kwargs["field_key"],
                display_label=legacy_kwargs.get("display_label", ""),
                short_rule=legacy_kwargs.get("short_rule", ""),
                detailed_rule=legacy_kwargs.get("detailed_rule", []),
                field_path=legacy_kwargs.get("field_path") or legacy_kwargs["field_key"],
                old_value=legacy_kwargs.get("old_value"),
                new_value=legacy_kwargs.get("new_value"),
                invoice_payload=legacy_kwargs.get("invoice_payload") or {},
                final_response=legacy_kwargs.get("final_response") or {},
                historical_examples=legacy_kwargs.get("historical_examples") or [],
            )
        prompt = RulePromptBuilder.normal_payload(context)
        system = _load_prompt("sentence_developer.txt", (
            "Reasoning: low. Generate a correction intent for one invoice field. Return exactly one JSON object "
            "with noop, behavior, label_policy, transform_policy, null_policy, scope, and one reusable "
            "imperative sentence. Do not repeat invoice values, add fallback hops, mention other fields, or "
            "reveal private reasoning. No Markdown or explanations."
        ))
        schema = {"type": "object", "properties": {
                      "noop": {"type": "boolean"}, "behavior": {"type": "string"},
                      "label_policy": {"type": "string"}, "transform_policy": {"type": "string"},
                      "null_policy": {"type": "string"}, "scope": {"type": "string"},
                      "sentence": {"type": "string"}},
                  "required": ["noop", "behavior", "label_policy", "transform_policy", "null_policy", "scope", "sentence"] if self.reasoning_supported else ["sentence"],
                  "additionalProperties": False}
        if not self.reasoning_supported:
            schema = {"type": "object", "properties": {"sentence": {"type": "string"}},
                      "required": ["sentence"], "additionalProperties": False}
        response_format = JsonSchemaResponseFormat(
            json_schema=ResponseJsonSchema(
                name="extraction_rule_sentence",
                description="One reusable extraction-rule sentence.",
                schema=schema,
                is_strict=True,
            )
        )
        usage_records = []
        generation_finish_reasons = []
        generation_request_ids = []
        validation_errors = []

        def annotate_output_error(error: ModelOutputError) -> ModelOutputError:
            """Attach bounded, non-sensitive retry diagnostics to the error."""
            error.diagnostics = {
                **(error.diagnostics or {}),
                "generation": {
                    "attempts": len(usage_records),
                    "application_output_limit_sent": False,
                    "provider_managed_limit": True,
                    "finish_reasons": generation_finish_reasons,
                    "request_ids": generation_request_ids,
                    "validation_errors": validation_errors,
                    "reason": "provider_output_truncated_after_repair"
                    if str(error.finish_reason or "").upper() in {"MAX_TOKENS", "LENGTH"}
                    else "output_validation_failed_after_repair",
                },
            }
            return error

        for repair_attempt in range(self.model_output_retries + 1):
            retry_system = system
            user_payload = prompt
            if repair_attempt:
                retry_system += " Your previous response was incomplete. Output the JSON object immediately, with no preamble."
                # Do not resend the full evidence/history payload during repair;
                # it consumes context while the requested output is one sentence.
                user_payload = {
                    "field_key": context.field_key,
                    "existing_rule": context.short_rule,
                    "correction_kind": build_sentence_payload(context)["correction_kind"],
                    "instruction": "Return exactly one reusable sentence. Do not repeat any literal value from the prior request.",
                }
            request = GenericChatRequest(
                api_format="GENERIC", messages=[
                    Message(role="SYSTEM", content=[TextContent(text=retry_system)]),
                    Message(role="USER", content=[TextContent(text=json.dumps(user_payload, ensure_ascii=False))]),
                ], response_format=response_format, temperature=0.0,
                **self._reasoning_request_kwargs(),
            )
            details = ChatDetails(compartment_id=self.compartment_id,
                                  serving_mode=self._serving_mode(),
                                  chat_request=request)
            logger.info(
                "OCI rule-generation request model=%s reasoning_effort=%s verbosity_sent=false",
                self.model_id,
                self.reasoning_effort_effective if self.reasoning_supported else None,
            )
            response = self._chat_with_retry(details)
            request_id = getattr(response, "headers", {}).get("opc-request-id") if hasattr(response, "headers") else None
            generation_request_ids.append(request_id)
            data = response.data
            chat_response = _field(data, "chat_response") or data
            usage_location = None
            usage_payload = _field(chat_response, "usage")
            if usage_payload is not None:
                usage_location = "chat_response.usage"
            else:
                usage_payload = _field(data, "usage")
                if usage_payload is not None:
                    usage_location = "data.usage"
                else:
                    usage_payload = _field(response, "usage")
                    if usage_payload is not None:
                        usage_location = "response.usage"
            usage_records.append(normalize_provider_usage(
                usage_payload, model=self.model_id, request_id=request_id,
                call_type="rule_generation", attempt=repair_attempt + 1,
                usage_location=usage_location,
            ))
            latest_usage = usage_records[-1]
            choices = getattr(chat_response, "choices", None) or []
            finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
            generation_finish_reasons.append(finish_reason)
            if str(finish_reason or "").upper() == "SAFETY":
                raise ModelOutputError("model output was blocked by safety policy", finish_reason=finish_reason)
            text = getattr(chat_response, "text", None)
            if choices:
                message = getattr(choices[0], "message", None)
                parts = getattr(message, "content", []) if message else []
                if parts:
                    try:
                        if self.reasoning_supported:
                            intent, format_used = parse_correction_intent_response(parts, finish_reason=finish_reason)
                            reason = None
                            # Compatibility only: older fake providers may
                            # still return decision_summary/reason. The active
                            # GPT-OSS schema does not request these fields.
                            try:
                                _, _, reason = parse_rule_parts_with_summary(parts, finish_reason=finish_reason)
                            except ModelOutputError:
                                pass
                        else:
                            sentence, format_used, reason = parse_rule_parts_with_summary(parts, finish_reason=finish_reason)
                            from .models import CorrectionIntent
                            intent = CorrectionIntent(sentence=sentence)
                        sentence = intent.sentence
                        try:
                            sentence = validate_sentence(sentence, user_payload)
                        except ValueError as exc:
                            # The model is an intent resolver, not an
                            # unrestricted prose author. Recover locally from
                            # the validated delta without another OCI call.
                            sentence = validate_sentence(assemble_local_sentence(user_payload), user_payload)
                            validation_errors.append(str(exc)[:160])
                        reason = self._validated_reason(reason, context)
                        return ProviderGenerationResult(
                            sentence=sentence, intent=intent, noop=intent.noop, correction_kind=user_payload.get("correction_kind") if isinstance(user_payload, dict) else None,
                            request_id=request_id, response_format=format_used,
                            usage=summarize_usage(usage_records), model=self.model_id,
                            finish_reason=finish_reason, attempts=len(usage_records),
                            usage_location=usage_location,
                            reasoning_effort_requested=self.reasoning_effort_requested,
                            reasoning_effort_effective=self.reasoning_effort_effective,
                            reasoning_supported=self.reasoning_supported,
                            visible_reasoning=False,
                            decision_summary=reason, reason=reason,
                            reasoning_summary_available=bool(reason),
                            reasoning_mode="correction_intent" if self.reasoning_supported else "not_available",
                            reasoning_parameter_sent=self.reasoning_supported,
                            verbosity_parameter_sent=False,
                            usage_diagnostics={
                                "normalizer_version": "provider-usage-v3",
                                "usage_location": usage_location,
                                "reasoning_tokens_location": latest_usage.reasoning_tokens_location,
                                "reasoning_tokens_reported": latest_usage.reasoning_tokens_reported,
                                "reasoning_tokens_status": latest_usage.reasoning_tokens_status,
                                "usage_fields_present": latest_usage.usage_fields_present,
                                "nested_detail_fields_present": latest_usage.nested_detail_fields_present,
                                "output_tokens_semantics": latest_usage.output_tokens_semantics,
                                "application_output_limit_sent": False,
                                "provider_managed_output_limit": True,
                                "max_tokens_sent": False,
                            },
                        )
                    except ModelOutputError as exc:
                        if exc.diagnostics.get("validation_error"):
                            validation_errors.append(str(exc.diagnostics["validation_error"])[:160])
                        if repair_attempt < self.model_output_retries and _retryable_output_error(exc):
                            continue
                        raise annotate_output_error(exc)
            if text:
                try:
                    if self.reasoning_supported:
                        intent, format_used = parse_correction_intent_response([{"text": text}], finish_reason=finish_reason)
                        sentence = intent.sentence
                        reason = None
                        try:
                            _, _, reason = parse_rule_parts_with_summary([{"text": text}], finish_reason=finish_reason)
                        except ModelOutputError:
                            pass
                    else:
                        sentence, format_used, reason = parse_rule_parts_with_summary([{"text": text}], finish_reason=finish_reason)
                        from .models import CorrectionIntent
                        intent = CorrectionIntent(sentence=sentence)
                    try:
                        sentence = validate_sentence(sentence, user_payload)
                    except ValueError as exc:
                        sentence = validate_sentence(assemble_local_sentence(user_payload), user_payload)
                        validation_errors.append(str(exc)[:160])
                    reason = self._validated_reason(reason, context)
                    return ProviderGenerationResult(
                        sentence=sentence, intent=intent, noop=intent.noop, correction_kind=build_sentence_payload(context)["correction_kind"], request_id=request_id, response_format=format_used,
                        usage=summarize_usage(usage_records), model=self.model_id,
                        finish_reason=finish_reason, attempts=len(usage_records),
                        usage_location=usage_location,
                        reasoning_effort_requested=self.reasoning_effort_requested,
                        reasoning_effort_effective=self.reasoning_effort_effective,
                        reasoning_supported=self.reasoning_supported,
                        visible_reasoning=False,
                        decision_summary=reason, reason=reason,
                        reasoning_summary_available=bool(reason),
                        reasoning_mode="correction_intent" if self.reasoning_supported else "not_available",
                        reasoning_parameter_sent=self.reasoning_supported,
                        verbosity_parameter_sent=False,
                        usage_diagnostics={
                            "normalizer_version": "provider-usage-v3",
                            "usage_location": usage_location,
                            "reasoning_tokens_location": latest_usage.reasoning_tokens_location,
                            "reasoning_tokens_reported": latest_usage.reasoning_tokens_reported,
                            "reasoning_tokens_status": latest_usage.reasoning_tokens_status,
                            "usage_fields_present": latest_usage.usage_fields_present,
                            "nested_detail_fields_present": latest_usage.nested_detail_fields_present,
                            "output_tokens_semantics": latest_usage.output_tokens_semantics,
                            "application_output_limit_sent": False,
                            "provider_managed_output_limit": True,
                            "max_tokens_sent": False,
                        },
                    )
                except ModelOutputError as exc:
                    if exc.diagnostics.get("validation_error"):
                        validation_errors.append(str(exc.diagnostics["validation_error"])[:160])
                    if repair_attempt < self.model_output_retries and _retryable_output_error(exc):
                        continue
                    raise annotate_output_error(exc)
            if repair_attempt >= self.model_output_retries:
                raise annotate_output_error(ModelOutputError(
                    f"OCI returned no text (finish_reason={finish_reason!r})",
                    finish_reason=finish_reason,
                ))
        raise annotate_output_error(ModelOutputError("OCI output repair attempts exhausted"))

    @staticmethod
    def _fallback_decision_summary(context) -> str:
        packet = getattr(context, "feedback_packet", None)
        label = getattr(context, "display_label", "") or getattr(context, "field_key", "field")
        if packet:
            for evidence in packet.evidence:
                if evidence.label:
                    label = evidence.label
                    break
        summary = f"Selected evidence associated with the explicit {label} label and applied the active correction policy."
        if packet and packet.competing_evidence:
            summary += " Unrelated competing occurrences were ignored."
        return summary

    @classmethod
    def _validated_reason(cls, reason, context) -> str:
        """Keep explanations safe and deterministic without exposing values."""
        if isinstance(reason, str) and reason.strip():
            lowered = reason.casefold()
            for value in _scalar_values(getattr(context, "old_value", None)) + _scalar_values(getattr(context, "new_value", None)):
                if len(value) >= 3 and value.casefold() in lowered:
                    return cls._fallback_decision_summary(context)
            return reason.strip()
        return cls._fallback_decision_summary(context)

    def _reasoning_request_kwargs(self) -> dict[str, str]:
        """Return only parameters supported by the active sentence model.

        GPT-OSS supports OCI reasoning controls.  The other sentence models
        remain on the common prompt contract but must not receive GPT-OSS-only
        request fields.
        """
        if not self.reasoning_supported:
            return {}
        return {"reasoning_effort": self.reasoning_effort_effective}

    def generate(self, context=None, **legacy_kwargs) -> tuple[str, str | None, str]:
        result = self.generate_with_metadata(context=context, **legacy_kwargs)
        return result.sentence, result.request_id, result.response_format or "plain_sentence"

    def _chat_with_retry(self, details):
        for attempt in range(3):
            try:
                with self._call_count_lock:
                    self._call_count += 1
                return self.client.chat(details)
            except Exception as exc:
                status = getattr(exc, "status", None)
                if status not in {408, 429, 500, 502, 503} or attempt == 2:
                    raise
                time.sleep((2 ** attempt) + random.uniform(0, 0.25))

    @property
    def chat_call_count(self) -> int:
        with self._call_count_lock:
            return self._call_count


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return value if value > 0 else default
    except ValueError:
        return default


def _field(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _scalar_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, dict):
        values = []
        for item in value.values():
            values.extend(_scalar_values(item))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_scalar_values(item))
        return values
    return []


def _retryable_output_error(error: ModelOutputError) -> bool:
    finish = str(error.finish_reason or "").upper()
    if finish in {"MAX_TOKENS", "LENGTH"}:
        return True
    prefix = error.prefix.lower().strip()
    return prefix.startswith("here is the json") or "validation_error" in (error.diagnostics or {})


def _load_prompt(filename: str, fallback: str) -> str:
    """Load a checked-in prompt contract without making the process cwd significant."""
    path = Path(__file__).resolve().parent.parent / "prompts" / filename
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return fallback
    return text or fallback


# Backward-compatible name retained for existing tests and callers.
OciGeminiRuleGenerator = OciNativeRuleGenerator
