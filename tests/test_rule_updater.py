from app.models import UpdateRequest
from app.service import UpdateRulesService
from app.rule_repository import RuleRepository
from app.audit import AuditRepository
import json
from types import SimpleNamespace
from app.oci_provider import OciGeminiRuleGenerator, OciNativeRuleGenerator
from app.model_output import (
    ModelOutputError,
    parse_rule_response,
    parse_rule_parts,
    parse_rule_response_with_summary,
    parse_rule_parts_with_summary,
)
from app.models import EvidenceMatch, ProviderGenerationResult, RuleFeedbackPacket, RuleGenerationContext


class FakeGenerator:
    def generate(self, **kwargs):
        return "When the tax field contains a rate, extract only an explicitly labeled tax amount.", "req-1"


def request(final="9.5%"):
    return UpdateRequest(invoice_payload={"TaxAmount": {"value": "133.00", "Page": "1"}},
                         final_response={"TaxAmount": {"value": final, "Page": "1"}},
                         allow_partial=False)


def isolated_repository(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps([{"ID": 46, "FIELD_KEY": "TaxAmount", "DISPLAY_LABEL": "Tax",
        "SHORT_RULE": "tax", "DETAILED_RULE": ["Extract labeled tax.", "Do not infer tax."]}]))
    return RuleRepository(str(path))


def test_changed_field_appends_sentence(tmp_path):
    repository = isolated_repository(tmp_path)
    before = repository.path.read_text()
    result = UpdateRulesService(FakeGenerator(), repository, AuditRepository(str(tmp_path / "audit.jsonl"))).update(request())
    assert result.summary["updated_rules"] == 0
    assert result.summary["preview_candidates"] == 1
    assert result.strategies[0].metadata["persistence_status"] == "awaiting_approval"
    assert len(result.updated_rules[0].DETAILED_RULE) == 3
    assert repository.path.read_text() == before


def test_business_reason_propagates_to_change_and_audit(tmp_path):
    class MetadataGenerator:
        def generate_with_metadata(self, context):
            return ProviderGenerationResult(
                sentence="Use the explicitly labeled tax amount.",
                reason="The sentence follows the current rule and prioritizes evidence associated with the explicit label.",
                decision_summary="Legacy alias.",
            )

    audit_path = tmp_path / "audit.jsonl"
    result = UpdateRulesService(
        MetadataGenerator(), isolated_repository(tmp_path), AuditRepository(str(audit_path))
    ).update(request())
    assert result.changes[0].reason.startswith("The sentence follows")
    record = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["metadata"]["reason"].startswith("The sentence follows")


def test_same_change_is_deduplicated(tmp_path):
    req = request()
    service = UpdateRulesService(FakeGenerator(), isolated_repository(tmp_path), AuditRepository(str(tmp_path / "audit.jsonl")))
    first = service.update(req)
    second = service.update(req)
    assert second.changes[0].status == "preview"


def test_page_only_change_is_ignored(tmp_path):
    req = request()
    req.final_response["TaxAmount"]["value"] = "133.00"
    req.final_response["TaxAmount"]["Page"] = "2"
    result = UpdateRulesService(FakeGenerator(), isolated_repository(tmp_path), AuditRepository(str(tmp_path / "audit.jsonl"))).update(req)
    assert result.changes == []


def test_unmapped_change_does_not_create_rule(tmp_path):
    req = request()
    req.final_response["Other"] = {"value": "x"}
    result = UpdateRulesService(FakeGenerator(), isolated_repository(tmp_path), AuditRepository(str(tmp_path / "audit.jsonl"))).update(req)
    assert any(x.status == "unmapped" for x in result.changes)


def test_oci_chat_result_nested_response_is_extracted(monkeypatch):
    message = SimpleNamespace(content=[SimpleNamespace(text='{"sentence":"Use the labeled tax amount."}')])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    data = SimpleNamespace(chat_response=SimpleNamespace(choices=[choice]))
    response = SimpleNamespace(data=data, headers={"opc-request-id": "req-test"})
    class Client:
        def chat(self, details):
            return response
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    result = OciGeminiRuleGenerator(client=Client()).generate(
        field_key="TaxAmount", display_label="Tax", short_rule="tax",
        detailed_rule=[], old_value="1", new_value="2")
    assert result == ("Use the labeled tax amount.", "req-test", "structured_json")


def test_oci_context_generation_preserves_feedback_without_unsupported_keywords(monkeypatch):
    captured = {}
    message = SimpleNamespace(content=[SimpleNamespace(text='{"sentence":"Use the explicit currency label."}')])
    response = SimpleNamespace(
        data=SimpleNamespace(chat_response=SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="STOP")])),
        headers={"opc-request-id": "req-context"},
    )

    class Client:
        def chat(self, details):
            captured["prompt"] = details.chat_request.messages[1].content[0].text
            return response

    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    context = RuleGenerationContext(
        field_key="InvoiceCurrency", display_label="Invoice Currency", short_rule="Extract currency.",
        old_value="USD", new_value="INR", field_path="InvoiceCurrency", rule_version="v2",
        document_id="doc-1", document_bytes=b"%PDF private bytes",
        feedback_packet=RuleFeedbackPacket(
            field_key="InvoiceCurrency", field_path="InvoiceCurrency", previous_value="USD", corrected_value="INR",
            evidence=[EvidenceMatch(page=1, label="Currency", value="INR", snippet="Currency: INR")],
            competing_evidence=[EvidenceMatch(page=1, value="USD", snippet="Reference amount in USD")],
            inferred_intent="Use the explicit currency label.",
            constraints=["Do not hard-code INR."],
        ),
    )
    result = OciGeminiRuleGenerator(client=Client()).generate(context=context)
    assert result[0] == "Use the explicit currency label."
    assert '"positive_labels": ["Currency"]' in captured["prompt"]
    assert '"field_key": "InvoiceCurrency"' in captured["prompt"]
    assert "Reference amount in USD" not in captured["prompt"]
    assert "%PDF private bytes" not in captured["prompt"]


def test_oci_rule_generation_reads_nested_chat_response_usage(monkeypatch):
    message = SimpleNamespace(content=[SimpleNamespace(text='{"sentence":"Use the labeled tax amount."}')])
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
    chat_response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="STOP")],
        usage=usage,
    )
    response = SimpleNamespace(data=SimpleNamespace(chat_response=chat_response), headers={"opc-request-id": "req-usage"})

    class Client:
        def chat(self, details):
            return response

    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    result = OciGeminiRuleGenerator(client=Client()).generate_with_metadata(
        field_key="TaxAmount", display_label="Tax", short_rule="tax", detailed_rule=[], old_value="1", new_value="2"
    )
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20
    assert result.usage.total_tokens == 120
    assert result.usage.by_call_type["rule_generation"].calls == 1
    assert result.usage_location == "chat_response.usage"


def test_oci_rule_generation_reads_direct_response_usage_dict(monkeypatch):
    response = SimpleNamespace(
        data=SimpleNamespace(chat_response=SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=[SimpleNamespace(text='{"sentence":"Use the labeled tax amount."}')]),
                finish_reason="STOP",
            )],
        )),
        usage={"input_tokens": 80, "output_tokens": 12, "total_tokens": 92},
        headers={"opc-request-id": "req-direct-usage"},
    )

    class Client:
        def chat(self, details):
            return response

    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    result = OciGeminiRuleGenerator(client=Client()).generate_with_metadata(
        field_key="TaxAmount", display_label="Tax", short_rule="tax", detailed_rule=[], old_value="1", new_value="2"
    )
    assert result.usage.input_tokens == 80
    assert result.usage.output_tokens == 12
    assert result.usage.total_tokens == 92
    assert result.usage_location == "response.usage"


def test_parser_uses_last_non_thought_part():
    from types import SimpleNamespace
    parts = [SimpleNamespace(text="internal reasoning", thought=True),
             SimpleNamespace(text='{"sentence":"Use the final labeled amount."}', thought=False)]
    assert parse_rule_parts(parts) == ("Use the final labeled amount.", "structured_json")


def test_parser_uses_last_valid_plain_part():
    from types import SimpleNamespace
    parts = [SimpleNamespace(text="invalid commentary", thought=False),
             SimpleNamespace(text="Use the final labeled amount.", thought=False)]
    assert parse_rule_parts(parts) == ("Use the final labeled amount.", "plain_sentence")


def test_parser_extracts_json_from_combined_reasoning_part():
    text = 'Internal reasoning omitted. Final answer: {"sentence":"Use the final labeled amount."}'
    assert parse_rule_response(text) == ("Use the final labeled amount.", "structured_json")


def test_parser_returns_safe_decision_summary():
    sentence, format_used, summary = parse_rule_response_with_summary(
        '{"sentence":"Use the final labeled amount.",'
        '"decision_summary":"Selected the explicit label and ignored unrelated evidence."}'
    )
    assert (sentence, format_used) == ("Use the final labeled amount.", "structured_json")
    assert summary == "Selected the explicit label and ignored unrelated evidence."


def test_parser_prefers_reason_over_legacy_decision_summary():
    sentence, _, reason = parse_rule_response_with_summary(
        '{"sentence":"Use the labeled amount.",'
        '"reason":"The corrected behavior follows the explicit label.",'
        '"decision_summary":"Legacy explanation."}'
    )
    assert sentence == "Use the labeled amount."
    assert reason == "The corrected behavior follows the explicit label."


def test_parser_keeps_sentence_only_response_backward_compatible():
    assert parse_rule_response_with_summary(
        '{"sentence":"Use the final labeled amount."}'
    ) == ("Use the final labeled amount.", "structured_json", None)


def test_parser_discards_private_reasoning_from_decision_summary():
    sentence, _, summary = parse_rule_response_with_summary(
        '{"sentence":"Use the final labeled amount.",'
        '"decision_summary":"This contains private chain-of-thought."}'
    )
    assert sentence == "Use the final labeled amount."
    assert summary is None


def test_parser_summary_ignores_thought_parts():
    parts = [
        SimpleNamespace(text="private chain-of-thought", thought=True),
        SimpleNamespace(
            text='{"sentence":"Use the final labeled amount.",'
                 '"decision_summary":"Selected the explicit label."}',
            thought=False,
        ),
    ]
    assert parse_rule_parts_with_summary(parts) == (
        "Use the final labeled amount.", "structured_json", "Selected the explicit label."
    )


def test_oci_rule_generation_propagates_decision_summary(monkeypatch):
    message = SimpleNamespace(content=[SimpleNamespace(
        text='{"sentence":"Use the labeled tax amount.",'
             '"decision_summary":"Selected the explicit tax label."}'
    )])
    response = SimpleNamespace(
        data=SimpleNamespace(chat_response=SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="STOP")]
        )),
        headers={"opc-request-id": "req-summary"},
    )

    class Client:
        def chat(self, details):
            request = details.chat_request
            assert list(request.response_format.json_schema.schema["properties"]) == ["sentence"]
            return response

    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    result = OciGeminiRuleGenerator(client=Client()).generate_with_metadata(
        field_key="TaxAmount", display_label="Tax", short_rule="tax",
        detailed_rule=[], old_value="1", new_value="2"
    )
    assert result.decision_summary == "Selected the explicit tax label."
    assert result.reason == "Selected the explicit tax label."
    assert result.reasoning_summary_available is True
    assert result.reasoning_mode == "safe_decision_summary"


def test_oci_rule_generation_propagates_reason_and_falls_back_safely(monkeypatch):
    message = SimpleNamespace(content=[SimpleNamespace(
        text='{"sentence":"Use the labeled PO number.",'
             '"reason":"The correction changes 100049722 to TP100049722."}'
    )])
    response = SimpleNamespace(
        data=SimpleNamespace(chat_response=SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="STOP")]
        )), headers={"opc-request-id": "req-reason"},
    )

    class Client:
        def chat(self, details):
            return response

    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    result = OciGeminiRuleGenerator(client=Client(), model_id="openai.gpt-oss-20b").generate_with_metadata(
        field_key="PONumber", display_label="PO Number", short_rule="po",
        detailed_rule=[], old_value="100049722", new_value="TP100049722"
    )
    assert result.reason is not None
    assert "100049722" not in result.reason
    assert result.reason == result.decision_summary


def test_parser_accepts_structured_json():
    assert parse_rule_response('{"sentence":"Extract the labeled tax amount."}')[0] == "Extract the labeled tax amount."


def test_parser_accepts_plain_sentence():
    assert parse_rule_response("Extract the labeled tax amount.")[0] == "Extract the labeled tax amount."


def test_parser_accepts_markdown_json_and_bom():
    text = '\ufeff```json\n{"sentence":"Extract the labeled tax amount."}\n```'
    assert parse_rule_response(text)[0] == "Extract the labeled tax amount."


def test_parser_rejects_commentary_and_multiple_sentences():
    import pytest
    with pytest.raises(ModelOutputError):
        parse_rule_response("Here is the rule: Extract the amount. Also ignore the rate.")


def test_parser_rejects_empty_output():
    import pytest
    with pytest.raises(ModelOutputError):
        parse_rule_response("   ", finish_reason="STOP")


def test_parser_rejects_truncated_preamble():
    import pytest
    with pytest.raises(ModelOutputError, match="all text parts"):
        parse_rule_parts([SimpleNamespace(text="Here is the JSON", thought=False)], finish_reason="STOP")


def test_parser_rejects_max_tokens_even_if_candidate_looks_valid():
    import pytest
    with pytest.raises(ModelOutputError, match="truncated"):
        parse_rule_parts([SimpleNamespace(text='{"sentence":"Extract the labeled tax amount."}', thought=False)], finish_reason="MAX_TOKENS")


def test_oci_request_has_no_stop_and_no_application_output_limit(monkeypatch):
    captured = {}
    response = SimpleNamespace(
        data=SimpleNamespace(chat_response=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="STOP", message=SimpleNamespace(content=[SimpleNamespace(text='{"sentence":"Use the labeled tax amount."}')]))])),
        headers={"opc-request-id": "req-test"})
    class Client:
        def chat(self, details):
            captured["request"] = details.chat_request
            return response
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    result = OciGeminiRuleGenerator(client=Client()).generate(field_key="TaxAmount", display_label="Tax", short_rule="tax", detailed_rule=[], old_value="1", new_value="2")
    assert result[0] == "Use the labeled tax amount."


def test_gpt_oss_reasoning_effort_is_sent_without_verbosity(monkeypatch):
    captured = {}
    response = SimpleNamespace(
        data=SimpleNamespace(chat_response=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="STOP", message=SimpleNamespace(content=[SimpleNamespace(
                text='{"sentence":"Use the labeled tax amount."}')]))])),
        headers={"opc-request-id": "req-reasoning"},
    )

    class Client:
        def chat(self, details):
            captured["request"] = details.chat_request
            return response

    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    provider = OciGeminiRuleGenerator(
        client=Client(), model_id="openai.gpt-oss-20b", reasoning_effort="medium"
    )
    result = provider.generate_with_metadata(
        field_key="TaxAmount", display_label="Tax", short_rule="tax", detailed_rule=[], old_value="1", new_value="2"
    )
    assert captured["request"].reasoning_effort == "MEDIUM"
    assert captured["request"].verbosity is None
    # OCI's BaseClient serializer uses attribute_map and omits None-valued
    # model fields. Exercise that same wire-shape rule, rather than only
    # checking Python object attributes.
    wire = {
        captured["request"].attribute_map[attr]: getattr(captured["request"], attr)
        for attr in captured["request"].swagger_types
        if getattr(captured["request"], attr) is not None
    }
    assert wire["reasoningEffort"] == "MEDIUM"
    assert "maxTokens" not in wire
    assert "verbosity" not in wire
    assert result.reasoning_effort_requested == "medium"
    assert result.reasoning_effort_effective == "MEDIUM"
    assert result.reasoning_supported is True
    assert result.visible_reasoning is False
    assert captured["request"].max_tokens is None
    assert getattr(captured["request"], "stop", None) in (None, [])


def test_oci_retries_provider_limit_once_without_application_budget(monkeypatch):
    requests = []
    responses = [
        SimpleNamespace(data=SimpleNamespace(chat_response=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="MAX_TOKENS", message=SimpleNamespace(content=[SimpleNamespace(text="Here is the JSON requested: `")]))])), headers={"opc-request-id": "req-1"}),
        SimpleNamespace(data=SimpleNamespace(chat_response=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="STOP", message=SimpleNamespace(content=[SimpleNamespace(text='{"sentence":"Use the labeled tax amount."}')]))])), headers={"opc-request-id": "req-2"}),
    ]
    class Client:
        def chat(self, details):
            requests.append(details.chat_request)
            return responses.pop(0)
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    result = OciGeminiRuleGenerator(client=Client()).generate(field_key="TaxAmount", display_label="Tax", short_rule="tax", detailed_rule=[], old_value="1", new_value="2")
    assert result[0] == "Use the labeled tax amount."
    assert [request.max_tokens for request in requests] == [None, None]


def test_oci_exhausted_truncation_exposes_safe_retry_diagnostics(monkeypatch):
    requests = []
    response = SimpleNamespace(data=SimpleNamespace(chat_response=SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="MAX_TOKENS",
            message=SimpleNamespace(content=[SimpleNamespace(text="{")]),
        )]
    )), headers={"opc-request-id": "req-truncated"})

    class Client:
        def chat(self, details):
            requests.append(details.chat_request)
            return response

    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    import pytest
    with pytest.raises(ModelOutputError) as caught:
        OciGeminiRuleGenerator(
            client=Client(), model_id="openai.gpt-oss-20b", reasoning_effort="high"
        ).generate(field_key="PONumber", display_label="PO Number", short_rule="po", detailed_rule=[], old_value="1", new_value="2")
    generation = caught.value.diagnostics["generation"]
    assert generation["attempts"] == 2
    assert generation["application_output_limit_sent"] is False
    assert generation["provider_managed_limit"] is True
    assert generation["finish_reasons"] == ["MAX_TOKENS"] * 2
    assert generation["request_ids"] == ["req-truncated"] * 2
    assert generation["reason"] == "provider_output_truncated_after_repair"
    assert [request.reasoning_effort for request in requests] == ["HIGH"] * 2


def test_oci_does_not_retry_safety(monkeypatch):
    calls = 0
    response = SimpleNamespace(data=SimpleNamespace(chat_response=SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="SAFETY", message=SimpleNamespace(content=[SimpleNamespace(text="blocked")]))])), headers={"opc-request-id": "req-safety"})
    class Client:
        def chat(self, details):
            nonlocal calls
            calls += 1
            return response
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    import pytest
    with pytest.raises(ModelOutputError):
        OciGeminiRuleGenerator(client=Client()).generate(field_key="TaxAmount", display_label="Tax", short_rule="tax", detailed_rule=[], old_value="1", new_value="2")
    assert calls == 1


def test_oci_does_not_retry_deterministic_400(monkeypatch):
    calls = 0

    class BadRequest(Exception):
        status = 400

    class Client:
        def chat(self, details):
            nonlocal calls
            calls += 1
            raise BadRequest("Invalid verbosity parameter")

    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    import pytest
    with pytest.raises(BadRequest):
        OciGeminiRuleGenerator(client=Client()).generate(
            field_key="TaxAmount", display_label="Tax", short_rule="tax",
            detailed_rule=[], old_value="1", new_value="2"
        )
    assert calls == 1


def test_repair_payload_is_compact_and_does_not_repeat_correction_values(monkeypatch):
    prompts = []
    responses = [
        SimpleNamespace(data=SimpleNamespace(chat_response=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="STOP", message=SimpleNamespace(content=[SimpleNamespace(
                text='{"sentence":"If the PO is present, use 9497384."}')]))])),
                       headers={"opc-request-id": "req-invalid"}),
        SimpleNamespace(data=SimpleNamespace(chat_response=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="STOP", message=SimpleNamespace(content=[SimpleNamespace(
                text='{"sentence":"Extract the value from the explicitly labeled purchase-order field."}')]))])),
                       headers={"opc-request-id": "req-repair"}),
    ]

    class Client:
        def chat(self, details):
            prompts.append(details.chat_request.messages[1].content[0].text)
            return responses.pop(0)

    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    result = OciNativeRuleGenerator(client=Client(), model_id="openai.gpt-oss-20b").generate_with_metadata(
        field_key="PONumber", display_label="PO Number", short_rule="Extract labeled PO.",
        detailed_rule=[], old_value="HK 9497384", new_value="9497384"
    )
    assert result.sentence.startswith("Extract the value")
    assert "HK 9497384" not in prompts[1]
    assert "9497384" not in prompts[1]


def test_when_is_valid_rule_language_but_fallback_hops_are_rejected():
    from app.sentence_validators import validate_sentence
    assert validate_sentence("Extract the value when it is next to its label.", {})
    import pytest
    with pytest.raises(ValueError):
        validate_sentence("Otherwise use a nearby value.", {})
