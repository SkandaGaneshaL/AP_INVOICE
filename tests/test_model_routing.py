from app.comparator import analyze_changes, find_changes
from app.model_registry import resolve_rule_generation_model, rule_generation_settings
from app.model_registry import RuleGenerationConfigurationError, validate_rule_generation_configuration
from app.models import RuleGenerationContext
from app.oci_provider import OciNativeRuleGenerator


class FakeChatClient:
    def __init__(self):
        self.requests = []

    def chat(self, details):
        self.requests.append(details)
        from types import SimpleNamespace
        return SimpleNamespace(
            data=SimpleNamespace(chat_response=SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="STOP",
                    message=SimpleNamespace(content=[SimpleNamespace(text='{"sentence":"Extract the value next to the invoice number label.","noop":false,"behavior":"labeled_value","label_policy":"explicit label","transform_policy":"preserve","null_policy":"labeled_empty_to_null","scope":"this_supplier_only"}', thought=False)])
                )],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=12, total_tokens=112),
            )),
            headers={"opc-request-id": "req-test"},
        )


def test_sentence_model_registry_defaults_to_gpt_oss(monkeypatch):
    monkeypatch.delenv("OCI_RULE_GENERATION_MODEL_ID", raising=False)
    selected = resolve_rule_generation_model(None)
    assert selected.key == "gpt-oss-20b"
    assert selected.model_id == "openai.gpt-oss-20b"


def test_gpt4o_model_registry_and_settings(monkeypatch):
    monkeypatch.setenv("OCI_GPT4O_RULE_REGION", "us-chicago-1")
    monkeypatch.setenv("OCI_GPT4O_RULE_ENDPOINT", "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com")
    monkeypatch.setenv("OCI_GPT4O_RULE_COMPARTMENT_ID", "ocid1.compartment.oc1..gpt4o")
    monkeypatch.setenv("OCI_GPT4O_RULE_SERVING_MODE", "on_demand")
    selected = resolve_rule_generation_model("gpt-4o")
    settings = rule_generation_settings(selected)
    assert selected.model_id == "openai.gpt-4o"
    assert settings["region"] == "us-chicago-1"
    assert settings["endpoint"].endswith("us-chicago-1.oci.oraclecloud.com")
    assert settings["serving_mode"] == "on_demand"
    validate_rule_generation_configuration(selected)


def test_gpt_oss_provider_uses_text_only_prompt_and_usage():
    client = FakeChatClient()
    provider = OciNativeRuleGenerator(client=client, compartment_id="ocid1.compartment.oc1..test", model_id="openai.gpt-oss-20b")
    context = RuleGenerationContext(
        field_key="InvoiceNumber",
        display_label="Invoice Number",
        short_rule="Extract the invoice number.",
        old_value="1",
        new_value="2",
    )

    result = provider.generate_with_metadata(context)

    request = client.requests[0].chat_request
    assert request.messages[1].content[0].text
    assert request.temperature == 0.0
    assert request.reasoning_effort == "LOW"
    assert result.sentence.startswith("Extract the value")
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 12
    assert result.request_id == "req-test"


def test_invoice_number_prefix_change_is_detected_without_stripping():
    changes = find_changes(
        {"InvoiceNumber": {"value": "100049722", "Page": 1}},
        {"InvoiceNumber": {"value": "TP100049722", "Page": 1}},
    )
    assert len(changes) == 1
    assert changes[0].old_value == "100049722"
    assert changes[0].new_value == "TP100049722"
    assert changes[0].normalized_new_value == "TP100049722"


def test_page_only_change_is_ignored():
    assert find_changes(
        {"InvoiceNumber": {"value": "100049722", "Page": 1}},
        {"InvoiceNumber": {"value": "100049722", "Page": 2}},
    ) == []


def test_change_analysis_reports_page_only_and_unchanged_fields():
    analysis = analyze_changes(
        {"InvoiceNumber": {"value": "100", "Page": 1}, "InvoiceDate": {"value": "2026-01-01"}},
        {"InvoiceNumber": {"value": "100", "Page": 2}, "InvoiceDate": {"value": "2026-01-01"}},
    )
    assert analysis.changes == []
    assert analysis.ignored_page_only_fields == ["InvoiceNumber"]
    assert "InvoiceDate" in analysis.ignored_unchanged_fields
    assert analysis.model_dump()["reason"] == "no_mapped_field_changes"


def test_dedicated_gpt_requires_endpoint(monkeypatch):
    monkeypatch.setenv("OCI_RULE_GENERATION_SERVING_MODE", "dedicated")
    monkeypatch.delenv("OCI_RULE_GENERATION_ENDPOINT_ID", raising=False)
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.example")
    model = resolve_rule_generation_model("gpt-oss-20b")
    try:
        validate_rule_generation_configuration(model)
    except RuleGenerationConfigurationError as exc:
        assert exc.code == "MODEL_ENDPOINT_CONFIGURATION_ERROR"
        assert exc.model == "openai.gpt-oss-20b"
        assert exc.region == "us-chicago-1"
    else:
        raise AssertionError("dedicated GPT configuration should require an endpoint")


def test_on_demand_gpt_does_not_require_endpoint(monkeypatch):
    monkeypatch.setenv("OCI_RULE_GENERATION_SERVING_MODE", "on_demand")
    monkeypatch.delenv("OCI_RULE_GENERATION_ENDPOINT_ID", raising=False)
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.example")
    model = resolve_rule_generation_model("gpt-oss-20b")
    validate_rule_generation_configuration(model)


def test_model_specific_rule_generation_settings_override_shared_defaults(monkeypatch):
    monkeypatch.setenv("OCI_RULE_GENERATION_REGION", "us-chicago-1")
    monkeypatch.setenv("OCI_RULE_GENERATION_ENDPOINT", "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com")
    monkeypatch.setenv("OCI_GPT_RULE_ENDPOINT", "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com")
    monkeypatch.setenv("OCI_GPT_RULE_COMPARTMENT_ID", "ocid1.compartment.oc1..valid")
    monkeypatch.setenv("OCI_GPT_RULE_SERVING_MODE", "on_demand")
    monkeypatch.setenv("OCI_GEMINI_RULE_ENDPOINT", "https://inference.generativeai.ap-hyderabad-1.oci.oraclecloud.com/openai/v1")
    monkeypatch.setenv("OCI_GEMINI_RULE_COMPARTMENT_ID", "ocid1.compartment.oc1..valid-hyd")
    monkeypatch.delenv("OCI_GEMINI_RULE_REGION", raising=False)

    gpt = rule_generation_settings(resolve_rule_generation_model("gpt-oss-20b"))
    gemini = rule_generation_settings(resolve_rule_generation_model("gemini-2.5-flash"))

    assert gpt["region"] == "us-chicago-1"
    assert gpt["compartment_id"] == "ocid1.compartment.oc1..valid"
    assert gemini["region"] == "ap-hyderabad-1"
    assert gemini["compartment_id"] == "ocid1.compartment.oc1..valid-hyd"


def test_regional_url_is_not_accepted_as_dedicated_endpoint_id(monkeypatch):
    monkeypatch.setenv("OCI_RULE_GENERATION_SERVING_MODE", "dedicated")
    monkeypatch.setenv("OCI_RULE_GENERATION_ENDPOINT_ID", "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com")
    monkeypatch.setenv("OCI_GPT_RULE_COMPARTMENT_ID", "ocid1.compartment.oc1..valid")
    model = resolve_rule_generation_model("gpt-oss-20b")
    try:
        validate_rule_generation_configuration(model)
    except RuleGenerationConfigurationError as exc:
        assert exc.code == "MODEL_ENDPOINT_CONFIGURATION_ERROR"
    else:
        raise AssertionError("a regional URL must not satisfy dedicated endpoint configuration")
