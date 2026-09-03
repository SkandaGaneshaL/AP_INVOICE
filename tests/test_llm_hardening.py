import json
from types import SimpleNamespace

from app.correction_kind import CorrectionKind, classify_correction
from app.models import RuleGenerationContext
from app.rule_merger import has_semantic_conflict
from app.sentence_payload import build_policy_gist, build_sentence_payload
from app.sentence_validators import assemble_local_sentence, validate_sentence
from app.extraction_prompt import InvoiceExtractionPromptBuilder
from app.models import RuleRecord
from app.update_jobs import RuleUpdateJobStore


def test_correction_taxonomy_is_value_delta_based():
    assert classify_correction("HK 9497384", "9497384") == CorrectionKind.IDENTIFIER_PREFIX
    assert classify_correction("Acme Ltd", "Acme") == CorrectionKind.LEGAL_SUFFIX
    assert classify_correction("01/02/2026", "02/01/2026") == CorrectionKind.DATE_FORMAT
    assert classify_correction("A  B", "A B") == CorrectionKind.WHITESPACE
    assert classify_correction("same", "same") == CorrectionKind.NOOP


def test_sentence_payload_is_compact_and_does_not_include_full_invoice():
    context = RuleGenerationContext(
        field_key="PONumber", display_label="PO Number", short_rule="Use the labeled PO.",
        detailed_rule=["Prefer the explicit label."], old_value="HK 9497384", new_value="9497384",
        invoice_payload={"PONumber": {"value": "HK 9497384"}, "Other": {"value": "secret"}},
        final_response={"PONumber": {"value": "9497384"}},
    )
    payload = build_sentence_payload(context)
    encoded = json.dumps(payload)
    assert payload["sentence_payload_version"] == "correction-delta-v1"
    assert "invoice_payload" not in payload
    assert "secret" not in encoded
    assert len(encoded) < 2000


def test_policy_gist_keeps_behavior_bearing_clauses():
    rule = SimpleNamespace(
        SHORT_RULE="Extract the explicitly labeled value.",
        DETAILED_RULE=["Normalize whitespace.", "Ignore unrelated occurrences.", "Historical noise."],
    )
    gist = build_policy_gist(rule)
    assert "labeled" in gist.lower()
    assert "normalize" in gist.lower()
    assert "ignore" in gist.lower()


def test_unsafe_sentence_falls_back_to_local_assembly():
    payload = {
        "field_key": "PONumber", "display_label": "PO Number",
        "correction_kind": "identifier_prefix_removed",
        "delta": {"old_value": "HK 9497384", "new_value": "9497384"},
    }
    try:
        validate_sentence("Otherwise use the invoice number.", payload)
    except ValueError:
        pass
    else:
        raise AssertionError("fallback wording must be rejected")
    sentence = validate_sentence(assemble_local_sentence(payload), payload)
    assert "prefix" in sentence.lower()


def test_merge_conflict_detection_is_explicit():
    assert has_semantic_conflict(["Do not normalize the value."], "Normalize the value from the label.")
    assert not has_semantic_conflict(["Extract the labeled value."], "Preserve the configured date format.")


def test_sentence_must_encode_the_observed_correction():
    payload = {
        "correction_kind": "date_format_changed",
        "delta": {"old_value": "23/06/2026", "new_value": "06/23/2026"},
    }
    import pytest
    with pytest.raises(ValueError, match="date correction"):
        validate_sentence("Extract the value from the invoice date field.", payload)
    assert validate_sentence("Extract the invoice date and apply the configured date format.", payload)


def test_scalar_rule_is_not_misclassified_by_negative_line_item_prose():
    scalar = RuleRecord(
        ID=1, FIELD_KEY="TaxAmount", DISPLAY_LABEL="Tax/VAT Amount",
        SHORT_RULE="explicit invoice tax amount",
        DETAILED_RULE=["Do not use line items or gross/net differences."],
    )
    list_rule = RuleRecord(
        ID=2, FIELD_KEY="LineItemDescription", DISPLAY_LABEL="Line Item Description",
        SHORT_RULE="invoice line item description", DETAILED_RULE=[],
    )
    builder = InvoiceExtractionPromptBuilder()
    assert not builder.is_list_rule(scalar)
    assert builder.is_list_rule(list_rule)


def test_job_call_counter_cannot_regress_when_partial_snapshot_has_zero():
    store = RuleUpdateJobStore()
    job_id = store.create()
    result = {"metadata": {"oci_sentence_generation_called": True, "oci_sentence_generation_call_count": 1},
              "usage": {"calls": 1}}
    store.update_normal(job_id, result, completed_fields=1, total_fields=2)
    store.update_normal(job_id, {"metadata": {"oci_sentence_generation_called": False,
                                                "oci_sentence_generation_call_count": 0}},
                        completed_fields=2, total_fields=2)
    assert store.get(job_id)["oci_sentence_generation_call_count"] == 1
