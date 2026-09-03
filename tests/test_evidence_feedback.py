from io import BytesIO

import pymupdf

from app.evidence import ExtractionEvidenceBuilder
from app.evaluation import ExtractionEvaluator, canonicalize
from app.models import EvidenceMatch, RuleFeedbackPacket, RuleGenerationContext, RuleRecord
from app.prompt_builder import RulePromptBuilder


def _pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _rule() -> RuleRecord:
    return RuleRecord(ID=1, FIELD_KEY="InvoiceCurrency", DISPLAY_LABEL="Invoice Currency",
                      SHORT_RULE="Extract invoice currency.",
                      DETAILED_RULE=["Use the currency explicitly labeled on the invoice."])


def _invoice_number_rule() -> RuleRecord:
    return RuleRecord(ID=6, FIELD_KEY="InvoiceNumber", DISPLAY_LABEL="Invoice Number",
                      SHORT_RULE="invoice reference",
                      DETAILED_RULE=[
                          "Extract the invoice number/reference exactly as written from fields such as Invoice Number, Invoice No., Invoice ID, Tax Invoice Number, Bill Number, or equivalent.",
                          "Always remove leading alphabetic characters from the invoice number if the remaining part is a valid number.",
                      ])


def test_evidence_grounding_finds_label_and_competing_value():
    packet = ExtractionEvidenceBuilder().build(
        document_bytes=_pdf("Currency: INR\nReference amount in USD"), rule=_rule(),
        field_path="InvoiceCurrency", old_value="USD", new_value="INR",
        original_field_node={"value": "USD", "Page": 1},
        corrected_field_node={"value": "INR", "Page": 1},
    )
    assert any(item.value == "INR" and item.page == 1 for item in packet.evidence)
    assert any(item.value == "USD" for item in packet.competing_evidence)
    assert packet.failure_type == "currency_label_conflict"


def test_invoice_number_evidence_records_prefix_transformation():
    packet = ExtractionEvidenceBuilder().build(
        document_bytes=_pdf("Invoice #: TP100049722"), rule=_invoice_number_rule(),
        field_path="InvoiceNumber", old_value="TP100049722", new_value="100049722",
    )
    assert packet.evidence[0].label == "Invoice #"
    assert packet.evidence[0].raw_value == "TP100049722"
    assert packet.evidence[0].normalized_value == "100049722"
    assert packet.evidence[0].transformation == "remove_leading_alphabetic_prefix"
    assert "None label" not in packet.inferred_intent


def test_prompt_contains_evidence_and_no_hardcoded_rule_instruction():
    packet = RuleFeedbackPacket(
        field_key="InvoiceCurrency", field_path="InvoiceCurrency", display_label="Invoice Currency",
        previous_value="USD", corrected_value="INR",
        evidence=[EvidenceMatch(page=1, label="Currency", value="INR", snippet="Currency: INR", confidence=.98)],
        competing_evidence=[EvidenceMatch(page=1, value="USD", snippet="Reference amount in USD")],
        inferred_intent="Extract the value next to the explicit Currency label.",
        constraints=["Do not hard-code INR as a universal value."],
    )
    context = RuleGenerationContext(field_key="InvoiceCurrency", display_label="Invoice Currency",
        short_rule="Extract invoice currency.", detailed_rule=[], old_value="USD", new_value="INR",
        feedback_packet=packet)
    prompt = RulePromptBuilder.seed(context)
    assert "InvoiceCurrency" in prompt
    assert "Currency" in prompt
    assert "Do not repeat either invoice value" in prompt
    assert "Currency: INR" not in prompt
    assert "Reference amount in USD" not in prompt


class FakeExecutor:
    def extract_full(self, **kwargs):
        return {"InvoiceCurrency": {"value": "INR", "Page": 1},
                "InvoiceNumber": {"value": "INV-1", "Page": 1}}


def test_evaluator_scores_evidence_supported_extraction():
    context = RuleGenerationContext(
        field_key="InvoiceCurrency", field_path="InvoiceCurrency", old_value="USD", new_value="INR",
        final_response={"InvoiceCurrency": {"value": "INR", "Page": 1},
                        "InvoiceNumber": {"value": "INV-1", "Page": 1}},
        document_bytes=b"%PDF fake",
        feedback_packet=RuleFeedbackPacket(
            field_key="InvoiceCurrency", field_path="InvoiceCurrency", corrected_value="INR",
            evidence=[EvidenceMatch(page=1, value="INR", snippet="Currency: INR", confidence=.98)],
        ),
    )
    result, trace = ExtractionEvaluator(FakeExecutor()).evaluate(context, "Extract the value next to the explicit currency label.")
    assert trace.match is True
    assert trace.evidence_supported is True
    assert result.score >= .9


def test_invoice_number_canonicalization_is_rule_specific():
    context = RuleGenerationContext(
        field_key="InvoiceNumber", short_rule="invoice reference",
        detailed_rule=["Remove leading alphabetic characters if the remainder is numeric."],
        normalization_mode="remove_prefix",
    )
    assert canonicalize("InvoiceNumber", "TP100049722", context) == "100049722"
    context.normalization_mode = "preserve_prefix"
    assert canonicalize("InvoiceNumber", "TP100049722", context) == "TP100049722"
    assert canonicalize("InvoiceCurrency", "USD", context) == "USD"
