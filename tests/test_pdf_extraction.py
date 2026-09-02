import json

import pytest

from app.models import RuleRecord
from app.pdf_extractor import OciPdfExtractor, PdfExtractionError
from app.oci_pdf_client import OciPdfClient


class Repo:
    def load(self):
        return [
            RuleRecord(ID=1, FIELD_KEY="InvoiceCurrency", DISPLAY_LABEL="Currency", SHORT_RULE="Currency", DETAILED_RULE=["Use the explicitly labeled invoice currency."]),
            RuleRecord(ID=2, FIELD_KEY="InvoiceNumber", DISPLAY_LABEL="Number", SHORT_RULE="Number", DETAILED_RULE=["Extract the invoice reference."]),
        ]


class ListRepo:
    def load(self):
        return [
            RuleRecord(ID=1, FIELD_KEY="LineItemDescription", DISPLAY_LABEL="Description", SHORT_RULE="line item description", DETAILED_RULE=["If multiple line items are present, return all descriptions."]),
            RuleRecord(ID=2, FIELD_KEY="InvoiceCurrency", DISPLAY_LABEL="Currency", SHORT_RULE="Currency", DETAILED_RULE=["Use the explicitly labeled invoice currency."]),
        ]


class Response:
    def __init__(self, text, finish_reason=None):
        self.text = text
        self.finish_reason = finish_reason
        self.request_id = "fake-request"


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def extract(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        return self.responses.pop(0)


def payload(currency="INR", page=1):
    return json.dumps({
        "InvoiceCurrency": {"value": currency, "Page": page},
        "InvoiceNumber": {"value": "INV-1", "Page": 1},
    })


def test_prompt_contains_all_local_rules_and_reference_instructions():
    client = FakeClient([Response(payload())])
    extractor = OciPdfExtractor(Repo(), client=client, model_id="google.gemini-2.5-flash")
    extractor.extract(b"%PDF-1.4 fake")
    prompt = client.prompts[0]
    assert "InvoiceCurrency" in prompt
    assert "Use the explicitly labeled invoice currency." in prompt
    assert "Do not infer, calculate, normalize, or guess values" in prompt
    assert "Every scalar field must contain exactly `value` and `Page`" in prompt


def test_numeric_page_is_normalized_and_diagnostics_are_returned():
    client = FakeClient([Response(payload(page="2"))])
    extracted, _, diagnostics = OciPdfExtractor(Repo(), client=client).extract(b"%PDF-1.4 fake")
    assert extracted["InvoiceCurrency"]["Page"] == 2
    assert diagnostics["configured_field_count"] == 2
    assert diagnostics["populated_field_count"] == 2
    assert diagnostics["request_id"] == "fake-request"


def test_provenance_fields_are_not_returned_in_public_extraction():
    output = json.dumps({
        "InvoiceCurrency": {
            "value": "INR", "Page": "2", "raw_value": "INR",
            "canonical_value": "INR", "bbox": [1, 2, 3, 4],
            "source_label": "Currency", "evidence_text": "Currency INR",
            "confidence": 0.99, "absent_reason": None,
        },
        "InvoiceNumber": {"value": "INV-1", "Page": 1, "confidence": 0.9},
    })
    extracted, _, _ = OciPdfExtractor(Repo(), client=FakeClient([Response(output)])).extract(b"%PDF-1.4 fake")
    assert extracted == {
        "InvoiceCurrency": {"value": "INR", "Page": 2},
        "InvoiceNumber": {"value": "INV-1", "Page": 1},
    }


def test_missing_markers_are_normalized_without_provenance_fields():
    output = json.dumps({
        "InvoiceCurrency": {"value": "Not present", "Page": "1", "raw_value": "Not present"},
        "InvoiceNumber": {"value": "N/A", "Page": 1, "confidence": 0.1},
    })
    extracted, _, _ = OciPdfExtractor(Repo(), client=FakeClient([Response(output)])).extract(b"%PDF-1.4 fake")
    assert extracted["InvoiceCurrency"] == {"value": None, "Page": 1}
    assert extracted["InvoiceNumber"] == {"value": None, "Page": 1}


def test_line_item_provenance_is_not_returned_and_null_fields_are_preserved():
    output = json.dumps({
        "LineItemDescription": [{
            "LineItemDescription": {"value": "Service", "Page": "1", "raw_value": "Service"},
            "TaxAmount": None,
        }],
        "InvoiceCurrency": {"value": "INR", "Page": 1},
    })
    extracted, _, _ = OciPdfExtractor(ListRepo(), client=FakeClient([Response(output)])).extract(b"%PDF-1.4 fake")
    assert extracted["LineItemDescription"][0]["LineItemDescription"] == {"value": "Service", "Page": 1}
    assert extracted["LineItemDescription"][0]["TaxAmount"] is None


def test_invalid_page_is_repaired_once_then_fails_with_details():
    invalid = json.dumps({
        "InvoiceCurrency": {"value": "INR", "Page": "Page 2"},
        "InvoiceNumber": {"value": "INV-1", "Page": 1},
    })
    client = FakeClient([Response(invalid), Response(invalid)])
    with pytest.raises(PdfExtractionError) as error:
        OciPdfExtractor(Repo(), client=client).extract(b"%PDF-1.4 fake")
    assert error.value.details["invalid_fields"] == ["InvoiceCurrency"]
    assert len(client.prompts) == 2


def test_markdown_fenced_output_is_accepted():
    client = FakeClient([Response(f"```json\n{payload()}\n```")])
    extracted, _, _ = OciPdfExtractor(Repo(), client=client).extract(b"%PDF-1.4 fake")
    assert extracted["InvoiceCurrency"]["value"] == "INR"


def test_list_valued_rules_are_normalized():
    output = json.dumps({
        "LineItemDescription": [{"value": "Service", "Page": "1"}],
        "InvoiceCurrency": {"value": "INR", "Page": 1},
    })
    extracted, _, _ = OciPdfExtractor(ListRepo(), client=FakeClient([Response(output)])).extract(b"%PDF-1.4 fake")
    assert extracted["LineItemDescription"] == [{"value": "Service", "Page": 1}]


def test_missing_or_unexpected_fields_are_rejected():
    missing = json.dumps({"InvoiceCurrency": {"value": "INR", "Page": 1}})
    client = FakeClient([Response(missing), Response(missing)])
    with pytest.raises(PdfExtractionError) as error:
        OciPdfExtractor(Repo(), client=client).extract(b"%PDF-1.4 fake")
    assert "InvoiceNumber" in error.value.details["missing_fields"]


def test_oci_pdf_client_sends_non_persistent_input_file(monkeypatch):
    monkeypatch.setenv("OCI_EXC_COMPARTMENT_ID", "ocid.test")

    class Responses:
        def __init__(self):
            self.request = None

        def create(self, **request):
            self.request = request
            return type("Response", (), {"output_text": "{}", "headers": {"opc-request-id": "req-1"}})()

    class Client:
        def __init__(self):
            self.responses = Responses()

    fake = Client()
    response = OciPdfClient(client=fake).extract(
        document_bytes=b"%PDF-1.4 fake", filename="invoice.pdf", prompt="Extract JSON"
    )
    request = fake.responses.request
    assert request["store"] is False
    assert request["input"][0]["content"][0]["type"] == "input_file"
    assert request["input"][0]["content"][0]["mime_type"] == "application/pdf"
    assert response.request_id == "req-1"
