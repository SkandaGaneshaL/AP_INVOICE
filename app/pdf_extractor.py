from __future__ import annotations

import json
import os
import re
from io import BytesIO
from typing import Any

from .extraction_prompt import InvoiceExtractionPromptBuilder
from .oci_pdf_client import OciPdfClient
from .models import TokenUsage, UsageSummary
from .usage import summarize_usage


class PdfExtractionError(ValueError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class OciPdfExtractor:
    def __init__(self, repository, client: Any = None, model_id: str | None = None):
        self.repository = repository
        self.model_id = model_id or os.getenv("OCI_EXTRACTION_MODEL_ID", "google.gemini-2.5-flash")
        self.client = client or OciPdfClient(model_id=self.model_id)
        self.prompt_builder = InvoiceExtractionPromptBuilder()

    def schema(self) -> dict[str, Any]:
        """Local documentation schema; it is intentionally not sent as OCI strict schema."""
        properties: dict[str, Any] = {}
        for rule in self.repository.load():
            properties[rule.FIELD_KEY] = {
                "type": "object",
                "properties": {"value": {}, "Page": {"type": "integer"}},
                "required": ["value", "Page"],
                "additionalProperties": False,
            }
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    def prompt(self, extra_instruction: str | None = None) -> str:
        return self.prompt_builder.build(self.repository.load(), candidate_instruction=extra_instruction)

    @staticmethod
    def _clean_json(text: str) -> str:
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        return cleaned

    @staticmethod
    def _page(value: Any, field: str) -> int | None:
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise PdfExtractionError(
                "invalid extraction page value",
                details={"missing_fields": [], "unexpected_fields": [], "invalid_fields": [field]},
            )
        return value

    def validate(self, parsed: dict[str, Any], *, normalization_mode: str = "none") -> dict[str, Any]:
        rules = self.repository.load()
        expected = {rule.FIELD_KEY for rule in rules}
        if not isinstance(parsed, dict):
            raise PdfExtractionError("extraction output must be a JSON object")
        missing = sorted(expected - set(parsed))
        unexpected = sorted(set(parsed) - expected)
        if missing or unexpected:
            raise PdfExtractionError(
                "extraction fields do not match configured rule fields",
                details={"missing_fields": missing, "unexpected_fields": unexpected, "invalid_fields": []},
            )

        normalized: dict[str, Any] = {}
        invalid: list[str] = []
        for rule in rules:
            field = rule.FIELD_KEY
            raw = parsed[field]
            if self.prompt_builder.is_list_rule(rule) and isinstance(raw, list):
                normalized_items: list[dict[str, Any]] = []
                for item in raw:
                    if not isinstance(item, dict) or set(item) != {"value", "Page"}:
                        invalid.append(field)
                        continue
                    normalized_items.append({"value": item["value"], "Page": self._page(item["Page"], field)})
                normalized[field] = normalized_items
                continue
            if not isinstance(raw, dict) or set(raw) != {"value", "Page"}:
                invalid.append(field)
                continue
            normalized[field] = {"value": raw["value"], "Page": self._page(raw["Page"], field)}

        if invalid:
            raise PdfExtractionError(
                "invalid extraction field shape",
                details={"missing_fields": [], "unexpected_fields": [], "invalid_fields": sorted(set(invalid))},
            )
        self._apply_explicit_normalizations(normalized, rules, normalization_mode=normalization_mode)
        return normalized

    @staticmethod
    def _apply_explicit_normalizations(values: dict[str, Any], rules, *, normalization_mode: str = "none") -> None:
        """Apply only the transformation explicitly represented by the correction."""
        if normalization_mode != "remove_prefix":
            return
        node = values.get("InvoiceNumber")
        if isinstance(node, dict) and isinstance(node.get("value"), str):
            value = node["value"].strip()
            if re.fullmatch(r"[A-Za-z]+\d+", value):
                node["value"] = re.sub(r"^[A-Za-z]+(?=\d+$)", "", value)

    @staticmethod
    def _page_count(document_bytes: bytes) -> int | None:
        try:
            from pypdf import PdfReader
            return len(PdfReader(BytesIO(document_bytes)).pages)
        except Exception:
            return None

    def _repair_line_items(
        self,
        document_bytes: bytes,
        filename: str,
        extracted: dict[str, Any],
        prompt: str,
        normalization_mode: str = "none",
    ) -> tuple[dict[str, Any], bool, UsageSummary]:
        if "LineItems" not in extracted or extracted.get("LineItems"):
            return extracted, False, UsageSummary()
        repair_prompt = (
            f"{prompt}\n\n"
            "Focused line-item repair: extract one object for each physical item or service row. "
            "Do not create rows for tax, VAT, subtotal, total, or summary rows. "
            "Do not calculate missing values. Return the same complete JSON object and populate LineItems."
        )
        try:
            response = self.client.extract(document_bytes=document_bytes, filename=filename, prompt=repair_prompt)
            usage = getattr(response, "usage", None) or TokenUsage(call_type="pdf_line_item_repair")
            text = getattr(response, "text", response if isinstance(response, str) else "")
            repaired = self.validate(json.loads(self._clean_json(text)), normalization_mode=normalization_mode)
            if repaired.get("LineItems"):
                extracted["LineItems"] = repaired["LineItems"]
            return extracted, True, summarize_usage([usage])
        except Exception:
            # A missing line-item table is valid for some documents; preserve the
            # primary extraction when the focused repair cannot be completed.
            pass
        return extracted, True, UsageSummary(calls=1, unknown_calls=1)

    def extract(
        self,
        document_bytes: bytes,
        instruction_override: str | None = None,
        *,
        filename: str = "invoice.pdf",
        normalization_mode: str = "none",
    ) -> tuple[dict[str, Any], int | None, dict[str, Any]]:
        if not document_bytes or not document_bytes.startswith(b"%PDF"):
            raise PdfExtractionError("uploaded file is not a valid PDF", details={"code": "PDF_REQUIRED"})
        page_count = self._page_count(document_bytes)
        try:
            max_pages = int(os.getenv("MAX_PDF_PAGES", "20"))
        except ValueError:
            max_pages = 20
        if page_count is not None and page_count > max_pages:
            raise PdfExtractionError(
                f"PDF has {page_count} pages; maximum is {max_pages}",
                details={"code": "PDF_TOO_MANY_PAGES", "page_count": page_count, "max_pages": max_pages},
            )
        rules = self.repository.load()
        prompt = self.prompt(instruction_override)
        last_error: PdfExtractionError | None = None
        request_id: str | None = None
        finish_reason: str | None = None
        response_length = 0
        usage_records = []

        for attempt in range(2):
            repair_attempted = attempt == 1
            current_prompt = prompt
            if repair_attempted:
                current_prompt += (
                    "\nYour previous response did not match the extraction contract. "
                    "Return only JSON. Include every configured field exactly once. "
                    "Every scalar field must contain exactly value and Page, where Page is an integer or null. "
                    "Do not output Markdown or explanations."
                )
            try:
                response = self.client.extract(document_bytes=document_bytes, filename=filename, prompt=current_prompt)
                text = getattr(response, "text", response if isinstance(response, str) else "")
                request_id = getattr(response, "request_id", None)
                finish_reason = getattr(response, "finish_reason", None)
                usage_records.append(getattr(response, "usage", None) or TokenUsage(call_type="pdf_extraction"))
                response_length = len(text or "")
                if str(finish_reason or "").upper() in {"MAX_TOKENS", "LENGTH"}:
                    raise PdfExtractionError("OCI extraction output was truncated")
                normalized = self.validate(json.loads(self._clean_json(text)), normalization_mode=normalization_mode)
                populated = sum(
                    1
                    for value in normalized.values()
                    if (isinstance(value, dict) and value.get("value") not in (None, ""))
                    or (isinstance(value, list) and any(item.get("value") not in (None, "") for item in value))
                )
                normalized, line_item_repair_attempted, line_item_usage = self._repair_line_items(
                    document_bytes, filename, normalized, prompt, normalization_mode
                )
                usage = summarize_usage([*usage_records, line_item_usage])
                diagnostics = {
                    "configured_field_count": len(rules),
                    "populated_field_count": populated,
                    "null_field_count": len(rules) - populated,
                    "response_length": response_length,
                    "finish_reason": finish_reason,
                    "request_id": request_id,
                    "repair_attempted": repair_attempted,
                    "line_item_repair_attempted": line_item_repair_attempted,
                    "warning": "OCI returned structurally valid JSON but no configured fields were populated."
                    if populated == 0 else None,
                    "usage": usage.model_dump(),
                }
                return normalized, page_count, diagnostics
            except PdfExtractionError as exc:
                last_error = exc
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = PdfExtractionError(
                    "OCI extraction output did not match the extraction contract",
                    details={"parse_error": str(exc)[:200]},
                )

        raise last_error or PdfExtractionError("OCI extraction failed")


class OciPdfExtractionExecutor:
    """Executes one candidate instruction against the actual uploaded PDF."""

    def __init__(self, repository, client: Any = None, model_id: str | None = None):
        self.extractor = OciPdfExtractor(repository, client=client, model_id=model_id)

    def extract(self, *, document_bytes: bytes, mime_type: str, field_key: str,
                field_path: str, instruction: str, rules: list[str], normalization_mode: str = "none") -> Any:
        extracted = self.extract_full(document_bytes=document_bytes, mime_type=mime_type,
                                      instruction=instruction, normalization_mode=normalization_mode)
        return extracted.get(field_key, {"value": None, "Page": None})

    def extract_full(self, *, document_bytes: bytes, mime_type: str, instruction: str,
                     normalization_mode: str = "none") -> dict[str, Any]:
        extracted, _ = self.extract_full_with_metadata(
            document_bytes=document_bytes, mime_type=mime_type,
            instruction=instruction, normalization_mode=normalization_mode,
        )
        return extracted

    def extract_full_with_metadata(self, *, document_bytes: bytes, mime_type: str, instruction: str,
                                   normalization_mode: str = "none") -> tuple[dict[str, Any], dict[str, Any]]:
        extracted, _, diagnostics = self.extractor.extract(
            document_bytes,
            instruction_override=instruction,
            filename="evaluation.pdf",
            normalization_mode=normalization_mode,
        )
        return extracted, diagnostics
