from __future__ import annotations

import hashlib
import re
from typing import Any

from .models import EvidenceMatch, RuleFeedbackPacket, RuleRecord


def _text_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value or "").strip()


def _camel_words(value: str) -> list[str]:
    return [part.lower() for part in re.sub(r"([a-z])([A-Z])", r"\1 \2", value).split() if part]


class ExtractionEvidenceBuilder:
    """Finds deterministic text/layout evidence for one corrected field."""

    def build(
        self,
        *,
        document_bytes: bytes,
        rule: RuleRecord,
        field_path: str,
        old_value: Any,
        new_value: Any,
        original_field_node: Any = None,
        corrected_field_node: Any = None,
    ) -> RuleFeedbackPacket:
        evidence: list[EvidenceMatch] = []
        competing: list[EvidenceMatch] = []
        try:
            import pymupdf

            document = pymupdf.open(stream=document_bytes, filetype="pdf")
        except Exception:
            return self._packet(rule, field_path, old_value, new_value, original_field_node,
                                corrected_field_node, evidence, competing, source="unavailable")

        labels = self._labels(rule)
        new_text = _text_value(new_value)
        old_text = _text_value(old_value)
        for page_index, page in enumerate(document, start=1):
            lines = self._lines(page)
            for line_text, bbox in lines:
                label_match = self._find_label(line_text, labels)
                if label_match:
                    label, end = label_match
                    raw = self._value_after_label(line_text, end)
                    if raw and new_text and self._exact_token(raw, new_text):
                        evidence.append(EvidenceMatch(
                            page=page_index, label=label, value=new_value, raw_value=raw,
                            normalized_value=new_value, snippet=line_text[:500], bbox=bbox,
                            confidence=0.98, label_match=True,
                        ))
                    elif raw and old_text and self._exact_token(raw, old_text):
                        transformation = self._transformation(old_text, new_text)
                        if transformation:
                            evidence.append(EvidenceMatch(
                                page=page_index, label=label, value=new_value, raw_value=raw,
                                normalized_value=new_value, snippet=line_text[:500], bbox=bbox,
                                confidence=0.98, match_type="derived_from_observed_value",
                                transformation=transformation, label_match=True,
                            ))
                        else:
                            competing.append(EvidenceMatch(
                                page=page_index, label=label, value=old_value, raw_value=raw,
                                normalized_value=old_value, snippet=line_text[:500], bbox=bbox,
                                confidence=0.72, label_match=True,
                            ))
                    continue

                # Values without a recognized label are retained only as weak
                # competing/source evidence when they are exact tokens. A
                # substring such as 100 inside TP100 is not evidence.
                if new_text and self._exact_token(line_text, new_text):
                    evidence.append(EvidenceMatch(page=page_index, value=new_value,
                        raw_value=new_text, normalized_value=new_value, snippet=line_text[:500],
                        bbox=bbox, confidence=0.55, label_match=False))
                if old_text and self._exact_token(line_text, old_text):
                    competing.append(EvidenceMatch(page=page_index, value=old_value,
                        raw_value=old_text, normalized_value=old_value, snippet=line_text[:500],
                        bbox=bbox, confidence=0.55, label_match=False))

            # When the label and value are on adjacent lines, preserve that
            # relationship as evidence even if the PDF text extraction split it.
            for index, (line_text, bbox) in enumerate(lines):
                if not self._find_label(line_text, labels):
                    continue
                adjacent = " ".join(item[0] for item in lines[index:index + 3])
                if new_text and self._exact_token(adjacent, new_text) and not any(
                    item.page == page_index and item.snippet == adjacent[:500] for item in evidence
                ):
                    evidence.append(EvidenceMatch(page=page_index, label=self._find_label(line_text, labels)[0],
                        value=new_value, raw_value=new_text, normalized_value=new_value,
                        snippet=adjacent[:500], bbox=bbox, confidence=0.90, label_match=True))

        document.close()
        return self._packet(rule, field_path, old_value, new_value, original_field_node,
                            corrected_field_node, evidence[:8], competing[:8])

    @staticmethod
    def _labels(rule: RuleRecord) -> list[str]:
        labels = {
            rule.FIELD_KEY.lower(),
            rule.DISPLAY_LABEL.lower(),
            " ".join(_camel_words(rule.FIELD_KEY)),
        }
        if rule.FIELD_KEY.lower() == "invoicecurrency":
            labels.update({"currency", "invoice currency", "currency code"})
        if rule.FIELD_KEY.lower() == "invoicenumber":
            labels.update({"invoice number", "invoice no", "invoice #", "invoice id",
                           "tax invoice number", "bill number", "bill no"})
        for item in rule.DETAILED_RULE:
            for match in re.findall(r"(?:labeled|labelled|fields? labeled|fields? such as|such as)\s+([^.;]+)", item, re.I):
                labels.update(part.strip().lower() for part in re.split(r",|/|\bor\b", match) if part.strip())
        return sorted((item for item in labels if len(item) >= 3), key=len, reverse=True)

    @staticmethod
    def _find_label(line: str, labels: list[str]) -> tuple[str, int] | None:
        for label in labels:
            pattern = re.escape(label).replace(r"\ ", r"\s+")
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(0), match.end()
        return None

    @staticmethod
    def _value_after_label(line: str, end: int) -> str:
        remainder = line[end:].lstrip(" \t:#-–—")
        match = re.match(r"([^\s,;|]+)", remainder)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _exact_token(text: str, value: str) -> bool:
        if not text or not value:
            return False
        return re.search(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", text, re.IGNORECASE) is not None

    @staticmethod
    def _transformation(old_value: str, new_value: str) -> str | None:
        if not old_value or not new_value or old_value == new_value:
            return None
        if old_value.lower().endswith(new_value.lower()) and re.fullmatch(r"[A-Za-z]+", old_value[:-len(new_value)] or "") and re.fullmatch(r"\d+", new_value):
            return "remove_leading_alphabetic_prefix"
        return None

    @staticmethod
    def _lines(page: Any) -> list[tuple[str, list[float]]]:
        words = page.get_text("words") or []
        grouped: dict[tuple[int, int], list[Any]] = {}
        for word in words:
            grouped.setdefault((int(word[5]), int(word[6])), []).append(word)
        lines: list[tuple[str, list[float]]] = []
        for items in grouped.values():
            items.sort(key=lambda item: item[0])
            text = " ".join(str(item[4]) for item in items).strip()
            if not text:
                continue
            bbox = [min(item[0] for item in items), min(item[1] for item in items),
                    max(item[2] for item in items), max(item[3] for item in items)]
            lines.append((text, bbox))
        return lines

    @staticmethod
    def _packet(rule, field_path, old_value, new_value, original_field_node,
                corrected_field_node, evidence, competing, source="pdf_text"):
        failure_type = "label_value_conflict" if evidence and competing else "missing_explicit_label"
        if evidence and evidence[0].transformation:
            failure_type = "invoice_number_normalization"
        if rule.FIELD_KEY.lower().startswith("invoicecurrency"):
            failure_type = "currency_label_conflict" if competing else "currency_label_lookup"
        label = evidence[0].label if evidence and evidence[0].label else None
        if evidence and evidence[0].transformation == "remove_leading_alphabetic_prefix":
            intent = ("Extract the invoice number next to the explicit invoice-number label and remove leading "
                      "alphabetic characters only when the remaining portion is a valid number.")
            observed = (f"The invoice number was observed as {evidence[0].raw_value} next to the {label} label. "
                        f"The corrected value is {new_value} after the permitted transformation.")
        elif label:
            intent = f"Extract the value associated with the explicit {label} label, not an unrelated occurrence."
            observed = f"The document evidence shows {evidence[0].raw_value or new_value} next to the {label} label."
        else:
            intent = "No explicit field label was located; use the configured field rule and do not infer from unrelated values."
            observed = f"Previous extracted value: {old_value}; corrected value: {new_value}."
        constraints = [
            f"Do not hard-code {_text_value(new_value)} as a universal value.",
            "Use only evidence present in the invoice.",
            "Prefer an explicit field label over an unrelated reference, conversion, or summary occurrence.",
            "Do not remove meaningful alphanumeric characters unless the configured field rule explicitly permits the transformation.",
        ]
        confidence = "high" if evidence and evidence[0].confidence >= 0.9 else "limited"
        return RuleFeedbackPacket(
            field_key=rule.FIELD_KEY,
            field_path=field_path,
            display_label=rule.DISPLAY_LABEL,
            previous_value=old_value,
            corrected_value=new_value,
            original_field_node=original_field_node,
            corrected_field_node=corrected_field_node,
            short_rule=rule.SHORT_RULE,
            detailed_rules=list(rule.DETAILED_RULE),
            evidence=evidence,
            competing_evidence=competing,
            observed_correction=observed,
            inferred_intent=intent,
            constraints=constraints,
            failure_type=failure_type,
            confidence=confidence if source != "unavailable" else "limited",
        )


def layout_signature(document_bytes: bytes) -> str:
    try:
        import pymupdf

        document = pymupdf.open(stream=document_bytes, filetype="pdf")
        shape = [(round(page.rect.width), round(page.rect.height), len(page.get_text("words") or [])) for page in document]
        document.close()
        return hashlib.sha256(repr(shape).encode()).hexdigest()[:16]
    except Exception:
        return "unknown"
