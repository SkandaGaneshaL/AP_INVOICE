from __future__ import annotations

import hashlib
import re
from typing import Any
from .type_inference import parse_date, parse_number

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
                if re.search(r"(?:order\s*no|ship\s*no|sales\s*order)", line_text, re.I):
                    reference = line_text
                    # The identifier is commonly emitted on the next text
                    # line by PDF extraction; the later pass also covers
                    # labels that share a line with another field.
                    raw_reference = reference
                    if not re.search(r"\b[A-Z]{1,6}[- ]?\d{4,}\b", reference, re.I):
                        try:
                            line_index = next(i for i, item in enumerate(lines) if item[0] == line_text and item[1] == bbox)
                            reference = " ".join(item[0] for item in lines[line_index:line_index + 2]).strip()
                        except StopIteration:
                            pass
                    ref_match = re.search(r"\b[A-Z]{1,6}[- ]?\d{4,}\b", reference, re.I)
                    raw_reference = ref_match.group(0) if ref_match else reference[:240]
                    competing.append(EvidenceMatch(
                        page=page_index, label="Order No./Ship No.", value=None,
                        raw_value=raw_reference, normalized_value=None,
                        snippet=reference[:500], bbox=bbox, confidence=.72,
                        match_type="competing_reference", label_match=True,
                    ))
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

                # Preserve nearby shipment/order references as explicit
                # competing evidence for identifier corrections. They must
                # never be silently treated as the selected labeled value.
                if re.search(r"(?:order\s*no|ship\s*no|sales\s*order)", line_text, re.I):
                    competing.append(EvidenceMatch(
                        page=page_index, label="Order No./Ship No.", value=None,
                        raw_value=line_text[:240], normalized_value=None,
                        snippet=line_text[:500], bbox=bbox, confidence=.55,
                        match_type="competing_reference", label_match=True,
                    ))

                # Values without a recognized label are retained only as weak
                # competing/source evidence when they are exact tokens. A
                # substring such as 100 inside TP100 is not evidence.
                if new_text and not evidence and not self._exact_token(line_text, old_text) and self._exact_token(line_text, new_text):
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
                if re.search(r"\bPO\s+BOX\b", line_text, re.I):
                    continue
                adjacent = " ".join(item[0] for item in lines[index:index + 3])
                adjacent_label = self._find_label(line_text, labels)
                transformation = self._transformation(old_text, new_text) if old_text and self._exact_token(adjacent, old_text) else None
                if ((new_text and self._exact_token(adjacent, new_text)) or transformation) and not any(
                    item.page == page_index and item.snippet == adjacent[:500] for item in evidence
                ):
                    evidence.append(EvidenceMatch(
                        page=page_index, label=adjacent_label[0] if adjacent_label else None,
                        value=new_value, raw_value=old_text if transformation else new_text,
                        normalized_value=new_value, snippet=adjacent[:500], bbox=bbox,
                        confidence=0.98 if transformation else 0.90,
                        match_type="derived_from_observed_value" if transformation else "exact",
                        transformation=transformation, label_match=adjacent_label is not None))

                # Some PDF text layers put an order/shipment identifier on
                # the line immediately after its label. Preserve that
                # reference as competing evidence without selecting it.
                if re.search(r"(?:order\s*no|ship\s*no|sales\s*order)", line_text, re.I):
                    reference = " ".join(item[0] for item in lines[index:index + 2]).strip()
                    ref_match = re.search(r"\b[A-Z]{1,6}[- ]?\d{4,}\b", reference, re.I)
                    raw_reference = ref_match.group(0) if ref_match else reference[:240]
                    if not any(item.raw_value == raw_reference and item.match_type == "competing_reference" for item in competing):
                        competing.append(EvidenceMatch(
                            page=page_index, label="Order No./Ship No.", value=None,
                            raw_value=raw_reference, normalized_value=None,
                            snippet=reference[:500], bbox=bbox, confidence=.72,
                            match_type="competing_reference", label_match=True,
                        ))

        document.close()
        evidence = sorted(evidence, key=lambda item: (not item.label_match, -item.confidence))
        evidence = list({(item.page, item.label, item.raw_value, item.transformation): item for item in evidence}.values())
        competing = sorted(competing, key=lambda item: (item.label_match, -item.confidence))
        if any(item.label_match for item in evidence):
            selected_raw = {item.raw_value for item in evidence if item.label_match}
            competing = [item for item in competing if item.label_match or item.match_type != "exact" or item.raw_value not in selected_raw]
        competing = list({(item.page, item.label, item.raw_value, item.match_type): item for item in competing}.values())
        competing = [item for item in competing if item.match_type != "competing_reference" or
                     re.search(r"\b[A-Z]{1,6}[- ]?\d{4,}\b", str(item.raw_value or ""), re.I)]
        return self._packet(rule, field_path, old_value, new_value, original_field_node,
                            corrected_field_node, evidence[:8], competing[:8])

    @staticmethod
    def _labels(rule: RuleRecord) -> list[str]:
        rule_text = " ".join([rule.FIELD_KEY, rule.DISPLAY_LABEL, rule.SHORT_RULE, *rule.DETAILED_RULE])
        labels = {
            rule.FIELD_KEY.lower(),
            rule.DISPLAY_LABEL.lower(),
            " ".join(_camel_words(rule.FIELD_KEY)),
        }
        if re.search(r"\bcurrency\b", rule_text, re.I):
            labels.add("currency")
        if re.search(r"\binvoice\b", rule_text, re.I) and re.search(r"\b(number|no\.?|id|#)\b", rule_text, re.I):
            labels.update({"invoice number", "invoice no", "invoice #", "invoice id"})
        if re.search(r"\bpo\b|purchase\s+order", rule_text, re.I):
            labels.update({"po", "po number", "purchase order"})
        for item in rule.DETAILED_RULE:
            for match in re.findall(r"(?:labeled|labelled|fields? labeled|fields? such as|such as)\s+([^.;]+)", item, re.I):
                labels.update(part.strip().lower() for part in re.split(r",|/|\bor\b", match) if part.strip())
        return sorted((item for item in labels if len(item) >= 2), key=len, reverse=True)

    @staticmethod
    def _find_label(line: str, labels: list[str]) -> tuple[str, int] | None:
        for label in labels:
            pattern = r"(?<!\w)" + re.escape(label).replace(r"\ ", r"\s+") + r"(?!\w)"
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                matched_label = match.group(0)
                prefix = line[:match.start()].rstrip()
                if matched_label.casefold() == "po" and re.search(r"customer$", prefix, re.I):
                    matched_label = "Customer PO"
                return matched_label, match.end()
        return None

    @staticmethod
    def _value_after_label(line: str, end: int) -> str:
        remainder = line[end:].lstrip(" \t:#-–—")
        # Keep the complete same-line value; identifier corrections often
        # contain a prefix token plus a numeric core (for example HK 9497384).
        return re.split(r"\s+(?=(?:Customer\s+PO|Order\s+No|Ship\s+No|Invoice\s+Date)\b)", remainder, maxsplit=1, flags=re.I)[0].strip()

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
        numeric_text = lambda value: not re.search(r"[A-Za-z]", str(value).replace("USD", "").replace("EUR", "").replace("INR", ""), re.I)
        if numeric_text(old_value) and numeric_text(new_value) and parse_number(old_value) is not None and parse_number(old_value) == parse_number(new_value):
            return "numeric_thousands_canonicalize"
        old_date, new_date = parse_date(old_value), parse_date(new_value)
        if old_date and new_date and old_date[0] == new_date[0]:
            return "date_format_repattern"
        match = re.match(r"^([A-Za-z]+)[\s_-]+(.+)$", old_value.strip())
        if match and match.group(2).casefold() == new_value.strip().casefold():
            return "identifier_strip_leading_alpha"
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
            failure_type = "value_normalization"
        label = evidence[0].label if evidence and evidence[0].label else None
        if label and "currency" in label.casefold():
            failure_type = "currency_label_conflict" if competing else "currency_label_lookup"
        if evidence and evidence[0].transformation in {"remove_leading_alphabetic_prefix", "identifier_strip_leading_alpha"}:
            intent = ("Extract the value next to the explicit field label and remove a leading alphabetic "
                      "prefix only when the remaining portion is a valid identifier core.")
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
