import re
from dataclasses import dataclass, field
from typing import Any
from .models import FieldChange


def _value(node: Any) -> Any:
    if isinstance(node, dict) and "value" in node:
        return node["value"]
    return node


_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
_DASH_RE = re.compile(r"[‐‑‒–—―−]")


@dataclass
class ChangeAnalysis:
    changes: list[FieldChange] = field(default_factory=list)
    ignored_page_only_fields: list[str] = field(default_factory=list)
    ignored_unchanged_fields: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            "changed_fields": [change.field_key for change in self.changes],
            "ignored_page_only_fields": self.ignored_page_only_fields,
            "ignored_unchanged_fields": self.ignored_unchanged_fields,
            "reason": "mapped_field_changes_detected" if self.changes else "no_mapped_field_changes",
        }


def canonicalize_for_diff(field_key: str, value: Any, path: str = "") -> Any:
    """Conservatively normalize parsed values before comparing them.

    This intentionally does not remove identifier prefixes or apply regex to raw
    JSON. Business transformations belong to extraction rules, not change detection.
    """
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    if field_key in {"InvoiceDate", "DueDate", "TaxPointDate"}:
        normalized = _DASH_RE.sub("-", normalized)
        normalized = _WHITESPACE_RE.sub(" ", normalized)
    elif field_key in {"InvoiceCurrency", "CurrencyCode"}:
        normalized = normalized.upper()
    elif field_key in {"VendorName", "SupplierName", "BuyerName"}:
        normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized


def _walk(a: Any, b: Any, path: str, key: str, out: list[FieldChange], analysis: ChangeAnalysis | None = None) -> None:
    if isinstance(a, list) or isinstance(b, list):
        aa, bb = a if isinstance(a, list) else [], b if isinstance(b, list) else []
        for i in range(max(len(aa), len(bb))):
            _walk(aa[i] if i < len(aa) else None, bb[i] if i < len(bb) else None,
                  f"{path}[{i}]", key, out, analysis)
        return
    if isinstance(a, dict) and isinstance(b, dict) and "value" not in a and "value" not in b:
        for child in sorted(set(a) | set(b)):
            if child != "Page":
                _walk(a.get(child), b.get(child), f"{path}.{child}" if path else child, child, out, analysis)
        return
    av, bv = _value(a), _value(b)
    normalized_av = canonicalize_for_diff(key, av, path)
    normalized_bv = canonicalize_for_diff(key, bv, path)
    if normalized_av != normalized_bv:
        out.append(FieldChange(
            field_key=key, path=path, old_value=av, new_value=bv,
            normalized_old_value=normalized_av, normalized_new_value=normalized_bv,
            comparison_method="structured_regex_normalization",
            change_reason="value_changed",
        ))
    elif analysis is not None:
        field_path = path or key
        if isinstance(a, dict) and isinstance(b, dict) and a.get("Page") != b.get("Page"):
            analysis.ignored_page_only_fields.append(field_path)
        else:
            analysis.ignored_unchanged_fields.append(field_path)


def find_changes(invoice: dict[str, Any], final: dict[str, Any]) -> list[FieldChange]:
    return analyze_changes(invoice, final).changes


def analyze_changes(invoice: dict[str, Any], final: dict[str, Any]) -> ChangeAnalysis:
    analysis = ChangeAnalysis()
    _walk(invoice, final, "", "", analysis.changes, analysis)
    analysis.changes = [change for change in analysis.changes if change.field_key]
    return analysis
