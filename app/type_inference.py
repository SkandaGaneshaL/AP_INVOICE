from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any


class FieldType(str, Enum):
    MONEY = "MONEY"
    DATE = "DATE"
    IDENTIFIER = "IDENTIFIER"
    ENUM = "ENUM"
    ADDRESS = "ADDRESS"
    ORG = "ORG"
    PERSON = "PERSON"
    LANGUAGE_SCRIPT = "LANGUAGE_SCRIPT"
    TEXT = "TEXT"
    NULL = "NULL"


InferredType = FieldType

_DATE = re.compile(r"^\d{1,4}[/-]\d{1,2}[/-]\d{1,4}$")
_MONEY = re.compile(r"^[\s$€£₹A-Z]*[-+]?\d[\d\s,.' ]*[.,]\d{2}$", re.I)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ./_-]*$")
_CURRENCY = re.compile(r"^(?:[A-Z]{3}|[$€£₹])$")


def _s(value: Any) -> str:
    return "" if value is None else str(value).strip()


def infer_field_type(field_key: str | None, display_label: str | None, old_value: Any,
                     new_value: Any, evidence_text: str | None, existing_program: Any = None) -> FieldType:
    if existing_program is not None and getattr(existing_program, "type", None):
        try:
            return FieldType(str(getattr(existing_program, "type")).upper())
        except ValueError:
            pass
    values = [_s(old_value), _s(new_value), _s(display_label), _s(evidence_text)]
    text = " ".join(values).casefold()
    if old_value is None and new_value is None:
        return FieldType.NULL
    if any(_MONEY.fullmatch(value) for value in values[:2]) or any(token in text for token in ("amount", "price", "total", "tax", "money")) and re.search(r"\d[,.]\d{2}", text):
        return FieldType.MONEY
    if any(_DATE.fullmatch(value) for value in values[:2]) or any(token in text for token in ("date", "due", "issued")) and re.search(r"\d[/-]\d", text):
        return FieldType.DATE
    if any(_IDENTIFIER.fullmatch(value) and any(char.isdigit() for char in value) for value in values[:2]):
        return FieldType.IDENTIFIER
    if _CURRENCY.fullmatch(_s(new_value).upper()) or any(token in text for token in ("currency", "country", "status", "type", "terms", "method")):
        return FieldType.ENUM
    if "\n" in _s(old_value) or any(token in text for token in ("address", "street", "city", "state", "postal", "bill to", "ship to", "remit")):
        return FieldType.ADDRESS
    if any(token in text for token in ("vendor", "supplier", "company", "organization", "customer")) or re.search(r"\b(inc|llc|ltd|gmbh|corp|pvt|llp)\b", text):
        return FieldType.ORG
    if any(token in text for token in ("person", "contact", "employee")):
        return FieldType.PERSON
    if any(ord(char) > 127 for value in values for char in value):
        return FieldType.LANGUAGE_SCRIPT
    return FieldType.TEXT


def infer_type(old_value: Any, new_value: Any, label: str = "", evidence_text: str = "") -> FieldType:
    return infer_field_type(None, label, old_value, new_value, evidence_text)


def parse_number(value: Any) -> Decimal | None:
    text = _s(value)
    if not text:
        return None
    cleaned = re.sub(r"[^0-9+\-.,]", "", text)
    if not cleaned or cleaned.count("-") > 1 or cleaned.count("+") > 1:
        return None
    # Treat the final separator as decimal only when exactly two digits follow it;
    # all other separators are grouping punctuation.
    if "." in cleaned or "," in cleaned:
        last = max(cleaned.rfind("."), cleaned.rfind(","))
        tail = cleaned[last + 1:]
        if len(tail) == 2 and cleaned[:last].replace(",", "").replace(".", "").isdigit():
            cleaned = cleaned[:last].replace(",", "").replace(".", "") + "." + tail
        else:
            cleaned = cleaned.replace(",", "").replace(".", "")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def parse_date(value: Any) -> tuple[datetime, str] | None:
    text = _s(value)
    formats = ("%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%m.%d.%Y")
    matches = []
    for pattern in formats:
        try:
            matches.append((datetime.strptime(text, pattern), pattern))
        except ValueError:
            continue
    if not matches:
        return None
    unique_dates = {item[0] for item in matches}
    if len(unique_dates) > 1:
        return None
    return matches[0]


def identifier_core(value: Any) -> str | None:
    text = _s(value)
    if not text or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _./-]*", text):
        return None
    compact = re.sub(r"[^A-Za-z0-9]", "", text)
    if not compact or not any(char.isdigit() for char in compact):
        return None
    return compact
