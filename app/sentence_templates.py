from __future__ import annotations

from typing import Any


TEMPLATES = {
    "numeric_thousands_canonicalize": "Parse the labeled {display_label} as a number: ignore grouping separators and currency symbols; keep the decimal part.",
    "money_canonicalize": "Parse the labeled {display_label} as a number: ignore grouping separators and currency symbols; keep the decimal part.",
    "parse_money": "Parse the labeled {display_label} as a number: ignore grouping separators and currency symbols; keep the decimal part.",
    "date_format_repattern": "Parse the labeled date; emit {target_pattern}.",
    "parse_date": "Parse the labeled date; emit {target_pattern}.",
    "identifier_strip_leading_alpha": "From the labeled identifier, drop a leading alphabetic prefix when the remainder is a valid identifier core; return the remainder.",
    "strip_leading_alpha_token": "From the labeled identifier, drop a leading alphabetic prefix when the remainder is a valid identifier core; return the remainder.",
    "strip_trailing_alpha": "From the labeled identifier, remove a trailing alphabetic suffix when the remaining value is a valid identifier core.",
    "collapse_ws": "Normalize whitespace in the labeled value before comparison.",
    "trim": "Trim surrounding whitespace from the labeled value before comparison.",
    "label_retarget": "Use the value aligned to {label}, not {competing_label}.",
}


def render_sentence(program: Any, display_label: str | None = None) -> str:
    transform = next((item for item in program.transform if item.op not in {"identity", "no_op"}), None)
    if transform is None:
        return program.sentence or "Extract the value from the explicitly labeled field."
    template = TEMPLATES.get(transform.op)
    if not template:
        return program.sentence or f"Extract the value from the explicitly labeled {display_label or 'field'}."
    args = dict(transform.args)
    args.setdefault("display_label", display_label or (program.select.label_aliases[0] if program.select.label_aliases else "field"))
    args.setdefault("target_pattern", program.format.target_pattern if hasattr(program, "format") else None)
    args.setdefault("label", program.select.label_aliases[0] if program.select.label_aliases else "the selected label")
    args.setdefault("competing_label", program.disambiguate.ignore[0] if program.disambiguate.ignore else "unrelated fields")
    return template.format(**args)
