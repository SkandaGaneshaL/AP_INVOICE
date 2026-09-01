from __future__ import annotations

import json
from typing import Any

from .models import RuleRecord


REFERENCE_BASIC_PROMPT = """You are an intelligent document parser that extracts structured information from invoices.

Invoices may be printed, handwritten, scanned, or multilingual. Understand labels in other languages while preserving the value exactly as shown. If a value contains non-Latin text, preserve the original and append a readable English equivalent using the format '<original> || <English equivalent>'. Do not translate proper names, identifiers, codes, tax IDs, part numbers, or brand names.

Extract only information explicitly present in the invoice. Do not infer, calculate, normalize, or guess values unless a field rule explicitly permits it. Return null for missing, unreadable, or not-present values. Preserve source formatting, punctuation, codes, and numbers where the field rule requires it. For address fields, preserve the complete address block and do not omit or reorder its components.

Return ONLY one valid JSON object. Do not return Markdown, explanations, reasoning, headings, or additional text.
"""

LINE_ITEM_RULES = """### LineItems rules
- Extract one row for each physical item or service row in the itemized table.
- Do not create rows from VAT, tax, GST, subtotal, total, grand total, balance, or summary rows.
- Do not shift values between rows.
- Do not calculate missing quantities, rates, amounts, tax rates, or tax amounts.
- Preserve TaxRate and TaxAmount when they are aligned with the same item row.
"""


class InvoiceExtractionPromptBuilder:
    """Builds the complete extraction prompt from local extraction rules."""

    def build(self, rules: list[RuleRecord], *, candidate_instruction: str | None = None) -> str:
        if not rules:
            raise ValueError("No extraction rules are configured")

        field_lines: list[str] = []
        rule_blocks: list[str] = []
        output: dict[str, Any] = {}
        for rule in rules:
            field_lines.append(f"- {rule.FIELD_KEY} ({rule.DISPLAY_LABEL or rule.FIELD_KEY})")
            detailed = [str(item).strip() for item in rule.DETAILED_RULE if str(item).strip()]
            rule_blocks.append(
                "\n".join(
                    [
                        f"### {rule.FIELD_KEY} — {rule.DISPLAY_LABEL or rule.FIELD_KEY}",
                        f"Short rule: {rule.SHORT_RULE or 'Extract the explicitly labeled value.'}",
                        *(f"- {item}" for item in detailed),
                    ]
                )
            )
            output[rule.FIELD_KEY] = [] if self.is_list_rule(rule) else {"value": None, "Page": None}

        candidate = ""
        if candidate_instruction and candidate_instruction.strip():
            candidate = (
                "\n### Candidate extraction instruction\n"
                "Apply this instruction only when it is consistent with the field rules. "
                "Never memorize a value from one invoice as a universal value.\n"
                f"{candidate_instruction.strip()}\n"
            )

        return (
            f"{REFERENCE_BASIC_PROMPT}\n"
            "### Task\nExtract every configured field below:\n"
            f"{chr(10).join(field_lines)}\n\n"
            "### Field extraction rules\n"
            f"{chr(10).join(rule_blocks)}\n\n"
            f"{LINE_ITEM_RULES if any(r.FIELD_KEY == 'LineItems' for r in rules) else ''}"
            f"{candidate}\n"
            "### Output contract\n"
            "Return exactly one top-level object with every configured field. "
            "Every scalar field must contain exactly `value` and `Page`. `Page` must be an integer or null. "
            "Use null for both values when the field is absent. Do not add top-level or nested fields.\n"
            f"Example shape:\n{json.dumps(output, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def is_list_rule(rule: RuleRecord) -> bool:
        text = " ".join([rule.FIELD_KEY, rule.DISPLAY_LABEL, rule.SHORT_RULE, *rule.DETAILED_RULE]).lower()
        return (
            "line item" in text
            or "each invoice line" in text
            or "multiple line" in text
            or rule.FIELD_KEY.lower().startswith("lineitem")
        )
