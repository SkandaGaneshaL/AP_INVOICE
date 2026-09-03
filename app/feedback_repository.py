from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import DemonstrationExample, RuleFeedbackPacket


class FeedbackRepository:
    """Loads redacted/historical feedback examples without exposing raw audit values."""

    def __init__(self, path: str | None = None):
        self.path = Path(path or "data/rule_feedback.jsonl")

    def load(self, field_key: str, limit: int = 10) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("field_key") == field_key:
                rows.append(row)
        return rows[-max(0, limit):]

    def retrieve_operator_examples(self, field_type: str, failure_type: str, layout_signature: Any,
                                   label_text: str, k: int = 2) -> list[DemonstrationExample]:
        """Retrieve compact, type-compatible examples across field keys."""
        wanted_layout = getattr(layout_signature, "values", lambda: layout_signature)()
        wanted_label = str(label_text or "").casefold()
        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in self.load_all():
            if str(row.get("field_type", "")).upper() != str(field_type).upper():
                continue
            score = 0.0
            if row.get("failure_type") == failure_type: score += 5
            if row.get("layout_signature") == wanted_layout: score += 3
            labels = str(row.get("label_text", "")).casefold().split()
            score += sum(1 for token in labels if token in wanted_label)
            if row.get("promoted"): score += 0.5
            ranked.append((score, row))
        result = []
        for _, row in sorted(ranked, key=lambda item: item[0], reverse=True)[:max(0, min(k, 2))]:
            try: result.append(DemonstrationExample.model_validate(row))
            except Exception: continue
        return result

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
        return rows

    def append(self, example: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(example, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def select_demonstrations(
        packet: RuleFeedbackPacket,
        history: list[dict[str, Any]],
        limit: int,
    ) -> list[DemonstrationExample]:
        """Select at most two compact compatible examples across fields."""
        ranked: list[tuple[int, dict[str, Any]]] = []
        for row in history:
            score = 0
            if row.get("failure_type") == packet.failure_type:
                score += 4
            if row.get("layout_signature") and row.get("layout_signature") == row.get("current_layout_signature"):
                score += 2
            packet_labels = {str(item.label).casefold() for item in packet.evidence if item.label}
            row_label = str(row.get("label_text", "")).casefold()
            score += sum(1 for label in packet_labels if label and label in row_label)
            if row.get("field_key") == packet.field_key:
                score += 1
            ranked.append((score, row))
        ranked.sort(key=lambda item: item[0], reverse=True)
        result: list[DemonstrationExample] = []
        for _, row in ranked[:max(0, min(2, limit))]:
            try:
                result.append(DemonstrationExample.model_validate(row))
            except Exception:
                continue
        return result
