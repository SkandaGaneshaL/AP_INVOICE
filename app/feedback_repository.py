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
        """Select same-field examples while preferring matching failure types and layouts."""
        ranked: list[tuple[int, dict[str, Any]]] = []
        for row in history:
            if row.get("field_key") != packet.field_key:
                continue
            score = 0
            if row.get("failure_type") == packet.failure_type:
                score += 4
            if row.get("layout_signature") and row.get("layout_signature") == row.get("current_layout_signature"):
                score += 2
            ranked.append((score, row))
        ranked.sort(key=lambda item: item[0], reverse=True)
        result: list[DemonstrationExample] = []
        for _, row in ranked[:max(0, limit)]:
            try:
                result.append(DemonstrationExample.model_validate(row))
            except Exception:
                continue
        return result
