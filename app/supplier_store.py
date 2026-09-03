from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def supplier_key(value: Any) -> str:
    text = " ".join(str(value or "").casefold().split())
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")[:80] or "unknown_supplier"


class SupplierRuleStore:
    def __init__(self, root: str | Path = "data/rules/suppliers"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> dict[str, Any]:
        path = self.root / f"{supplier_key(key)}.json"
        if not path.exists():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}

    def resolve(self, global_rules: dict[str, Any], key: str) -> dict[str, Any]:
        merged = dict(global_rules)
        for field, value in self.load(key).items():
            merged[field] = {**(merged.get(field) or {}), **(value or {})} if isinstance(value, dict) else value
        return merged

    def save_field(self, key: str, field_key: str, rule: dict[str, Any]) -> None:
        """Persist only an explicitly promoted supplier overlay."""
        path = self.root / f"{supplier_key(key)}.json"
        current = self.load(key)
        current[field_key] = rule
        path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
