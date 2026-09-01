import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from .models import RuleRecord
from dotenv import load_dotenv

load_dotenv()


class RuleRepository:
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.getenv("RULES_FILE", "data/extraction_rules.json"))

    def load(self) -> list[RuleRecord]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        records = raw if isinstance(raw, list) else raw.get("rules", [])
        rules = [RuleRecord.model_validate(x) for x in records]
        keys = [r.FIELD_KEY for r in rules]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate FIELD_KEY found in rules file")
        return rules

    def save(self, rules: list[RuleRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            backup = self.path.with_suffix(self.path.suffix + "." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".bak")
            shutil.copy2(self.path, backup)
        fd, temp_name = tempfile.mkstemp(prefix="rules-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump([r.model_dump() for r in rules], handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


class InMemoryRuleRepository:
    """Read-only rule snapshot used by candidate evaluation."""

    def __init__(self, rules: list[RuleRecord]):
        self._rules = [rule.model_copy(deep=True) for rule in rules]

    def load(self) -> list[RuleRecord]:
        return [rule.model_copy(deep=True) for rule in self._rules]
