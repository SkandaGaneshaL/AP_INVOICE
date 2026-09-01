import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


class AuditRepository:
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.getenv("AUDIT_FILE", "data/rule_update_audit.jsonl"))

    def append(self, *, rule_id, field_key, path, old_value, new_value, sentence, status, request_id,
               strategy="generative", evaluation_score=None, evaluation_feedback=None,
               selected_for_persistence=False, promotion_status="not_promoted", candidate_id=None,
               prompt_hash=None, metadata=None, model=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "ID": rule_id,
                  "FIELD_KEY": field_key, "path": path, "old_value_hash": _hash(old_value),
                  "new_value_hash": _hash(new_value), "generated_sentence": sentence,
                  "status": status, "model": model or (metadata or {}).get("model") or os.getenv("OCI_MODEL_ID", "google.gemini-2.5-flash"),
                  "oci_request_id": request_id, "prompt_version": os.getenv("GEPA_PROMPT_VERSION", "v2"),
                  "strategy": strategy, "evaluation_score": evaluation_score,
                  "evaluation_feedback": evaluation_feedback,
                  "selected_for_persistence": selected_for_persistence,
                  "promotion_status": promotion_status, "candidate_id": candidate_id,
                  "prompt_hash": prompt_hash, "metadata": metadata or {}}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def find_candidate(self, candidate_id: str):
        if not self.path.exists():
            return None
        for line in reversed(self.path.read_text(encoding="utf-8").splitlines()):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("candidate_id") == candidate_id:
                return record
        return None
