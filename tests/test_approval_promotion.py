import json

from app.audit import AuditRepository
from app.models import PromoteBatchRequest, RuleRecord
from app.rule_repository import RuleRepository
from app.service import UpdateRulesService


def _candidate(audit, candidate_id, *, field_key="InvoiceCurrency", rule_id=9, confidence="normal"):
    audit.append(
        rule_id=rule_id,
        field_key=field_key,
        path=field_key,
        old_value="USD",
        new_value="INR",
        sentence="Extract the explicitly labeled invoice currency.",
        status="preview",
        request_id="req-1",
        strategy="generative",
        candidate_id=candidate_id,
        metadata={
            "rule_version": "v1",
            "promotion_eligible": True,
            "persistence_status": "awaiting_approval",
            "confidence": confidence,
        },
    )


def _service(tmp_path):
    rules_path = tmp_path / "rules.json"
    audit_path = tmp_path / "audit.jsonl"
    rules_path.write_text(json.dumps([
        RuleRecord(ID=9, FIELD_KEY="InvoiceCurrency", DISPLAY_LABEL="Currency",
                   SHORT_RULE="currency", DETAILED_RULE=["Read the currency."]).model_dump()
    ]), encoding="utf-8")
    audit = AuditRepository(str(audit_path))
    return UpdateRulesService(None, RuleRepository(str(rules_path)), audit), audit, rules_path


def test_generation_candidates_are_not_persisted_until_batch_approval(tmp_path):
    service, audit, rules_path = _service(tmp_path)
    before = rules_path.read_text(encoding="utf-8")
    _candidate(audit, "candidate-1")

    result = service.promote_batch(PromoteBatchRequest(
        candidate_ids=["candidate-1"], expected_rule_version="v1", confirm=True
    ))

    assert result.status == "promoted"
    assert rules_path.read_text(encoding="utf-8") != before
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    assert rules[0]["DETAILED_RULE"][-1] == "Extract the explicitly labeled invoice currency."


def test_batch_rejects_low_confidence_without_writing(tmp_path):
    service, audit, rules_path = _service(tmp_path)
    _candidate(audit, "candidate-low", confidence="limited")
    before = rules_path.read_text(encoding="utf-8")

    result = service.promote_batch(PromoteBatchRequest(
        candidate_ids=["candidate-low"], expected_rule_version="v1", confirm=True
    ))

    assert result.status == "rejected"
    assert "low-confidence" in (result.reason or "")
    assert rules_path.read_text(encoding="utf-8") == before


def test_batch_validation_is_atomic(tmp_path):
    service, audit, rules_path = _service(tmp_path)
    _candidate(audit, "candidate-valid")
    _candidate(audit, "candidate-invalid", field_key="MissingField", rule_id=999)
    before = rules_path.read_text(encoding="utf-8")

    result = service.promote_batch(PromoteBatchRequest(
        candidate_ids=["candidate-valid", "candidate-invalid"],
        expected_rule_version="v1", confirm=True,
    ))

    assert result.status == "rejected"
    assert rules_path.read_text(encoding="utf-8") == before
