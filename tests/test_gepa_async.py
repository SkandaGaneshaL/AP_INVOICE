from types import SimpleNamespace

import gepa

from app.evaluation import ExtractionEvaluator
from app.gepa_jobs import GepaJobStore
from app.models import RuleGenerationContext
from app.strategies import GepaRuleOptimizer


class _Executor:
    def extract_full(self, **kwargs):
        return {"InvoiceCurrency": {"value": "INR", "Page": 1}}


def test_job_store_transitions_and_returns_progress():
    store = GepaJobStore(ttl_seconds=60)
    job_id = store.create({"iteration": 0, "max_iterations": 2})
    store.mark_running(job_id)
    store.update_progress(job_id, iteration=1, metric_calls=3)
    current = store.get(job_id)
    assert current["status"] == "running"
    assert current["progress"]["metric_calls"] == 3
    store.complete(job_id, {"strategy": "gepa"})
    assert store.get(job_id)["status"] == "completed"


def test_gepa_optimizer_receives_iteration_and_timeout_stoppers(monkeypatch):
    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            best_candidate={"extraction_instruction": "Extract the invoice currency."},
            detailed_results=SimpleNamespace(total_metric_calls=1, best_idx=0, candidates=[]),
        )

    monkeypatch.setattr(gepa, "optimize", fake_optimize)
    context = RuleGenerationContext(
        field_key="InvoiceCurrency", field_path="InvoiceCurrency", short_rule="Extract invoice currency.",
        old_value="USD", new_value="INR", final_response={"InvoiceCurrency": {"value": "INR", "Page": 1}},
        document_bytes=b"pdf",
    )
    optimizer = GepaRuleOptimizer(
        fallback_generator=None,
        evaluator=ExtractionEvaluator(_Executor()),
        max_iterations=2,
        max_metric_calls=5,
        timeout_seconds=120,
    )
    candidate = optimizer.generate_candidate(context)
    assert candidate.sentence == "Extract the invoice currency."
    assert len(captured["stop_callbacks"]) == 2
    assert captured["max_metric_calls"] == 5
