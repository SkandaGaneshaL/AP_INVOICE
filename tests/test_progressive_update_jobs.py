from app.gepa_jobs import GepaJobStore
from app.models import GepaRunConfig


def test_job_store_preserves_requested_and_effective_gepa_config():
    config = GepaRunConfig(max_iterations=10, max_metric_calls=50, timeout_seconds=120)
    store = GepaJobStore(ttl_seconds=60)
    job_id = store.create(
        {"normal_total_fields": 3, "normal_completed_fields": 0},
        requested_config=config.model_dump(),
        effective_config=config.model_dump(),
    )

    job = store.get(job_id)
    assert job["requested_config"]["max_iterations"] == 10
    assert job["requested_config"]["max_metric_calls"] == 50
    assert job["effective_config"]["timeout_seconds"] == 120


def test_job_store_keeps_normal_result_when_gepa_fails():
    store = GepaJobStore(ttl_seconds=60)
    job_id = store.create({"normal_total_fields": 1})
    normal = {"strategy": "generative", "changes": [{"FIELD_KEY": "InvoiceNumber"}]}
    store.mark_normal_running(job_id, total_fields=1)
    store.update_normal(job_id, normal, completed_fields=1, total_fields=1)
    store.mark_gepa_queued(job_id)
    store.mark_running(job_id)
    store.fail(job_id, {"code": "GEPA_OPTIMIZATION_FAILED"})

    job = store.get(job_id)
    assert job["normal_result"] == normal
    assert job["normal_status"] == "completed"
    assert job["gepa_status"] == "failed"
    assert job["error"]["code"] == "GEPA_OPTIMIZATION_FAILED"

