import json

from app.update_jobs import RuleUpdateJobStore


def test_update_job_events_are_ordered_and_resume_from_last_event():
    store = RuleUpdateJobStore(ttl_seconds=60)
    job_id = store.create(reasoning={"supported": True, "requested_effort": "high"})
    store.mark_normal_running(job_id, total_fields=1)
    store.publish(job_id, "decision_summary", {
        "field_key": "InvoiceNumber",
        "summary": "Selected the explicit field label.",
    })
    store.complete_normal(job_id, {"strategy": "generative", "changes": []})

    events = list(store.stream_events(job_id))
    assert [event["event"] for event in events] == [
        "job_queued", "normal_generation_started", "reasoning_status",
        "decision_summary", "completed",
    ]
    resumed = list(store.stream_events(job_id, last_event_id=3))
    assert [event["event"] for event in resumed] == ["decision_summary", "completed"]


def test_update_job_events_filter_sensitive_fields():
    store = RuleUpdateJobStore(ttl_seconds=60)
    job_id = store.create()
    store.publish(job_id, "diagnostic", {
        "prompt": "private prompt",
        "chain_of_thought": "private reasoning",
        "safe": "visible",
    })
    store.complete_normal(job_id, {"strategy": "generative", "changes": []})
    event = next(event for event in store.stream_events(job_id, last_event_id=1)
                 if event["event"] == "diagnostic")
    encoded = json.dumps(event["data"])
    assert "private prompt" not in encoded
    assert "private reasoning" not in encoded
    assert event["data"]["safe"] == "visible"


def test_completed_job_stream_terminates_after_completion():
    store = RuleUpdateJobStore(ttl_seconds=60)
    job_id = store.create()
    store.complete_normal(job_id, {"strategy": "generative", "changes": []})
    events = list(store.stream_events(job_id))
    assert events[-1]["event"] == "completed"
