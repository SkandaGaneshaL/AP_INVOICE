from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
import json


@dataclass
class _UpdateJob:
    job_id: str
    status: str = "queued"
    phase: str = "queued"
    normal_status: str = "queued"
    gepa_status: str = "disabled"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    progress: dict[str, Any] = field(default_factory=dict)
    normal_result: Any = None
    requested_config: dict[str, Any] = field(default_factory=dict)
    effective_config: dict[str, Any] = field(default_factory=lambda: {
        "gepa_enabled": False,
        "reason": "GEPA is detached from the active application",
    })
    usage: dict[str, Any] = field(default_factory=dict)
    sentence_generation_usage: dict[str, Any] | None = None
    extraction_usage: dict[str, Any] | None = None
    termination: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    requested_model: str = "gpt-oss-20b"
    effective_model: str = "openai.gpt-oss-20b"
    extraction_model: str = "google.gemini-2.5-flash"
    change_detection: dict[str, Any] = field(default_factory=dict)
    oci_sentence_generation_called: bool = False
    oci_sentence_generation_call_count: int = 0
    reasoning: dict[str, Any] = field(default_factory=dict)
    reasoning_ui_contract_version: str = "safe-summary-v1"
    events: list[dict[str, Any]] = field(default_factory=list)
    next_event_id: int = 1


class RuleUpdateJobStore:
    """Thread-safe, in-memory store for the active Generative OCI workflow."""

    def __init__(self, ttl_seconds: int = 1800):
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, _UpdateJob] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    @staticmethod
    def _safe_event_data(data: dict[str, Any]) -> dict[str, Any]:
        """Keep SSE events metadata-only and bounded."""
        blocked = {"thought", "chain_of_thought", "reasoning_trace", "prompt", "pdf", "document_bytes"}

        def clean(value):
            if isinstance(value, dict):
                return {str(key): clean(item) for key, item in value.items()
                        if str(key).lower() not in blocked}
            if isinstance(value, list):
                return [clean(item) for item in value[:20]]
            if isinstance(value, str):
                return value[:500]
            return value

        return clean(dict(data))

    def publish(self, job_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
        with self._condition:
            job = self._get(job_id)
            event = {
                "id": job.next_event_id,
                "event": event_type,
                "data": self._safe_event_data({"job_id": job_id, **(data or {})}),
                "created_at": time.time(),
            }
            job.next_event_id += 1
            job.events.append(event)
            del job.events[:-100]
            job.updated_at = time.time()
            self._condition.notify_all()

    def stream_events(self, job_id: str, last_event_id: int = 0):
        """Yield SSE-ready events until the job reaches a terminal state."""
        cursor = max(0, last_event_id)
        while True:
            with self._condition:
                job = self._get(job_id)
                pending = [event for event in job.events if event["id"] > cursor]
                terminal = job.status in {"completed", "failed", "expired"}
                if not pending and not terminal:
                    notified = self._condition.wait(timeout=15)
                    if notified:
                        continue
                    pending = [{
                        "id": cursor,
                        "event": "heartbeat",
                        "data": {"job_id": job_id, "status": job.status},
                        "created_at": time.time(),
                    }]
                if not pending and terminal:
                    return
                pending = list(pending)
            for event in pending:
                cursor = event["id"]
                yield event

    def _purge(self) -> None:
        now = time.time()
        for job_id, job in list(self._jobs.items()):
            age = now - job.updated_at
            if age > self.ttl_seconds * 2:
                del self._jobs[job_id]
            elif age > self.ttl_seconds:
                job.status = "expired"
                job.phase = "expired"
                job.normal_result = None
                job.error = {"code": "UPDATE_JOB_EXPIRED", "message": "Update job result expired"}

    def _get(self, job_id: str) -> _UpdateJob:
        self._purge()
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def create(self, progress=None, *, requested_config=None, effective_config=None,
               requested_model="gpt-oss-20b", effective_model="openai.gpt-oss-20b",
               extraction_model="google.gemini-2.5-flash", change_detection=None, reasoning=None) -> str:
        with self._lock:
            self._purge()
            job_id = str(uuid.uuid4())
            self._jobs[job_id] = _UpdateJob(
                job_id=job_id,
                progress=dict(progress or {}),
                requested_config=dict(requested_config or {}),
                effective_config=dict(effective_config or {
                    "gepa_enabled": False,
                    "reason": "GEPA is detached from the active application",
                }),
                requested_model=requested_model,
                effective_model=effective_model,
                extraction_model=extraction_model,
                change_detection=dict(change_detection or {}),
                reasoning=dict(reasoning or {}),
                reasoning_ui_contract_version="safe-summary-v1",
            )
            self.publish(job_id, "job_queued", {
                "status": "queued",
                "reasoning": dict(reasoning or {}),
            })
            return job_id

    def mark_normal_running(self, job_id: str, *, total_fields: int) -> None:
        with self._lock:
            job = self._get(job_id)
            job.status = "running"
            job.phase = "normal_running"
            job.normal_status = "running"
            job.progress.update(normal_completed_fields=0, normal_total_fields=total_fields)
            job.updated_at = time.time()
            self.publish(job_id, "normal_generation_started", {
                "status": "running", "total_fields": total_fields,
            })
            if job.reasoning:
                self.publish(job_id, "reasoning_status", job.reasoning)

    def update_normal(self, job_id: str, result: Any, *, completed_fields: int, total_fields: int) -> None:
        with self._lock:
            job = self._get(job_id)
            job.normal_result = result
            job.normal_status = "running" if completed_fields < total_fields else "completed"
            job.progress.update(normal_completed_fields=completed_fields, normal_total_fields=total_fields)
            if isinstance(result, dict) and result.get("usage"):
                job.usage = {"normal": result["usage"]}
                job.sentence_generation_usage = dict(result["usage"])
            if isinstance(result, dict):
                metadata = result.get("metadata") or {}
                job.oci_sentence_generation_called = bool(metadata.get("oci_sentence_generation_called", job.oci_sentence_generation_called))
                job.oci_sentence_generation_call_count = int(metadata.get("oci_sentence_generation_call_count", job.oci_sentence_generation_call_count))
                if metadata.get("reasoning"):
                    job.reasoning = dict(metadata["reasoning"])
            job.updated_at = time.time()
            for change in (result.get("changes", []) if isinstance(result, dict) else []):
                field_event = {
                    "field_key": change.get("FIELD_KEY"),
                    "status": change.get("status"),
                    "decision_summary": change.get("decision_summary"),
                    "reasoning": {
                        "requested_effort": change.get("reasoning_effort_requested"),
                        "effective_effort": change.get("reasoning_effort_effective"),
                        "supported": change.get("reasoning_supported", False),
                        "hidden_reasoning_exposed": False,
                        "reasoning_mode": "safe_decision_summary" if change.get("reasoning_supported") else "not_available",
                    },
                }
                self.publish(job_id, "field_generation_completed", field_event)
                if change.get("decision_summary"):
                    self.publish(job_id, "decision_summary", {
                        "field_key": change.get("FIELD_KEY"),
                        "summary": change["decision_summary"],
                    })
            if isinstance(result, dict) and result.get("usage"):
                self.publish(job_id, "token_usage", {"usage": result["usage"]})

    def complete_normal(self, job_id: str, result: Any, *, termination=None, usage=None) -> None:
        with self._lock:
            job = self._get(job_id)
            job.status = "completed"
            job.phase = "normal_completed"
            job.normal_status = "completed"
            job.gepa_status = "disabled"
            job.normal_result = result
            job.termination = dict(termination or {"reason": "gepa_disabled"})
            if usage is not None:
                # Keep the legacy aggregate shape stable while exposing the
                # convenient sentence_generation_usage alias separately.
                if isinstance(usage, dict) and isinstance(usage.get("normal"), dict):
                    job.usage = dict(usage)
                    normal_usage = usage["normal"]
                else:
                    normal_usage = dict(usage) if isinstance(usage, dict) else None
                    job.usage = {"normal": normal_usage} if normal_usage is not None else {}
                if normal_usage is not None:
                    job.sentence_generation_usage = dict(normal_usage)
            if isinstance(result, dict):
                metadata = result.get("metadata") or {}
                job.oci_sentence_generation_called = bool(metadata.get("oci_sentence_generation_called", job.oci_sentence_generation_called))
                job.oci_sentence_generation_call_count = int(metadata.get("oci_sentence_generation_call_count", job.oci_sentence_generation_call_count))
            job.updated_at = time.time()
            self.publish(job_id, "completed", {"status": "completed"})

    def fail(self, job_id: str, error: dict[str, Any]) -> None:
        with self._lock:
            job = self._get(job_id)
            job.status = "failed"
            job.phase = "failed"
            job.normal_status = "failed" if job.normal_status != "completed" else job.normal_status
            job.gepa_status = "disabled"
            job.error = dict(error)
            job.updated_at = time.time()
            self.publish(job_id, "generation_failed", {"status": "failed", "error": error})
            self.publish(job_id, "completed", {"status": "failed"})

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._get(job_id)
            return {
                "job_id": job.job_id,
                "status": job.status,
                "phase": job.phase,
                "normal_status": job.normal_status,
                "gepa_status": job.gepa_status,
                "normal_result": job.normal_result,
                "gepa_result": None,
                "progress": dict(job.progress),
                "requested_config": dict(job.requested_config),
                "effective_config": dict(job.effective_config),
                "termination": dict(job.termination),
                "usage": dict(job.usage),
                "sentence_generation_usage": dict(job.sentence_generation_usage) if job.sentence_generation_usage else None,
                "extraction_usage": dict(job.extraction_usage) if job.extraction_usage else None,
                "error": dict(job.error) if job.error else None,
                "requested_model": job.requested_model,
                "effective_model": job.effective_model,
                "extraction_model": job.extraction_model,
                "change_detection": dict(job.change_detection),
                "oci_sentence_generation_called": job.oci_sentence_generation_called,
                "oci_sentence_generation_call_count": job.oci_sentence_generation_call_count,
                "reasoning": dict(job.reasoning),
                "reasoning_ui_contract_version": job.reasoning_ui_contract_version,
            }
