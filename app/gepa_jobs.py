from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Job:
    job_id: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    progress: dict[str, Any] = field(default_factory=dict)
    phase: str = "queued"
    normal_status: str = "queued"
    gepa_status: str = "waiting_for_normal"
    normal_result: Any = None
    gepa_result: Any = None
    requested_config: dict[str, Any] = field(default_factory=dict)
    effective_config: dict[str, Any] = field(default_factory=dict)
    termination: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    strategy: Any = None  # legacy alias for the GEPA result
    error: dict[str, Any] | None = None


class GepaJobStore:
    def __init__(self, ttl_seconds: int = 1800):
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()

    def _purge(self) -> None:
        now = time.time()
        cutoff = now - self.ttl_seconds
        hard_cutoff = now - (self.ttl_seconds * 2)
        for job_id, job in list(self._jobs.items()):
            if job.updated_at < hard_cutoff:
                del self._jobs[job_id]
            elif job.updated_at < cutoff:
                job.status = "expired"
                job.strategy = None
                job.normal_result = None
                job.gepa_result = None
                job.error = {"code": "GEPA_JOB_EXPIRED", "message": "GEPA job result expired"}

    def create(self, progress: dict[str, Any] | None = None, *, requested_config=None, effective_config=None) -> str:
        with self._lock:
            self._purge()
            job_id = str(uuid.uuid4())
            self._jobs[job_id] = _Job(job_id=job_id, progress=progress or {},
                                     requested_config=dict(requested_config or {}),
                                     effective_config=dict(effective_config or {}))
            return job_id

    def _get(self, job_id: str) -> _Job:
        self._purge()
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._get(job_id)
            job.status = "running"
            job.phase = "gepa_running"
            job.gepa_status = "running"
            job.updated_at = time.time()

    def mark_normal_running(self, job_id: str, *, total_fields: int) -> None:
        with self._lock:
            job = self._get(job_id)
            job.status = "running"
            job.phase = "normal_running"
            job.normal_status = "running"
            job.progress.update(normal_completed_fields=0, normal_total_fields=total_fields)
            job.updated_at = time.time()

    def update_normal(self, job_id: str, result: Any, *, completed_fields: int, total_fields: int) -> None:
        with self._lock:
            job = self._get(job_id)
            job.normal_result = result
            if isinstance(result, dict) and result.get("usage"):
                job.usage = {"normal": result["usage"]}
            job.normal_status = "running" if completed_fields < total_fields else "completed"
            job.progress.update(normal_completed_fields=completed_fields, normal_total_fields=total_fields)
            job.updated_at = time.time()

    def mark_gepa_queued(self, job_id: str) -> None:
        with self._lock:
            job = self._get(job_id)
            job.phase = "gepa_queued"
            job.gepa_status = "queued"
            job.updated_at = time.time()

    def update_progress(self, job_id: str, **progress: Any) -> None:
        with self._lock:
            job = self._get(job_id)
            job.progress.update(progress)
            job.updated_at = time.time()

    def complete(self, job_id: str, strategy: Any) -> None:
        with self._lock:
            job = self._get(job_id)
            job.status = "completed"
            job.strategy = strategy
            job.gepa_result = strategy
            job.phase = "completed"
            job.gepa_status = "completed"
            job.updated_at = time.time()

    def complete_update(self, job_id: str, normal_result: Any, gepa_result: Any = None, *, termination=None, usage=None) -> None:
        with self._lock:
            job = self._get(job_id)
            job.status = "completed"
            job.phase = "completed"
            job.normal_status = "completed"
            job.gepa_status = "completed" if gepa_result is not None else "disabled"
            job.normal_result = normal_result
            job.gepa_result = gepa_result
            job.strategy = gepa_result
            job.termination = dict(termination or {})
            if usage is not None:
                job.usage = dict(usage)
            job.updated_at = time.time()

    def fail(self, job_id: str, error: dict[str, Any]) -> None:
        with self._lock:
            job = self._get(job_id)
            job.status = "failed"
            job.phase = "failed"
            if job.normal_status != "completed":
                job.normal_status = "failed"
            if job.gepa_status in {"waiting_for_normal", "queued", "running"}:
                job.gepa_status = "failed"
            job.error = error
            job.updated_at = time.time()

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._get(job_id)
            return {
                "job_id": job.job_id,
                "status": job.status,
                "phase": job.phase,
                "normal_status": job.normal_status,
                "gepa_status": job.gepa_status,
                "progress": dict(job.progress),
                "normal_result": job.normal_result,
                "gepa_result": job.gepa_result,
                "strategy": job.strategy,
                "requested_config": dict(job.requested_config),
                "effective_config": dict(job.effective_config),
                "termination": dict(job.termination),
                "usage": dict(job.usage),
                "error": dict(job.error) if job.error else None,
            }
