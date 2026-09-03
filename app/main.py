import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
import json
from .models import (ChangeResult, ExtractionResponse, PromoteBatchRequest,
                     PromoteBatchResponse, PromoteRequest, UsageSummary,
                     PromoteResponse, StrategyResult, UpdateJobCreateResponse,
                     UpdateJobStatusResponse, UpdateRequest, UpdateResponse)
from .oci_native_rule_provider import OciNativeRuleGenerator
from .model_registry import (RuleGenerationConfigurationError,
                             resolve_rule_generation_model,
                             validate_rule_generation_configuration,
                             rule_generation_settings, reasoning_config,
                             normalize_reasoning_effort)
from .service import RuleGenerationError, UpdateRulesService
from .rule_repository import InMemoryRuleRepository, RuleRepository
from .audit import AuditRepository
from .model_output import ModelOutputError
from .errors import status_code_for_error
from .executor import load_extraction_executor
from .document_store import InMemoryDocumentStore
from .oci_pdf_client import ExtractionConfigurationError
from .pdf_extractor import OciPdfExtractor, OciPdfExtractionExecutor, PdfExtractionError
from .update_jobs import RuleUpdateJobStore
from .comparator import analyze_changes
from .usage import summarize_usage, USAGE_NORMALIZER_VERSION

app = FastAPI(title="OCI Invoice Extraction Rule Updater", version="1.0.0")
logger = logging.getLogger(__name__)
REASONING_UI_CONTRACT_VERSION = "safe-summary-v1"
document_store = InMemoryDocumentStore(int(os.getenv("DOCUMENT_TTL_SECONDS", "1800")))
update_jobs = RuleUpdateJobStore(int(os.getenv("UPDATE_JOB_TTL_SECONDS", "1800")))
job_executor = ThreadPoolExecutor(max_workers=1)
normal_field_executor = ThreadPoolExecutor(max_workers=max(1, int(os.getenv("NORMAL_WORKER_COUNT", "4"))))


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_, exc: RequestValidationError):
    """Keep the reasoning selector error actionable at the API boundary."""
    if any(error.get("loc", ()) and error["loc"][-1] == "reasoning_effort" for error in exc.errors()):
        return JSONResponse(status_code=422, content={
            "detail": {
                "code": "INVALID_REASONING_EFFORT",
                "message": "Reasoning effort must be one of none, minimal, low, medium, or high",
            }
        })
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/")
def root():
    return {
        "service": "OCI Invoice Extraction Rule Updater",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "update_endpoint": "POST /v1/extraction-rules/update",
    }


@app.post("/v1/extraction-rules/update", response_model=UpdateJobCreateResponse)
def update_rules(request: UpdateRequest):
    try:
        repository = RuleRepository()
        selected_model = resolve_rule_generation_model(request.rule_generation_model)
        validate_rule_generation_configuration(selected_model)
        requested_reasoning = normalize_reasoning_effort(request.reasoning_effort)
        reasoning = reasoning_config(selected_model, requested_reasoning)
        original_rules = [rule.model_copy(deep=True) for rule in repository.load()]
        change_analysis = analyze_changes(request.invoice_payload, request.final_response)
        mapped_keys = {rule.FIELD_KEY for rule in original_rules}
        changes = [change for change in change_analysis.changes if change.field_key in mapped_keys]
        change_detection = change_analysis.model_dump()
        change_detection["changed_fields"] = [change.field_key for change in changes]
        change_detection["ignored_unmapped_fields"] = [
            change.field_key for change in change_analysis.changes if change.field_key not in mapped_keys
        ]
        if not changes:
            change_detection["reason"] = "no_mapped_field_changes"
        model_settings = rule_generation_settings(selected_model)
        effective = {
            "model": selected_model.model_id,
            "region": model_settings.get("region"),
            "serving_mode": model_settings.get("serving_mode"),
            "application_output_limit_sent": False,
            "provider_managed_output_limit": True,
            "reasoning_effort": reasoning.get("effective_effort"),
            "reason": "Normal LLM-first correction path",
            "sentence_payload_version": "correction-delta-v1",
            "intent_schema_version": "correction-intent-v1",
            "merge_schema_version": "rule-merge-v1",
            "repair_calls": 0,
        }
        job_id = update_jobs.create(
            {"normal_completed_fields": 0, "normal_total_fields": len(changes)},
        requested_config={"reasoning_effort": requested_reasoning}, effective_config=effective,
            requested_model=selected_model.key, effective_model=selected_model.model_id,
            extraction_model=os.getenv("OCI_EXTRACTION_MODEL_ID", "google.gemini-2.5-flash"),
            change_detection=change_detection,
            reasoning=reasoning,
        )
        request_snapshot = request.model_copy(deep=True)
        job_executor.submit(_run_update_job, job_id, request_snapshot, original_rules, changes)
        return UpdateJobCreateResponse(job_id=job_id, status="queued", normal_status="queued",
                                       requested_config={"reasoning_effort": requested_reasoning},
                                       effective_config=effective,
                                       requested_model=selected_model.key,
                                       effective_model=selected_model.model_id,
                                       extraction_model=os.getenv("OCI_EXTRACTION_MODEL_ID", "google.gemini-2.5-flash"),
                                       change_detection=change_detection,
                                       reasoning=reasoning,
                                       sentence_generation_usage=None,
                                       extraction_usage=None,
                                       rule_merge_usage=None,
                                       reasoning_ui_contract_version=REASONING_UI_CONTRACT_VERSION)
    except KeyError as exc:
        raise HTTPException(status_code=410, detail={"code": "DOCUMENT_EXPIRED", "message": "Please upload and extract the invoice again."}) from exc
    except RuleGenerationConfigurationError as exc:
        raise HTTPException(status_code=422, detail={
            "code": exc.code,
            "message": exc.message,
            "operation": "rule_generation",
            "model": exc.model,
            "region": exc.region,
        }) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={
            "code": "UNSUPPORTED_RULE_GENERATION_MODEL",
            "message": "Unsupported sentence-generation model",
            "reason": str(exc),
        }) from exc
    except RuleGenerationError as exc:
        logger.exception("Rule generation failed for field %s", exc.field_key)
        raise HTTPException(status_code=502, detail={
            "code": "OCI_RULE_GENERATION_FAILED",
            "message": "OCI rule generation failed",
            "reason": str(exc)[:500],
            "field_key": exc.field_key,
            "strategy": exc.strategy,
        }) from exc
    except Exception as exc:
        logger.exception("Rule update failed")
        safe_reason = str(exc)
        for secret in ("OCI_API_KEY", "PRIVATE_KEY", "private.pem"):
            safe_reason = safe_reason.replace(secret, "[redacted]")
        if isinstance(exc, ModelOutputError):
            logger.error("Safe model-output diagnostics: %s", exc.diagnostics)
            provider_limit = str(exc.finish_reason or "").upper() in {"MAX_TOKENS", "LENGTH"} or "truncated" in str(exc).lower()
            detail = {
                "code": "OCI_PROVIDER_OUTPUT_LIMIT" if provider_limit else "OCI_INVALID_MODEL_OUTPUT",
                "message": "OCI terminated the response at its provider-managed output limit" if provider_limit else "OCI returned an invalid rule-generation response",
                "reason": exc.reason,
                "finish_reason": exc.finish_reason,
                "response_text_length": exc.text_length,
                "response_prefix": exc.prefix,
                **exc.diagnostics,
            }
        else:
            detail = {
            "code": "OCI_RULE_GENERATION_FAILED",
            "message": "OCI rule generation failed",
            "reason": safe_reason[:500],
            "hint": "Check the terminal traceback, OCI_COMPARTMENT_ID, model availability, and IAM policy.",
            }
        raise HTTPException(status_code=status_code_for_error(exc), detail=detail) from exc


def _aggregate_normal(original_rules, results):
    rules = [rule.model_copy(deep=True) for rule in original_rules]
    changes = []
    candidates = []
    for result, field_candidates in results:
        changes.extend(result.changes)
        candidates.extend(field_candidates)
        for preview_rule in result.updated_rules:
            target = next((rule for rule in rules if str(rule.ID) == str(preview_rule.ID)), None)
            if target is not None:
                target.DETAILED_RULE = list(preview_rule.DETAILED_RULE)
                target.PROGRAM = preview_rule.PROGRAM
    return rules, changes, candidates


def _normal_strategy_result(original_rules, results, service):
    rules, changes, candidates = _aggregate_normal(original_rules, results)
    evaluations = [candidate.evaluation for candidate, _, _ in candidates if candidate.evaluation]
    usage = summarize_usage([candidate.usage for candidate, _, _ in candidates])
    # Usage is the authoritative logical-generation count. The provider's
    # transport counter can be a stale/incomplete snapshot while concurrent
    # field futures are still finishing.
    call_count = usage.calls
    first_metadata = candidates[0][0].metadata if candidates else {}
    reasoning = {
        "requested_effort": first_metadata.get("reasoning_effort_requested", "low"),
        "effective_effort": first_metadata.get("reasoning_effort_effective"),
        "supported": bool(first_metadata.get("reasoning_supported", False)),
        "visible_reasoning": bool(first_metadata.get("visible_reasoning", False)),
        "hidden_reasoning_exposed": False,
        "decision_summary_available": bool(first_metadata.get("decision_summary")),
        "reasoning_mode": first_metadata.get("reasoning_mode", "not_available"),
        "reasoning_parameter_sent": bool(first_metadata.get("reasoning_parameter_sent", False)),
        "verbosity_parameter_sent": bool(first_metadata.get("verbosity_parameter_sent", False)),
        "usage_diagnostics": first_metadata.get("usage_diagnostics", {}),
        "reasoning_tokens": (candidates[0][0].usage.reasoning_tokens if candidates else None),
        "reasoning_tokens_status": (candidates[0][0].usage.reasoning_tokens_status if candidates else "not_applicable"),
        "output_tokens_semantics": (candidates[0][0].usage.output_tokens_semantics if candidates else "provider_reported_output_total"),
        "usage_normalizer_version": USAGE_NORMALIZER_VERSION,
    }
    return StrategyResult(
        strategy="generative", updated_rules=rules, changes=changes,
        summary=service._summary(changes),
        evaluation=evaluations[0] if len(evaluations) == 1 else None,
        metadata={"status": "completed", "candidate_count": len(candidates),
                  "selected_for_persistence": False, "persistence_status": "awaiting_approval",
                  "requires_user_approval": True,
                  "oci_sentence_generation_called": call_count > 0,
                  "oci_sentence_generation_call_count": call_count,
                  "reasoning": reasoning,
                  "decision_summary": first_metadata.get("decision_summary")},
        usage=usage,
        sentence_generation_usage=usage,
    ), candidates


def _run_update_job(job_id: str, request: UpdateRequest, original_rules, changes):
    try:
        update_jobs.mark_normal_running(job_id, total_fields=len(changes))
        if not changes:
            normal_result = StrategyResult(
                strategy="generative", updated_rules=original_rules, changes=[], summary={},
                metadata={
                    "status": "completed",
                    "candidate_count": 0,
                    "candidate_status": "not_applicable",
                    "persistence_status": "not_applicable",
                    "oci_sentence_generation_called": False,
                    "oci_sentence_generation_call_count": 0,
                    "reason": "No changed mapped fields were detected",
                },
            )
            update_jobs.update_normal(job_id, normal_result.model_dump(), completed_fields=0, total_fields=0)
            update_jobs.complete_normal(job_id, normal_result.model_dump(), termination={"reason": "no_mapped_field_changes"})
            return
        selected_model = resolve_rule_generation_model(request.rule_generation_model)
        model_settings = rule_generation_settings(selected_model)
        base = OciNativeRuleGenerator(
            model_id=selected_model.model_id,
            region=model_settings["region"],
            compartment_id=model_settings["compartment_id"],
            serving_mode=model_settings["serving_mode"],
            endpoint_id=model_settings["endpoint_id"],
            reasoning_effort=request.reasoning_effort,
        )
        # Normal correction evaluation is text/evidence-gate based.  Do not
        # construct an extraction executor (and therefore do not create a
        # second OCI client) unless an explicitly enabled gold-set evaluation
        # asks for it.
        evaluator = None
        if os.getenv("FULL_GOLD_EVAL", "false").lower() == "true":
            from .evaluation import ExtractionEvaluator
            evaluation_repository = InMemoryRuleRepository(original_rules)
            executor = OciPdfExtractionExecutor(evaluation_repository) if request.document_id else load_extraction_executor()
            evaluator = ExtractionEvaluator(executor)
        service = UpdateRulesService(
            base, RuleRepository(), AuditRepository(), document_store=document_store,
            evaluator=evaluator,
        )
        field_results = []
        for change in changes:
            update_jobs.publish(job_id, "field_generation_started", {
                "field_key": change.field_key,
                "status": "running",
            })
        futures = {normal_field_executor.submit(service.run_normal_field, request, original_rules, change): change
                   for change in changes}
        for completed, future in enumerate(as_completed(futures), start=1):
            change = futures[future]
            try:
                field_result, field_candidates = future.result()
                field_results.append((field_result, field_candidates))
                update_jobs.update_normal(job_id, _normal_strategy_result(original_rules, field_results, service)[0].model_dump(),
                                        completed_fields=completed, total_fields=len(changes))
            except Exception as exc:
                logger.exception("Normal field job failed for %s", change.field_key)
                failed = StrategyResult(strategy="generative", updated_rules=original_rules,
                                        changes=[ChangeResult(
                                            FIELD_KEY=change.field_key,
                                            path=change.path,
                                            old_value=change.old_value,
                                            new_value=change.new_value,
                                            status="generation_failed",
                                            strategy="generative",
                                            reason=str(exc),
                                        )],
                                        summary={"generation_failures": 1},
                                        metadata={"status": "partial"})
                field_results.append((failed, []))
                update_jobs.update_normal(job_id, _normal_strategy_result(original_rules, field_results, service)[0].model_dump(),
                                        completed_fields=completed, total_fields=len(changes))
        normal_result, normal_candidates = _normal_strategy_result(original_rules, field_results, service)
        if not request.dry_run:
            service._audit_candidates(normal_candidates, selected=False)
            service._store_feedback_examples(normal_candidates)
        update_jobs.update_normal(job_id, normal_result.model_dump(), completed_fields=len(changes), total_fields=len(changes))

        update_jobs.complete_normal(job_id, normal_result.model_dump(),
                                     termination={"reason": "normal_generation_completed"},
                                     usage=normal_result.usage.model_dump() if normal_result.usage else None)
    except Exception as exc:
        logger.exception("Update background job failed: %s", job_id)
        reason = str(exc)
        status = getattr(exc, "status", None)
        lowered = reason.lower()
        if status in {401, 403, 404}:
            error = {
                "code": "OCI_RULE_GENERATION_AUTHORIZATION_OR_RESOURCE_ERROR",
                "message": "OCI rule-generation authorization or resource lookup failed",
                "reason": reason[:500],
                "operation": "rule_generation",
            }
        elif status == 400 and ("verbosity" in lowered or "invalid json" in lowered):
            error = {
                "code": "OCI_RULE_GENERATION_INVALID_REQUEST",
                "message": "OCI rejected the GPT-OSS rule-generation request",
                "reason": "Invalid OCI request payload",
            }
        else:
            error = {
                "code": "UPDATE_JOB_FAILED",
                "message": "Extraction-rule update job failed",
                "reason": reason[:500],
            }
        update_jobs.fail(job_id, {
            **error,
        })


def _job_response(job_id: str):
    try:
        data = update_jobs.get(job_id)
        return UpdateJobStatusResponse(
            job_id=data["job_id"], status=data["status"], phase=data.get("phase", data["status"]),
            normal_status=data.get("normal_status", "unknown"),
            normal_result=data.get("normal_result"),
            progress=data.get("progress", {}), requested_config=data.get("requested_config", {}),
            effective_config=data.get("effective_config", {}), termination=data.get("termination", {}),
            usage=data.get("usage", {}),
            change_detection=data.get("change_detection", {}),
            oci_sentence_generation_called=data.get("oci_sentence_generation_called", False),
            oci_sentence_generation_call_count=data.get("oci_sentence_generation_call_count", 0),
            reasoning=data.get("reasoning", {}),
            reasoning_ui_contract_version=data.get("reasoning_ui_contract_version", "missing"),
            sentence_generation_usage=data.get("sentence_generation_usage"),
            extraction_usage=data.get("extraction_usage"),
            rule_merge_usage=data.get("rule_merge_usage"),
            error=data.get("error"),
            requested_model=data.get("requested_model", "gpt-oss-20b"),
            effective_model=data.get("effective_model", "openai.gpt-oss-20b"),
            extraction_model=data.get("extraction_model", "google.gemini-2.5-flash"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail={
            "code": "UPDATE_JOB_NOT_FOUND",
            "message": "Update job was not found or has expired",
        }) from exc


@app.get("/v1/extraction-rules/update-jobs/{job_id}", response_model=UpdateJobStatusResponse)
def get_update_job(job_id: str):
    return _job_response(job_id)


@app.get("/v1/extraction-rules/update-jobs/{job_id}/events")
def get_update_job_events(job_id: str, request: Request):
    try:
        job = update_jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail={
            "code": "UPDATE_JOB_NOT_FOUND",
            "message": "Update job was not found or has expired",
        }) from exc
    if job.get("status") == "expired":
        raise HTTPException(status_code=410, detail={
            "code": "UPDATE_JOB_EXPIRED",
            "message": "Update job result expired",
        })
    try:
        last_event_id = int(request.headers.get("last-event-id", "0"))
    except ValueError:
        last_event_id = 0

    def event_stream():
        try:
            for event in update_jobs.stream_events(job_id, last_event_id):
                yield f"id: {event['id']}\nevent: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
        except KeyError:
            return

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


@app.get("/health")
def health():
    sentence_key = os.getenv("OCI_RULE_GENERATION_MODEL_KEY", "gpt-oss-20b")
    try:
        sentence_model = resolve_rule_generation_model(sentence_key)
    except ValueError:
        sentence_model = resolve_rule_generation_model(None)
    sentence_settings = rule_generation_settings(sentence_model)
    return {
        "status": "ok",
        "model_id": os.getenv("OCI_EXTRACTION_MODEL_ID", os.getenv("OCI_MODEL_ID", "google.gemini-2.5-flash")),
        "region": os.getenv("OCI_EXTRACTION_REGION") or os.getenv("OCI_EXC_REGION") or os.getenv("OCI_REGION", "ap-hyderabad-1"),
        "application_output_limit_sent": False,
        "provider_managed_output_limit": True,
        "extraction_model": os.getenv("OCI_EXTRACTION_MODEL_ID", "google.gemini-2.5-flash"),
        "extraction_region": os.getenv("OCI_EXTRACTION_REGION") or os.getenv("OCI_EXC_REGION") or os.getenv("OCI_REGION", "unknown"),
        "extraction_project_configured": bool(os.getenv("OCI_EXTRACTION_PROJECT_ID") or os.getenv("PROJECT_ID")),
        "sentence_generation_model": sentence_model.model_id,
        "sentence_generation_region": sentence_settings.get("region") or "unknown",
        "sentence_generation_transport": "oci_native",
        "sentence_generation_project_required": False,
        "sentence_generation_serving_mode": sentence_settings.get("serving_mode") or "on_demand",
        "sentence_generation_endpoint_configured": bool(sentence_settings.get("endpoint_id")),
        "usage_normalizer_version": USAGE_NORMALIZER_VERSION,
        "sentence_generation_usage_available": True,
        "extraction_usage_available": True,
    }


@app.get("/v1/extraction-rules")
def get_rules():
    return {"rules": [r.model_dump() for r in RuleRepository().load()]}


@app.post("/v1/invoices/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    if file.content_type not in {"application/pdf", "application/octet-stream"} and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail={"code": "PDF_REQUIRED", "message": "Upload a PDF invoice"})
    maximum = int(os.getenv("MAX_PDF_BYTES", str(20 * 1024 * 1024)))
    document_bytes = await file.read(maximum + 1)
    if len(document_bytes) > maximum:
        raise HTTPException(status_code=413, detail={"code": "PDF_TOO_LARGE", "message": f"Maximum PDF size is {maximum} bytes"})
    document_id = document_store.put(document_bytes, file.filename or "invoice.pdf", "application/pdf")
    try:
        extractor = OciPdfExtractor(RuleRepository(), model_id=os.getenv("OCI_EXTRACTION_MODEL_ID", "google.gemini-2.5-flash"))
        extracted, page_count, diagnostics = extractor.extract(
            document_bytes,
            filename=file.filename or "invoice.pdf",
        )
        return ExtractionResponse(document_id=document_id, filename=file.filename or "invoice.pdf", extracted_json=extracted,
                                  model=extractor.model_id, page_count=page_count, diagnostics=diagnostics,
                                  usage=UsageSummary.model_validate(diagnostics.get("usage", {})))
    except ExtractionConfigurationError as exc:
        document_store.delete(document_id)
        status_code = 422 if exc.code.endswith("MISSING") or exc.code.endswith("MISMATCH") else 502
        raise HTTPException(status_code=status_code, detail={
            "code": exc.code,
            "message": str(exc),
            "operation": "pdf_extraction",
        }) from exc
    except PdfExtractionError as exc:
        document_store.delete(document_id)
        error_code = exc.details.get("code")
        status_code = 413 if error_code == "PDF_TOO_MANY_PAGES" else 415 if error_code == "PDF_REQUIRED" else 502
        raise HTTPException(status_code=status_code, detail={"code": error_code or "INVALID_EXTRACTION_OUTPUT",
            "message": str(exc), "details": exc.details}) from exc
    except Exception as exc:
        message = str(exc)
        if "Invalid OpenAI project" in message or "invalid project" in message.lower():
            document_store.delete(document_id)
            logger.exception("Invoice extraction project validation failed")
            raise HTTPException(status_code=502, detail={
                "code": "OCI_EXTRACTION_PROJECT_INVALID",
                "message": "The configured Generative AI Project is invalid or inaccessible",
                "operation": "pdf_extraction",
            }) from exc
        document_store.delete(document_id)
        logger.exception("Invoice extraction failed")
        raise HTTPException(status_code=502, detail={"code": "OCI_EXTRACTION_FAILED", "message": "Invoice extraction failed"}) from exc


@app.post("/v1/extraction-rules/promote", response_model=PromoteResponse)
def promote_candidate(request: PromoteRequest):
    try:
        service = UpdateRulesService(None, RuleRepository(), AuditRepository())
        return service.promote(request.candidate_id, request.expected_rule_version, request.dry_run,
                               request.promotion_scope, request.supplier_key)
    except Exception as exc:
        logger.exception("Candidate promotion failed")
        raise HTTPException(status_code=500, detail={"code": "PROMOTION_FAILED", "reason": str(exc)[:500]}) from exc


@app.post("/v1/extraction-rules/promote-batch", response_model=PromoteBatchResponse)
def promote_batch(request: PromoteBatchRequest):
    try:
        service = UpdateRulesService(None, RuleRepository(), AuditRepository())
        return service.promote_batch(request)
    except Exception as exc:
        logger.exception("Batch candidate promotion failed")
        raise HTTPException(status_code=500, detail={"code": "PROMOTION_FAILED", "reason": str(exc)[:500]}) from exc
