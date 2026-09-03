from __future__ import annotations

import os
import uuid
import hashlib
import json

from .comparator import find_changes
from .evidence import ExtractionEvidenceBuilder, layout_signature
from .feedback_repository import FeedbackRepository
from .models import ChangeResult, RuleGenerationContext, StrategyResult, UpdateRequest, UpdateResponse, UsageSummary
from .normalization import infer_policy
from .normal_strategy import GeneratedRuleCandidate, GenerativeRuleGenerator, prompt_hash
from .usage import summarize_usage
from .model_output import ModelOutputError
from .sentence_gates import evaluate_sentence_gates
from .rule_merger import GeminiRuleMerger, has_semantic_conflict, merge_rule_sentences
from .sentence_payload import build_sentence_payload
from .supplier_store import SupplierRuleStore


class RuleGenerationError(RuntimeError):
    def __init__(self, message: str, *, field_key: str, strategy: str):
        super().__init__(message)
        self.field_key = field_key
        self.strategy = strategy


class UpdateRulesService:
    def __init__(self, generator, repository, audit, feedback_repository=None, document_store=None,
                 evaluator=None, baseline_instructions=None, supplier_store=None, rule_merger=None):
        self.generator = generator
        self.repository = repository
        self.audit = audit
        self.feedback = feedback_repository or FeedbackRepository()
        self.document_store = document_store
        self.evaluator = evaluator
        self.baseline_instructions = baseline_instructions or {}
        self.supplier_store = supplier_store or SupplierRuleStore()
        self.rule_merger = rule_merger

    @staticmethod
    def _candidate(generator, context: RuleGenerationContext) -> GeneratedRuleCandidate:
        if hasattr(generator, "generate_candidate"):
            return generator.generate_candidate(context)
        return GenerativeRuleGenerator(generator).generate_candidate(context)

    @staticmethod
    def _summary(results):
        return {
            "changed_fields": len(results),
            "updated_rules": sum(x.status == "updated" for x in results),
            "preview_candidates": sum(x.status == "preview" for x in results),
            "duplicates": sum(x.status == "duplicate" for x in results),
            "unmapped_fields": sum(x.status == "unmapped" for x in results),
            "generation_failures": sum(x.status in {"generation_failed", "unavailable"} for x in results),
        }

    def _context(self, request, rule, change):
        # Resolve the read path as global rules overlaid by the requested
        # supplier rule. The repository's global rule object is never mutated
        # by preview generation.
        if request.supplier_key:
            overlay = self.supplier_store.resolve({}, request.supplier_key).get(rule.FIELD_KEY)
            if isinstance(overlay, dict):
                from .models import RuleRecord
                rule = RuleRecord.model_validate({**rule.model_dump(), **overlay})
        # Feedback is selected across fields by compatible type/correction
        # metadata; same-field history is only a tie-breaker.
        history = self.feedback.load_all()
        document = self.document_store.get(request.document_id) if self.document_store and request.document_id else None
        packet = None
        if document:
            packet = ExtractionEvidenceBuilder().build(
                document_bytes=document.document_bytes,
                rule=rule,
                field_path=change.path,
                old_value=change.old_value,
                new_value=change.new_value,
                original_field_node=request.invoice_payload.get(rule.FIELD_KEY),
                corrected_field_node=request.final_response.get(rule.FIELD_KEY),
            )
            current_layout = layout_signature(document.document_bytes)
            history = [{**row, "current_layout_signature": current_layout} for row in history]
            packet.historical_examples = self.feedback.select_demonstrations(packet, history, min(request.history_limit, 2))
        else:
            history = history[-min(max(request.history_limit, 0), 2):]
        context = RuleGenerationContext(field_key=rule.FIELD_KEY, display_label=rule.DISPLAY_LABEL,
            short_rule=rule.SHORT_RULE, detailed_rule=list(rule.DETAILED_RULE), old_value=change.old_value,
            new_value=change.new_value, field_path=change.path, invoice_payload=request.invoice_payload,
            final_response=request.final_response,
            historical_examples=[item.model_dump() for item in packet.historical_examples] if packet else history,
            rule_version="v1", extraction_function="OciPdfExtractionExecutor" if document else "unconfigured",
            document_id=document.document_id if document else None, mime_type=document.mime_type if document else "application/pdf",
            document_bytes=document.document_bytes if document else None,
            feedback_packet=packet,
            baseline_instruction=self.baseline_instructions.get(rule.FIELD_KEY),
            normalization_mode=infer_policy(rule.FIELD_KEY, change.old_value, change.new_value).mode,
            reasoning_effort=request.reasoning_effort,
            current_program=rule.PROGRAM if isinstance(rule.PROGRAM, dict) else None)
        return context

    def _run_strategy(self, generator, strategy, request, rules, changes, *, tolerate_failure=False):
        working = [r.model_copy(deep=True) for r in rules]
        by_key = {r.FIELD_KEY: r for r in working}
        results = []
        candidates = []
        for change in changes:
            rule = by_key.get(change.field_key)
            if not rule:
                results.append(ChangeResult(FIELD_KEY=change.field_key, path=change.path, old_value=change.old_value,
                    new_value=change.new_value, status="unmapped", strategy=strategy))
                continue
            try:
                context = self._context(request, rule, change)
                # The active architecture is intentionally LLM-first. The
                # legacy deterministic candidate helper remains available for
                # rollback experiments but is not used in production flow.
                existing_preview = self.audit.find_preview_by_prompt_hash(prompt_hash(context))
                if existing_preview and existing_preview.get("generated_sentence"):
                    candidate = GeneratedRuleCandidate(
                        sentence=existing_preview["generated_sentence"], strategy=strategy,
                        request_id=existing_preview.get("oci_request_id"),
                        candidate_id=existing_preview.get("candidate_id") or str(uuid.uuid4()),
                        prompt_hash=prompt_hash(context),
                        reason=existing_preview.get("metadata", {}).get("reason"),
                        metadata=dict(existing_preview.get("metadata") or {}),
                        # Reusing a preview is not a new OCI call. Keep any
                        # historical usage in the audit record, but report
                        # current-job usage as zero.
                        usage=UsageSummary(),
                    )
                    candidate.metadata["idempotent_reuse"] = True
                    candidate.metadata["oci_calls"] = 0
                else:
                    candidate = self._candidate(generator, context)
                if not candidate.sentence:
                    raise ValueError("strategy produced an empty rule sentence")
                sentence_payload = build_sentence_payload(context)
                if os.getenv("FULL_GOLD_EVAL", "false").lower() == "true" and self.evaluator and context.document_bytes:
                    candidate.evaluation, _ = self.evaluator.evaluate(context, candidate.sentence)
                else:
                    candidate.evaluation = evaluate_sentence_gates(candidate.sentence, context)
                rejected = bool(candidate.evaluation and candidate.evaluation.candidate_status == "rejected")
                added = False
                if not rejected:
                    before_rules = list(rule.DETAILED_RULE)
                    if has_semantic_conflict(before_rules, candidate.sentence):
                        merged = (self.rule_merger or GeminiRuleMerger()).merge(
                            context, candidate.sentence, candidate.metadata.get("correction_kind", "value_replacement")
                        )
                    else:
                        merged = merge_rule_sentences(before_rules, candidate.sentence)
                    rule.DETAILED_RULE = merged.updated_detailed_rule
                    if merged.short_rule:
                        rule.SHORT_RULE = merged.short_rule
                    if candidate.metadata.get("program"):
                        rule.PROGRAM = candidate.metadata["program"]
                    added = candidate.sentence.casefold() not in {item.casefold() for item in before_rules}
                    candidate.metadata["rule_diff"] = {
                        "before": before_rules,
                        "after": merged.updated_detailed_rule,
                        "dropped_bullets": merged.dropped_bullets,
                        "conflict_resolved": merged.conflict_resolved,
                        "short_rule": merged.short_rule,
                    }
                result = ChangeResult(ID=rule.ID, FIELD_KEY=rule.FIELD_KEY, path=change.path,
                    old_value=change.old_value, new_value=change.new_value,
                    status="rejected" if rejected else "preview", generated_sentence=candidate.sentence,
                    oci_request_id=candidate.request_id, response_format_used=candidate.response_format,
                    strategy=strategy, evaluation_score=candidate.evaluation.score if candidate.evaluation else None,
                    evaluation_feedback=candidate.evaluation.feedback if candidate.evaluation else None,
                    persistence_status="not_persisted" if rejected else "awaiting_approval",
                    candidate_status=(candidate.evaluation.candidate_status if candidate.evaluation else "generated"),
                    promotion_eligible=bool(candidate.evaluation and candidate.evaluation.promotion_eligible and not rejected))
                result.usage = candidate.usage
                result.model = candidate.metadata.get("model")
                result.reasoning_effort_requested = candidate.metadata.get("reasoning_effort_requested")
                result.reasoning_effort_effective = candidate.metadata.get("reasoning_effort_effective")
                result.reasoning_supported = bool(candidate.metadata.get("reasoning_supported", False))
                result.visible_reasoning = bool(candidate.metadata.get("visible_reasoning", False))
                result.decision_summary = candidate.metadata.get("decision_summary")
                result.reason = candidate.metadata.get("reason") or result.decision_summary
                # The correction kind is an immutable service-side fact from
                # the structured delta, never a model-provided label.
                result.correction_kind = sentence_payload["correction_kind"]
                result.observed_correction = sentence_payload["delta"]["observed_change"]
                intent = candidate.metadata.get("intent")
                if isinstance(intent, dict):
                    from .models import CorrectionIntent
                    try:
                        intent = CorrectionIntent.model_validate(intent)
                    except Exception:
                        intent = None
                result.intent = intent
                result.correction_intent = result.intent
                result.oci_calls = int(candidate.metadata.get("oci_calls", candidate.usage.calls or (1 if candidate.request_id else 0)))
                if rejected:
                    result.rejection_reason = candidate.evaluation.feedback if candidate.evaluation else "candidate rejected"
                result.reason_basis = ["current_rule", "corrected_field_behavior", "correction_policy"]
                if context.feedback_packet and context.feedback_packet.evidence:
                    result.reason_basis.append("matched_labeled_evidence")
                if context.feedback_packet and context.feedback_packet.competing_evidence:
                    result.reason_basis.append("competing_evidence_rejected")
                result.evidence = context.feedback_packet.evidence if context.feedback_packet else []
                result.competing_evidence = context.feedback_packet.competing_evidence if context.feedback_packet else []
                result.evidence_status = (
                    "supported" if result.evidence else
                    "ambiguous" if result.competing_evidence else
                    "unavailable"
                )
                result.rule_diff = candidate.metadata.get("rule_diff", {})
                candidate.metadata["would_add_rule"] = added
                candidate.metadata["sentence_payload"] = sentence_payload
                candidate.metadata["correction_kind"] = sentence_payload["correction_kind"]
                candidate.metadata["intent"] = result.intent.model_dump() if result.intent else None
                candidate.metadata["correction_intent"] = result.intent.model_dump() if result.intent else None
                candidate.metadata["reason_basis"] = list(result.reason_basis)
                candidate.metadata["supplier_key"] = request.supplier_key
                candidate.metadata["promotion_scope"] = "supplier" if request.supplier_key else "global_compatibility"
                candidate.metadata["usage"] = candidate.usage.model_dump()
                candidate.metadata["rule_version"] = context.rule_version
                candidate.metadata["persistence_status"] = result.persistence_status
                candidate.metadata["selected_for_persistence"] = False
                candidate.metadata["promotion_eligible"] = bool(
                    candidate.evaluation and candidate.evaluation.promotion_eligible and not rejected
                )
                # ``reason`` is the safe business explanation for successful
                # candidates. Operational failures retain their existing
                # failure reason in the exception path below.
                result.candidate_id = candidate.candidate_id
                result.prompt_hash = candidate.prompt_hash
                if context.feedback_packet:
                    result.evidence = context.feedback_packet.evidence
                    result.competing_evidence = context.feedback_packet.competing_evidence
                    result.feedback_packet = context.feedback_packet.model_dump(exclude={"original_field_node", "corrected_field_node"})
                    result.confidence = context.feedback_packet.confidence
                    result.demonstrations_used = len(context.feedback_packet.historical_examples)
                candidates.append((candidate, context, result))
                results.append(result)
            except Exception as exc:
                status = "unavailable" if "unavailable" in str(exc).lower() or "not configured" in str(exc).lower() else "generation_failed"
                diagnostics = getattr(exc, "diagnostics", {}) or {}
                generation = diagnostics.get("generation", {}) if isinstance(diagnostics, dict) else {}
                truncated = isinstance(exc, ModelOutputError) and (
                    str(exc.finish_reason or "").upper() in {"MAX_TOKENS", "LENGTH"}
                    or "truncated" in str(exc).lower()
                )
                validation_failed = isinstance(exc, ModelOutputError) and bool(diagnostics.get("validation_error"))
                result = ChangeResult(ID=rule.ID, FIELD_KEY=rule.FIELD_KEY, path=change.path,
                    old_value=change.old_value, new_value=change.new_value, status=status,
                    strategy=strategy, reason=str(exc),
                    error={
                        "code": ("OCI_PROVIDER_OUTPUT_LIMIT" if truncated else
                                 "OCI_RULE_GENERATION_OUTPUT_VALIDATION" if validation_failed else
                                 "RULE_GENERATION_FAILED"),
                        "message": str(exc),
                    },
                    generation=generation,
                    oci_calls=int(generation.get("attempts", 0) or 0),
                    rejection_reason=("provider_output_truncated" if truncated else
                                      "provider_error" if not validation_failed else
                                      "sentence_validation_failed"))
                results.append(result)
                if not tolerate_failure:
                    raise RuleGenerationError(
                        f"{strategy} rule generation failed for {change.field_key}: {exc}",
                        field_key=change.field_key, strategy=strategy,
                    ) from exc
        return working, results, candidates

    def _audit_candidates(self, candidates, *, selected):
        for candidate, context, result in candidates:
            evaluation = candidate.evaluation
            self.audit.append(rule_id=result.ID, field_key=result.FIELD_KEY, path=result.path,
                old_value=result.old_value, new_value=result.new_value, sentence=result.generated_sentence,
                status=result.status, request_id=result.oci_request_id, strategy=candidate.strategy,
                evaluation_score=evaluation.score if evaluation else None,
                evaluation_feedback=evaluation.feedback if evaluation else None,
                selected_for_persistence=selected, promotion_status="not_promoted",
                candidate_id=candidate.candidate_id, prompt_hash=candidate.prompt_hash,
                metadata=candidate.metadata, model=candidate.metadata.get("model"))

    def _store_feedback_examples(self, candidates):
        """Store only explicitly enabled, redacted evidence demonstrations."""
        if os.getenv("STORE_FEEDBACK_EVIDENCE", "false").lower() != "true":
            return
        for _, context, result in candidates:
            packet = context.feedback_packet
            if not packet:
                continue
            def fingerprint(value):
                return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]
            self.feedback.append({
                "field_key": packet.field_key,
                "field_type": "list" if isinstance(packet.previous_value, list) else "scalar",
                "failure_type": packet.failure_type,
                "old_value_fingerprint": fingerprint(packet.previous_value),
                "new_value_fingerprint": fingerprint(packet.corrected_value),
                "label_text": ", ".join(sorted({str(item.label) for item in packet.evidence if item.label}))[:160],
                "selected_sentence": result.generated_sentence,
                "rule_diff": (result.rule_diff or {}),
                "promotion_status": result.persistence_status,
                "token_metrics": result.usage.model_dump() if result.usage else {},
                "timestamp": __import__("time").time(),
            })

    def update(self, request: UpdateRequest) -> UpdateResponse:
        rules = self.repository.load()
        changes = find_changes(request.invoice_payload, request.final_response)
        normal_rules, normal_changes, normal_candidates = self._run_strategy(
            self.generator, "generative", request, rules, changes, tolerate_failure=request.allow_partial)
        normal_summary = self._summary(normal_changes)
        if not request.dry_run:
            self._audit_candidates(normal_candidates, selected=False)
            self._store_feedback_examples(normal_candidates)

        normal_aggregate = [candidate.evaluation for candidate, _, _ in normal_candidates if candidate.evaluation]
        normal_evaluation = normal_aggregate[0] if len(normal_aggregate) == 1 else None
        strategies = [StrategyResult(strategy="generative", updated_rules=normal_rules,
            changes=normal_changes, summary=normal_summary, evaluation=normal_evaluation,
            metadata={"status": "completed", "candidate_count": len(normal_candidates),
                      "selected_for_persistence": False, "persistence_status": "awaiting_approval",
                      "requires_user_approval": True})]

        return UpdateResponse(strategies=strategies, updated_rules=normal_rules,
                              changes=normal_changes, summary=normal_summary)

    def run_normal_field(self, request: UpdateRequest, original_rules, change):
        """Run one normal candidate in isolation for progressive job updates."""
        working, changes, candidates = self._run_strategy(
            self.generator, "generative", request, original_rules, [change], tolerate_failure=True
        )
        evaluations = [candidate.evaluation for candidate, _, _ in candidates if candidate.evaluation]
        result = StrategyResult(
            strategy="generative",
            updated_rules=working,
            changes=changes,
            summary=self._summary(changes),
            evaluation=evaluations[0] if len(evaluations) == 1 else None,
            metadata={
                "status": "completed" if not any(c.status == "generation_failed" for c in changes) else "partial",
                "candidate_count": len(candidates),
                "selected_for_persistence": False,
                "persistence_status": "awaiting_approval",
                "requires_user_approval": True,
            },
            usage=summarize_usage([candidate.usage for candidate, _, _ in candidates]),
        )
        result.sentence_generation_usage = result.usage
        return result, candidates

    def promote(self, candidate_id: str, expected_rule_version: str = "v1", dry_run: bool = False,
                promotion_scope: str = "supplier", supplier_key: str | None = None):
        from .models import PromoteResponse
        record = self.audit.find_candidate(candidate_id)
        if not record:
            return PromoteResponse(candidate_id=candidate_id, status="not_found", reason="candidate was not found")
        if record.get("strategy") != "generative":
            return PromoteResponse(candidate_id=candidate_id, status="rejected", reason="unsupported candidate strategy")
        if record.get("promotion_status") == "promoted":
            return PromoteResponse(candidate_id=candidate_id, status="already_promoted",
                                   rule_id=record.get("ID"), field_key=record.get("FIELD_KEY"),
                                   sentence=record.get("generated_sentence"))
        if expected_rule_version != "v1":
            return PromoteResponse(candidate_id=candidate_id, status="stale", reason="rule version mismatch")
        rules = self.repository.load()
        rule = next((r for r in rules if str(r.ID) == str(record.get("ID"))), None)
        if rule is None:
            return PromoteResponse(candidate_id=candidate_id, status="rejected", reason="target rule no longer exists")
        sentence = record.get("generated_sentence")
        if not sentence:
            return PromoteResponse(candidate_id=candidate_id, status="rejected", reason="candidate has no sentence")
        metadata = record.get("metadata") or {}
        if record.get("status") not in {"preview", "evaluated", "generated"}:
            return PromoteResponse(candidate_id=candidate_id, status="rejected",
                                   reason="candidate is not an approval-eligible preview")
        if metadata.get("persistence_status") != "awaiting_approval":
            return PromoteResponse(candidate_id=candidate_id, status="rejected",
                                   reason="candidate is not awaiting approval")
        if (metadata.get("confidence") or "").lower() in {"limited", "low", "unavailable"}:
            return PromoteResponse(candidate_id=candidate_id, status="rejected",
                                   reason="low-confidence candidates cannot be promoted")
        if not metadata.get("promotion_eligible", False):
            return PromoteResponse(candidate_id=candidate_id, status="rejected",
                                   reason="candidate did not pass evaluation or promotion gates")
        scope = (promotion_scope or "supplier").strip().lower()
        if scope not in {"supplier", "global"}:
            return PromoteResponse(candidate_id=candidate_id, status="rejected", reason="invalid promotion scope")
        target_supplier = supplier_key or metadata.get("supplier_key")
        if scope == "supplier" and not target_supplier:
            # Preserve compatibility for older clients that did not submit a
            # supplier identity; new clients should always provide one.
            scope = "global"
        # Apply the validated preview diff exactly once. Re-running the old
        # append operation here duplicated the generated sentence and could
        # discard conflict removals made during preview.
        diff_after = (metadata.get("rule_diff") or {}).get("after")
        if isinstance(diff_after, list):
            rule.DETAILED_RULE = [str(item) for item in diff_after]
        elif sentence.casefold() not in {item.casefold() for item in rule.DETAILED_RULE}:
            rule.DETAILED_RULE.append(sentence)
        short_rule = (metadata.get("rule_diff") or {}).get("short_rule")
        if short_rule:
            rule.SHORT_RULE = str(short_rule)
        if not dry_run:
            if scope == "supplier":
                self.supplier_store.save_field(target_supplier, rule.FIELD_KEY, rule.model_dump())
            else:
                self.repository.save(rules)
            self.audit.append(rule_id=record.get("ID"), field_key=record.get("FIELD_KEY"), path=record.get("path", ""),
                old_value=None, new_value=None, sentence=sentence, status="promoted", request_id=record.get("oci_request_id"),
                strategy=record.get("strategy", "generative"), selected_for_persistence=True, promotion_status="promoted", candidate_id=candidate_id,
                prompt_hash=record.get("prompt_hash"), metadata=record.get("metadata", {}))
        return PromoteResponse(candidate_id=candidate_id, status="promoted" if not dry_run else "valid",
                               rule_id=record.get("ID"), field_key=record.get("FIELD_KEY"), sentence=sentence)

    def promote_batch(self, request):
        from .models import PromoteBatchResponse

        candidate_ids = request.candidate_ids
        if not request.confirm:
            return PromoteBatchResponse(status="rejected", reason="explicit confirmation is required")
        if len(candidate_ids) != len(set(candidate_ids)):
            return PromoteBatchResponse(status="rejected", reason="duplicate candidate IDs are not allowed")
        if request.expected_rule_version != "v1":
            return PromoteBatchResponse(status="rejected", reason="rule version mismatch")
        scope = (request.promotion_scope or "supplier").strip().lower()
        if scope not in {"supplier", "global"}:
            return PromoteBatchResponse(status="rejected", reason="invalid promotion scope")

        records = [self.audit.find_candidate(candidate_id) for candidate_id in candidate_ids]
        if any(record is None for record in records):
            return PromoteBatchResponse(status="rejected", reason="one or more candidates were not found")
        fields = [record.get("FIELD_KEY") for record in records]
        if len(fields) != len(set(fields)):
            return PromoteBatchResponse(status="rejected", reason="only one candidate may be approved per field")

        rules = self.repository.load()
        rule_by_id = {str(rule.ID): rule for rule in rules}
        validation_errors = []
        for record in records:
            metadata = record.get("metadata") or {}
            if record.get("strategy") != "generative":
                validation_errors.append("unsupported strategy")
            if record.get("promotion_status") == "promoted":
                validation_errors.append(f"candidate {record.get('candidate_id')} was already promoted")
            if record.get("status") not in {"preview", "evaluated", "generated"}:
                validation_errors.append(f"candidate {record.get('candidate_id')} is not an approval-eligible preview")
            if not record.get("generated_sentence"):
                validation_errors.append(f"candidate {record.get('candidate_id')} has no sentence")
            if metadata.get("persistence_status") != "awaiting_approval":
                validation_errors.append(f"candidate {record.get('candidate_id')} is not awaiting approval")
            if (metadata.get("confidence") or "").lower() in {"limited", "low", "unavailable"}:
                validation_errors.append(f"candidate {record.get('candidate_id')} is low-confidence")
            if not metadata.get("promotion_eligible", False):
                validation_errors.append(f"candidate {record.get('candidate_id')} failed promotion gates")
            if str(record.get("ID")) not in rule_by_id:
                validation_errors.append(f"target rule for {record.get('candidate_id')} no longer exists")
            if metadata.get("rule_version", "v1") != request.expected_rule_version:
                validation_errors.append(f"candidate {record.get('candidate_id')} is stale")
        if validation_errors:
            return PromoteBatchResponse(status="rejected", reason="; ".join(validation_errors))

        for record in records:
            rule = rule_by_id[str(record.get("ID"))]
            metadata = record.get("metadata") or {}
            diff_after = (metadata.get("rule_diff") or {}).get("after")
            if isinstance(diff_after, list):
                rule.DETAILED_RULE = [str(item) for item in diff_after]
            elif record["generated_sentence"].casefold() not in {item.casefold() for item in rule.DETAILED_RULE}:
                rule.DETAILED_RULE.append(record["generated_sentence"])
            short_rule = (metadata.get("rule_diff") or {}).get("short_rule")
            if short_rule:
                rule.SHORT_RULE = str(short_rule)
        if request.dry_run:
            return PromoteBatchResponse(
                status="valid", rule_version=request.expected_rule_version,
                promoted_candidates=[{"candidate_id": r["candidate_id"], "FIELD_KEY": r["FIELD_KEY"], "strategy": r["strategy"], "status": "valid"} for r in records],
                backup_created=False,
            )

        target_supplier = request.supplier_key or (records[0].get("metadata") or {}).get("supplier_key")
        if scope == "supplier" and not target_supplier:
            scope = "global"
        if scope == "supplier":
            for record in records:
                self.supplier_store.save_field(
                    target_supplier,
                    record["FIELD_KEY"],
                    rule_by_id[str(record.get("ID"))].model_dump(),
                )
        else:
            self.repository.save(rules)
        promoted = []
        for record in records:
            self.audit.append(rule_id=record.get("ID"), field_key=record.get("FIELD_KEY"), path=record.get("path", ""),
                              old_value=None, new_value=None, sentence=record.get("generated_sentence"), status="promoted",
                              request_id=record.get("oci_request_id"), strategy=record.get("strategy", "generative"),
                              selected_for_persistence=True, promotion_status="promoted", candidate_id=record.get("candidate_id"),
                              prompt_hash=record.get("prompt_hash"), metadata={**(record.get("metadata") or {}),
                                                                            "rule_version": request.expected_rule_version})
            promoted.append({"candidate_id": record["candidate_id"], "FIELD_KEY": record["FIELD_KEY"],
                             "strategy": record["strategy"], "status": "persisted"})
        return PromoteBatchResponse(status="promoted", rule_version=request.expected_rule_version,
                                    promoted_candidates=promoted, backup_created=True)
