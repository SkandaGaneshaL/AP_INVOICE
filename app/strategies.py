from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from .evaluation import ExtractionEvaluator
from .gepa_adapter import ExtractionRuleGEPAAdapter, serialize_context_example
from .model_output import parse_rule_response
from .models import EvaluationResult, ProviderGenerationResult, RuleGenerationContext, UsageSummary
from .prompt_builder import RulePromptBuilder
from .oci_reflection import OciReflectionLM
from .usage import summarize_usage


@dataclass
class GeneratedRuleCandidate:
    sentence: str | None
    strategy: str
    request_id: str | None = None
    response_format: str | None = None
    evaluation: EvaluationResult | None = None
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_candidate_id: str | None = None
    prompt_hash: str | None = None
    status: str = "generated"
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: UsageSummary = field(default_factory=UsageSummary)


class RuleCandidateGenerator(Protocol):
    name: str

    def generate_candidate(self, context: RuleGenerationContext) -> GeneratedRuleCandidate: ...


class _GepaProgressCallback:
    def __init__(self, callback):
        self.callback = callback
        self.metric_calls = 0
        self.last_iteration = 0

    def on_iteration_start(self, event):
        self.last_iteration = int(event.get("iteration", 0) or 0)
        self.callback(iteration=self.last_iteration, metric_calls=self.metric_calls)

    def on_evaluation_end(self, event):
        self.metric_calls += len(event.get("scores", []))
        self.last_iteration = int(event.get("iteration", self.last_iteration) or self.last_iteration)
        self.callback(iteration=self.last_iteration, metric_calls=self.metric_calls)


class GenerativeRuleGenerator:
    name = "generative"

    def __init__(self, generator):
        self.generator = generator

    def generate_candidate(self, context):
        try:
            if hasattr(self.generator, "generate_with_metadata"):
                provider_result = self.generator.generate_with_metadata(context=context)
                if isinstance(provider_result, ProviderGenerationResult):
                    return GeneratedRuleCandidate(
                        sentence=provider_result.sentence, strategy=self.name,
                        request_id=provider_result.request_id, response_format=provider_result.response_format,
                        prompt_hash=prompt_hash(context), usage=provider_result.usage,
                        metadata={"model": provider_result.model, "attempts": provider_result.attempts,
                                  "finish_reason": provider_result.finish_reason},
                    )
            result = self.generator.generate(context=context)
        except TypeError as exc:
            # Compatibility path for existing fake/legacy providers with an
            # explicit keyword-only generate() signature.
            if "context" not in str(exc) and "unexpected keyword" not in str(exc):
                raise
            result = self.generator.generate(
                field_key=context.field_key,
                display_label=context.display_label,
                short_rule=context.short_rule,
                detailed_rule=context.detailed_rule,
                old_value=context.old_value,
                new_value=context.new_value,
                field_path=context.field_path,
                invoice_payload=context.invoice_payload,
                final_response=context.final_response,
                historical_examples=context.historical_examples,
            )
        sentence, request_id, format_used = result if len(result) == 3 else (*result, None)
        return GeneratedRuleCandidate(sentence=sentence, strategy=self.name, request_id=request_id,
                                      response_format=format_used, prompt_hash=prompt_hash(context),
                                      usage=UsageSummary(calls=1, unknown_calls=1))


class GepaRuleOptimizer:
    name = "gepa"

    def __init__(self, fallback_generator, *, reflection_generator=None, evaluator=None, max_iterations=2,
                 max_metric_calls=5, max_reflection_cost=None, timeout_seconds=120,
                 validation_limit=5, seed=0, progress_callback=None):
        self.fallback_generator = fallback_generator
        self.reflection_generator = reflection_generator or fallback_generator
        self.evaluator = evaluator or ExtractionEvaluator()
        self.max_iterations = max_iterations
        self.max_metric_calls = max_metric_calls
        self.max_reflection_cost = max_reflection_cost
        self.timeout_seconds = timeout_seconds
        self.validation_limit = validation_limit
        self.seed = seed
        self.progress_callback = progress_callback

    def generate_candidate(self, context):
        if self.evaluator.executor is None:
            raise RuntimeError("GEPA unavailable: EXTRACTION_EXECUTOR is not configured")
        try:
            import gepa
        except ImportError as exc:
            raise RuntimeError("GEPA is unavailable; install the optional 'gepa' dependency") from exc
        # GEPA is intentionally independent from the normal OCI candidate.
        # Its seed comes from the existing rule and correction policy only.
        seed_instruction = RulePromptBuilder.gepa_seed(context)
        seed = {"extraction_instruction": seed_instruction}
        current_example = serialize_context_example(context)
        historical_examples = []
        for item in context.historical_examples:
            historical = dict(item)
            historical["_example_type"] = "historical"
            historical["document_available"] = False
            historical_examples.append(historical)
        # Always include the active correction. Historical metadata-only
        # examples are useful for reflection but must not execute on the
        # current uploaded PDF.
        examples = [current_example, *historical_examples]
        trainset = examples[:-self.validation_limit] or examples
        valset = examples[-self.validation_limit:]
        adapter = ExtractionRuleGEPAAdapter(context, self.evaluator)
        reflection_lm = OciReflectionLM(self.reflection_generator)
        from gepa.utils import MaxCandidateProposalsStopper, TimeoutStopCondition
        progress_callback = _GepaProgressCallback(self.progress_callback) if self.progress_callback else None
        result = gepa.optimize(seed_candidate=seed, trainset=trainset, valset=valset, adapter=adapter,
                               reflection_strategy=reflection_lm, candidate_selection_strategy="pareto",
                               max_metric_calls=self.max_metric_calls, max_reflection_cost=self.max_reflection_cost,
                               stop_callbacks=[MaxCandidateProposalsStopper(self.max_iterations),
                                               TimeoutStopCondition(self.timeout_seconds)],
                               callbacks=[progress_callback] if progress_callback else None,
                               seed=self.seed, display_progress_bar=False, track_best_outputs=True)
        best = getattr(result, "best_candidate", None) or {}
        text = best.get("extraction_instruction", "") if isinstance(best, dict) else str(best)
        fallback_to_seed = not bool(str(text).strip())
        if fallback_to_seed:
            text = seed_instruction
        try:
            sentence, format_used = parse_rule_response(text)
        except Exception:
            # GEPA may return a multi-line reflective artifact when no valid
            # mutation is selected. Keep the candidate safe and usable by
            # falling back to the compact seed instruction.
            fallback_to_seed = True
            sentence, format_used = parse_rule_response(seed_instruction)
        evaluation, _ = self.evaluator.evaluate(context, sentence)
        baseline, _ = self.evaluator.evaluate(context, seed["extraction_instruction"])
        evaluation.baseline_score = baseline.score
        if evaluation.score is not None and baseline.score is not None:
            evaluation.improvement = evaluation.score - baseline.score
        evaluation.confidence = "limited" if len(examples) == 1 else "normal"
        if evaluation.improvement is not None and evaluation.improvement <= 0:
            evaluation.termination_reason = "no_improvement"
            evaluation.candidate_status = "no_improvement"
            evaluation.promotion_eligible = False
        elif evaluation.candidate_status == "accepted" and evaluation.evidence_supported:
            evaluation.promotion_eligible = evaluation.confidence != "limited"
        details = getattr(result, "detailed_results", None)
        metadata = {"optimizer": "gepa", "optimizer_version": getattr(gepa, "__version__", "unknown"),
                    "metric_calls": getattr(details, "total_metric_calls", self.max_metric_calls),
                    "last_iteration": progress_callback.last_iteration if progress_callback else None,
                    "requested_max_iterations": self.max_iterations,
                    "requested_max_metric_calls": self.max_metric_calls,
                    "timeout_seconds": self.timeout_seconds,
                    "best_index": getattr(details, "best_idx", None),
                    "candidate_count": len(getattr(details, "candidates", []) or []),
                    "termination_reason": evaluation.termination_reason or "budget_or_convergence",
                    "evidence_supported": bool(getattr(evaluation, "score", None) is not None and evaluation.score >= .75),
                    "demonstrations_used": len(context.feedback_packet.historical_examples) if context.feedback_packet else 0,
                    "generalization_status": "limited" if len(examples) == 1 else "unverified",
                    "baseline_score": baseline.score,
                    "final_score": evaluation.score,
                    "score_improvement": evaluation.improvement,
                    "fallback_to_seed": fallback_to_seed,
                    "candidate_status": evaluation.candidate_status if not fallback_to_seed else "seed_fallback"}
        metadata["seed_source"] = "existing_rule_and_correction_context"
        metadata["usage"] = {
            "extraction_evaluation": adapter.usage.model_dump(),
            "reflection": reflection_lm.usage.model_dump(),
            "total": summarize_usage([adapter.usage, reflection_lm.usage]).model_dump(),
        }
        return GeneratedRuleCandidate(sentence=sentence, strategy=self.name, response_format=format_used,
                                      evaluation=evaluation, prompt_hash=prompt_hash(context), metadata=metadata,
                                      usage=summarize_usage([adapter.usage, reflection_lm.usage]))


def prompt_hash(context: RuleGenerationContext) -> str:
    return hashlib.sha256(json.dumps(context.model_dump(), sort_keys=True, default=str).encode()).hexdigest()
