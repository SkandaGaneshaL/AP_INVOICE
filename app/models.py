from typing import Any
from pydantic import BaseModel, Field

from .model_registry import ReasoningEffort, SentenceGenerationModel


class RuleRecord(BaseModel):
    ID: int | str
    FIELD_KEY: str
    DISPLAY_LABEL: str = ""
    SHORT_RULE: str = ""
    DETAILED_RULE: list[str] = Field(default_factory=list)


class UpdateRequest(BaseModel):
    invoice_payload: dict[str, Any]
    final_response: dict[str, Any]
    dry_run: bool = False
    allow_partial: bool = False
    enable_gepa: bool = True
    gepa_max_iterations: int = 2
    gepa_max_metric_calls: int = 5
    gepa_history_limit: int = 10
    gepa_validation_limit: int = 5
    gepa_seed: int = 0
    gepa_reflection_budget: float | None = None
    document_id: str | None = None
    rule_generation_model: SentenceGenerationModel = "gpt-oss-20b"
    reasoning_effort: ReasoningEffort = "low"


class ReasoningConfig(BaseModel):
    requested_effort: str = "low"
    effective_effort: str | None = None
    supported: bool = False
    visible_reasoning: bool = False


class GepaRunConfig(BaseModel):
    max_iterations: int = Field(default=2, ge=1, le=100)
    max_metric_calls: int = Field(default=5, ge=1, le=1000)
    reflection_budget: float | None = Field(default=None, ge=0)
    timeout_seconds: float = Field(default=120, gt=0, le=7200)
    random_seed: int = 0


class FieldChange(BaseModel):
    field_key: str
    path: str
    old_value: Any = None
    new_value: Any = None
    normalized_old_value: Any = None
    normalized_new_value: Any = None
    comparison_method: str = "structured_comparison"
    change_reason: str | None = None


class EvidenceMatch(BaseModel):
    page: int
    label: str | None = None
    value: Any = None
    raw_value: Any = None
    normalized_value: Any = None
    snippet: str = ""
    bbox: list[float] | None = None
    source: str = "pdf_text"
    confidence: float = 0.0
    match_type: str = "exact"
    transformation: str | None = None
    label_match: bool = False


class DemonstrationExample(BaseModel):
    field_key: str
    input_evidence: dict[str, Any] = Field(default_factory=dict)
    expected_output: dict[str, Any] = Field(default_factory=dict)
    failure_type: str = "unknown"
    rule_lesson: str = ""
    layout_signature: str | None = None
    source: str = "history"


class RuleFeedbackPacket(BaseModel):
    field_key: str
    field_path: str
    display_label: str = ""
    previous_value: Any = None
    corrected_value: Any = None
    original_field_node: Any = None
    corrected_field_node: Any = None
    short_rule: str = ""
    detailed_rules: list[str] = Field(default_factory=list)
    evidence: list[EvidenceMatch] = Field(default_factory=list)
    competing_evidence: list[EvidenceMatch] = Field(default_factory=list)
    observed_correction: str = ""
    inferred_intent: str = ""
    constraints: list[str] = Field(default_factory=list)
    failure_type: str = "unknown"
    historical_examples: list[DemonstrationExample] = Field(default_factory=list)
    confidence: str = "limited"


class ChangeResult(BaseModel):
    ID: int | str | None = None
    FIELD_KEY: str
    path: str
    old_value: Any = None
    new_value: Any = None
    status: str
    generated_sentence: str | None = None
    oci_request_id: str | None = None
    response_format_used: str | None = None
    reason: str | None = None
    error: dict[str, Any] | None = None
    generation: dict[str, Any] = Field(default_factory=dict)
    strategy: str = "generative"
    evaluation_score: float | None = None
    evaluation_feedback: str | None = None
    candidate_id: str | None = None
    prompt_hash: str | None = None
    evidence: list[EvidenceMatch] = Field(default_factory=list)
    competing_evidence: list[EvidenceMatch] = Field(default_factory=list)
    feedback_packet: dict[str, Any] | None = None
    confidence: str = "unknown"
    demonstrations_used: int = 0
    persistence_status: str = "not_persisted"
    candidate_status: str = "unknown"
    promotion_eligible: bool = False
    usage: UsageSummary | None = None
    model: str | None = None
    reasoning_effort_requested: str | None = None
    reasoning_effort_effective: str | None = None
    reasoning_supported: bool = False
    visible_reasoning: bool = False
    decision_summary: str | None = None
    reason: str | None = None


class RuleGenerationContext(BaseModel):
    field_key: str
    display_label: str = ""
    short_rule: str = ""
    detailed_rule: list[str] = Field(default_factory=list)
    old_value: Any = None
    new_value: Any = None
    field_path: str = ""
    invoice_payload: dict[str, Any] = Field(default_factory=dict)
    final_response: dict[str, Any] = Field(default_factory=dict)
    historical_examples: list[dict[str, Any]] = Field(default_factory=list)
    rule_version: str = "v1"
    extraction_function: str = "unconfigured"
    document_id: str | None = None
    mime_type: str = "application/pdf"
    document_bytes: bytes | None = Field(default=None, exclude=True, repr=False)
    feedback_packet: RuleFeedbackPacket | None = None
    baseline_instruction: str | None = None
    baseline_evaluation: Any = None
    normalization_mode: str = "none"
    reasoning_effort: ReasoningEffort = "low"


class EvaluationResult(BaseModel):
    score: float | None = None
    feedback: str = ""
    confidence: str = "unknown"
    baseline_score: float | None = None
    improvement: float | None = None
    termination_reason: str | None = None
    field_match: bool | None = None
    evidence_supported: bool | None = None
    schema_valid: bool | None = None
    candidate_status: str = "unknown"
    promotion_eligible: bool = False
    expected_value: Any = None
    actual_value: Any = None
    canonical_actual_value: Any = None
    transformation_expected: str | None = None


class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    reported: bool = False
    source: str = "oci"
    model: str | None = None
    request_id: str | None = None
    call_type: str | None = None
    attempt: int = 1
    usage_location: str | None = None
    reasoning_tokens_location: str | None = None
    reasoning_tokens_reported: bool = False
    reasoning_tokens_status: str = "provider_unavailable"
    usage_fields_present: list[str] = Field(default_factory=list)
    nested_detail_fields_present: list[str] = Field(default_factory=list)
    output_tokens_semantics: str = "provider_reported_output_total"


class UsageSummary(BaseModel):
    calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    reported_calls: int = 0
    unknown_calls: int = 0
    by_call_type: dict[str, "UsageSummary"] = Field(default_factory=dict)
    missing_categories: dict[str, int] = Field(default_factory=dict)
    reasoning_tokens_reported: bool = False
    reasoning_tokens_status: str = "not_applicable"
    output_tokens_semantics: str = "provider_reported_output_total"

    def add_call(self, usage: TokenUsage) -> None:
        self.calls += 1
        self.reported_calls += int(usage.reported)
        self.unknown_calls += int(not usage.reported)
        self.reasoning_tokens_reported = self.reasoning_tokens_reported or usage.reasoning_tokens_reported
        if usage.output_tokens_semantics == "may_include_reasoning_tokens":
            self.output_tokens_semantics = "may_include_reasoning_tokens"
        if usage.reasoning_tokens_reported:
            self.reasoning_tokens_status = "reported"
        elif usage.reasoning_tokens_status in {"provider_unavailable", "unsupported_response_shape"}:
            if self.reasoning_tokens_status != "reported":
                self.reasoning_tokens_status = usage.reasoning_tokens_status
        for name in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "total_tokens"):
            value = getattr(usage, name)
            if value is None:
                self.missing_categories[name] = self.missing_categories.get(name, 0) + 1
                setattr(self, name, None)
            elif not self.missing_categories.get(name):
                current = getattr(self, name)
                setattr(self, name, current + value if current is not None else value)
        if usage.call_type:
            self.by_call_type.setdefault(usage.call_type, UsageSummary()).add_call(
                usage.model_copy(update={"call_type": None})
            )

    def add_summary(self, other: "UsageSummary") -> None:
        self.calls += other.calls
        self.reported_calls += other.reported_calls
        self.unknown_calls += other.unknown_calls
        self.reasoning_tokens_reported = self.reasoning_tokens_reported or other.reasoning_tokens_reported
        if other.output_tokens_semantics == "may_include_reasoning_tokens":
            self.output_tokens_semantics = "may_include_reasoning_tokens"
        if other.reasoning_tokens_status == "reported":
            self.reasoning_tokens_status = "reported"
        elif other.reasoning_tokens_status in {"provider_unavailable", "unsupported_response_shape"}:
            if self.reasoning_tokens_status != "reported":
                self.reasoning_tokens_status = other.reasoning_tokens_status
        for name in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "total_tokens"):
            left, right = getattr(self, name), getattr(other, name)
            missing = self.missing_categories.get(name, 0) + other.missing_categories.get(name, 0)
            if missing:
                self.missing_categories[name] = missing
                setattr(self, name, None)
            else:
                setattr(self, name, left + right if left is not None and right is not None else right if left is None else left)
        for call_type, summary in other.by_call_type.items():
            self.by_call_type.setdefault(call_type, UsageSummary()).add_summary(summary)


class ProviderGenerationResult(BaseModel):
    sentence: str
    request_id: str | None = None
    response_format: str | None = None
    usage: UsageSummary = Field(default_factory=UsageSummary)
    model: str | None = None
    finish_reason: str | None = None
    attempts: int = 1
    usage_location: str | None = None
    reasoning_effort_requested: str | None = None
    reasoning_effort_effective: str | None = None
    reasoning_supported: bool = False
    visible_reasoning: bool = False
    decision_summary: str | None = None
    reason: str | None = None
    reasoning_summary_available: bool = False
    reasoning_mode: str = "safe_decision_summary"
    reasoning_parameter_sent: bool = False
    verbosity_parameter_sent: bool = False
    usage_diagnostics: dict[str, Any] = Field(default_factory=dict)


class StrategyResult(BaseModel):
    strategy: str
    updated_rules: list[RuleRecord]
    changes: list[ChangeResult]
    summary: dict[str, int]
    evaluation: EvaluationResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    usage: UsageSummary | None = None


class UpdateResponse(BaseModel):
    strategies: list[StrategyResult] = Field(default_factory=list)
    updated_rules: list[RuleRecord]
    changes: list[ChangeResult]
    summary: dict[str, int]
    gepa_job_id: str | None = None
    gepa_status: str | None = None


class UpdateJobCreateResponse(BaseModel):
    job_id: str
    status: str
    normal_status: str
    gepa_status: str
    requested_config: dict[str, Any] = Field(default_factory=dict)
    effective_config: dict[str, Any] = Field(default_factory=dict)
    requested_model: str = "gpt-oss-20b"
    effective_model: str = "openai.gpt-oss-20b"
    extraction_model: str = "google.gemini-2.5-flash"
    change_detection: dict[str, Any] = Field(default_factory=dict)
    oci_sentence_generation_called: bool = False
    oci_sentence_generation_call_count: int = 0
    reasoning: ReasoningConfig | None = None
    reasoning_ui_contract_version: str = "safe-summary-v1"


class UpdateJobStatusResponse(BaseModel):
    job_id: str
    status: str
    phase: str
    normal_status: str
    gepa_status: str
    normal_result: StrategyResult | None = None
    gepa_result: StrategyResult | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    requested_config: dict[str, Any] = Field(default_factory=dict)
    effective_config: dict[str, Any] = Field(default_factory=dict)
    termination: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    reasoning: ReasoningConfig | None = None
    requested_model: str = "gpt-oss-20b"
    effective_model: str = "openai.gpt-oss-20b"
    extraction_model: str = "google.gemini-2.5-flash"
    change_detection: dict[str, Any] = Field(default_factory=dict)
    oci_sentence_generation_called: bool = False
    oci_sentence_generation_call_count: int = 0
    reasoning_ui_contract_version: str = "safe-summary-v1"


class GepaJobResponse(BaseModel):
    job_id: str
    status: str
    progress: dict[str, Any] = Field(default_factory=dict)
    strategy: StrategyResult | None = None
    error: dict[str, Any] | None = None


class PromoteRequest(BaseModel):
    candidate_id: str
    expected_rule_version: str = "v1"
    dry_run: bool = False


class PromoteBatchRequest(BaseModel):
    candidate_ids: list[str] = Field(min_length=1)
    expected_rule_version: str = "v1"
    confirm: bool = False
    dry_run: bool = False


class PromoteBatchResponse(BaseModel):
    status: str
    code: str | None = None
    rule_version: str = "v1"
    promoted_candidates: list[dict[str, Any]] = Field(default_factory=list)
    backup_created: bool = False
    reason: str | None = None


class PromoteResponse(BaseModel):
    candidate_id: str
    status: str
    code: str | None = None
    rule_id: int | str | None = None
    field_key: str | None = None
    sentence: str | None = None
    reason: str | None = None


class ExtractionResponse(BaseModel):
    document_id: str
    filename: str
    extracted_json: dict[str, Any]
    model: str
    page_count: int | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    usage: UsageSummary | None = None


class StoredDocument(BaseModel):
    document_id: str
    filename: str
    mime_type: str
    document_bytes: bytes = Field(exclude=True, repr=False)
