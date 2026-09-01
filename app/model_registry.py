from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


SentenceGenerationModel = Literal["gpt-oss-20b", "gpt-4o", "gemini-2.5-flash"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]

REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high")
REASONING_EFFORT_ENUMS = {
    "none": "NONE",
    "minimal": "MINIMAL",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
}


@dataclass(frozen=True)
class RuleGenerationModel:
    key: SentenceGenerationModel
    model_id: str
    display_name: str
    transport: str


def rule_generation_settings(model: RuleGenerationModel) -> dict[str, str | None]:
    """Resolve endpoint settings for the selected sentence model only.

    Model-specific variables take precedence over the legacy generic
    variables.  This is important because extraction and sentence generation
    intentionally run in different OCI regions and compartments.
    """
    if model.key == "gpt-oss-20b":
        prefix = "OCI_GPT_RULE"
        legacy_region = "OCI_RULE_GENERATION_REGION"
        legacy_endpoint = "OCI_RULE_GENERATION_ENDPOINT"
        legacy_compartment = "OCI_RULE_GENERATION_COMPARTMENT_ID"
        legacy_mode = "OCI_RULE_GENERATION_SERVING_MODE"
        legacy_endpoint_id = "OCI_RULE_GENERATION_ENDPOINT_ID"
    elif model.key == "gpt-4o":
        prefix = "OCI_GPT4O_RULE"
        legacy_region = "OCI_RULE_GENERATION_GPT4O_REGION"
        legacy_endpoint = "OCI_RULE_GENERATION_GPT4O_ENDPOINT"
        legacy_compartment = "OCI_RULE_GENERATION_GPT4O_COMPARTMENT_ID"
        legacy_mode = "OCI_RULE_GENERATION_GPT4O_SERVING_MODE"
        legacy_endpoint_id = "OCI_RULE_GENERATION_GPT4O_ENDPOINT_ID"
    else:
        prefix = "OCI_GEMINI_RULE"
        legacy_region = "OCI_RULE_GENERATION_GEMINI_REGION"
        legacy_endpoint = "OCI_RULE_GENERATION_GEMINI_ENDPOINT"
        legacy_compartment = "OCI_RULE_GENERATION_GEMINI_COMPARTMENT_ID"
        legacy_mode = "OCI_RULE_GENERATION_GEMINI_SERVING_MODE"
        legacy_endpoint_id = "OCI_RULE_GENERATION_GEMINI_ENDPOINT_ID"

    endpoint = os.getenv(f"{prefix}_ENDPOINT") or os.getenv(legacy_endpoint)
    endpoint_region = None
    if endpoint:
        match = re.search(r"inference\.generativeai\.([a-z0-9-]+)\.oci\.oraclecloud\.com", endpoint)
        endpoint_region = match.group(1) if match else None
    region = (os.getenv(f"{prefix}_REGION") or os.getenv(legacy_region)
              or endpoint_region or os.getenv("OCI_RULE_GENERATION_REGION")
              or os.getenv("OCI_REGION"))
    compartment = (os.getenv(f"{prefix}_COMPARTMENT_ID")
                   or os.getenv(legacy_compartment)
                   or os.getenv("OCI_COMPARTMENT_ID"))
    mode = (os.getenv(f"{prefix}_SERVING_MODE") or os.getenv(legacy_mode)
            or "on_demand").lower()
    endpoint_id = os.getenv(f"{prefix}_ENDPOINT_ID") or os.getenv(legacy_endpoint_id)
    # A regional URL is not a dedicated endpoint OCID.  Treat legacy values
    # that accidentally contain a URL as unset so dedicated mode fails before
    # making an OCI request instead of sending an invalid resource identifier.
    if endpoint_id and endpoint_id.lower().startswith(("http://", "https://")):
        endpoint_id = None
    return {
        "region": region,
        "endpoint": endpoint,
        "compartment_id": compartment,
        "serving_mode": mode,
        "endpoint_id": endpoint_id,
    }


class RuleGenerationConfigurationError(ValueError):
    """Configuration is invalid for a valid sentence-generation model."""

    def __init__(self, code: str, message: str, *, model: RuleGenerationModel | str, region: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.model = model.model_id if isinstance(model, RuleGenerationModel) else str(model)
        self.region = region


RULE_GENERATION_MODELS: dict[str, RuleGenerationModel] = {
    "gpt-oss-20b": RuleGenerationModel(
        key="gpt-oss-20b",
        model_id="openai.gpt-oss-20b",
        display_name="GPT-OSS 20B",
        transport="oci_native",
    ),
    "gpt-4o": RuleGenerationModel(
        key="gpt-4o",
        model_id="openai.gpt-4o",
        display_name="GPT-4o",
        transport="oci_native",
    ),
    "gemini-2.5-flash": RuleGenerationModel(
        key="gemini-2.5-flash",
        model_id="google.gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        transport="oci_native",
    ),
}


def normalize_reasoning_effort(value: str | None) -> str:
    selected = (value or os.getenv("OCI_RULE_GENERATION_REASONING_EFFORT", "low")).strip().lower()
    if selected not in REASONING_EFFORT_VALUES:
        allowed = ", ".join(REASONING_EFFORT_VALUES)
        raise ValueError(f"Reasoning effort must be one of {allowed}")
    return selected


def reasoning_supported(model: RuleGenerationModel | str) -> bool:
    model_id = model.model_id if isinstance(model, RuleGenerationModel) else str(model)
    return model_id == "openai.gpt-oss-20b"


def reasoning_config(model: RuleGenerationModel | str, requested_effort: str | None = None) -> dict[str, object]:
    requested = normalize_reasoning_effort(requested_effort)
    supported = reasoning_supported(model)
    return {
        "requested_effort": requested,
        "effective_effort": REASONING_EFFORT_ENUMS[requested] if supported else None,
        "supported": supported,
        "visible_reasoning": False,
    }


def resolve_rule_generation_model(key: str | None) -> RuleGenerationModel:
    selected = key or "gpt-oss-20b"
    try:
        model = RULE_GENERATION_MODELS[selected]
        configured_id = (
            os.getenv("OCI_GPT_RULE_MODEL_ID") or os.getenv("OCI_RULE_GENERATION_MODEL_ID")
            if selected == "gpt-oss-20b"
            else os.getenv("OCI_GPT4O_RULE_MODEL_ID") or os.getenv("OCI_RULE_GENERATION_GPT4O_MODEL_ID")
            if selected == "gpt-4o"
            else os.getenv("OCI_GEMINI_RULE_MODEL_ID") or os.getenv("OCI_RULE_GENERATION_GEMINI_MODEL_ID")
        )
        return RuleGenerationModel(
            key=model.key,
            model_id=configured_id or model.model_id,
            display_name=model.display_name,
            transport=model.transport,
        )
    except KeyError as exc:
        allowed = ", ".join(sorted(RULE_GENERATION_MODELS))
        raise ValueError(f"unsupported sentence-generation model {selected!r}; choose one of {allowed}") from exc


def validate_rule_generation_configuration(model: RuleGenerationModel) -> None:
    settings = rule_generation_settings(model)
    region = settings["region"]
    if not region:
        raise RuleGenerationConfigurationError(
            "RULE_GENERATION_REGION_INVALID",
            "No OCI region is configured for sentence generation",
            model=model,
            region="unknown",
        )
    mode_var = {
        "gpt-oss-20b": "OCI_GPT_RULE_SERVING_MODE",
        "gpt-4o": "OCI_GPT4O_RULE_SERVING_MODE",
        "gemini-2.5-flash": "OCI_GEMINI_RULE_SERVING_MODE",
    }[model.key]
    mode = settings["serving_mode"] or "on_demand"
    if mode not in {"on_demand", "dedicated"}:
        raise RuleGenerationConfigurationError(
            "RULE_GENERATION_REGION_INVALID",
            f"{mode_var} must be 'on_demand' or 'dedicated'",
            model=model,
            region=region,
        )
    if mode == "dedicated" and not settings["endpoint_id"]:
        raise RuleGenerationConfigurationError(
            "MODEL_ENDPOINT_CONFIGURATION_ERROR",
            f"{model.model_id} sentence generation requires a dedicated endpoint ID",
            model=model,
            region=region,
        )
    if not settings["compartment_id"]:
        raise RuleGenerationConfigurationError(
            "RULE_GENERATION_COMPARTMENT_MISSING",
            "OCI_RULE_GENERATION_COMPARTMENT_ID or OCI_COMPARTMENT_ID must be configured",
            model=model,
            region=region,
        )
    if not str(settings["compartment_id"]).startswith("ocid1.compartment.oc1."):
        raise RuleGenerationConfigurationError(
            "RULE_GENERATION_COMPARTMENT_INVALID",
            "The configured sentence-generation compartment OCID is invalid",
            model=model,
            region=region,
        )
