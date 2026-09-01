from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .model_output import parse_rule_response
from .models import RuleGenerationContext
from .models import ProviderGenerationResult, TokenUsage, UsageSummary


class OciReflectionLM:
    """GEPA ReflectionLM backed by the existing OCI Generative AI client."""

    def __init__(self, generator):
        self.generator = generator
        self.usage = UsageSummary()

    def reflect(self, candidate: dict[str, str], reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
                components_to_update: list[str]):
        from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal

        component = components_to_update[0]
        side_info = json.dumps({k: list(v) for k, v in reflective_dataset.items()}, ensure_ascii=False, default=str)
        prompt = {
            "current_candidate": candidate.get(component, ""),
            "component": component,
            "feedback_examples": side_info,
            "instruction": "Improve only the extraction instruction using the feedback. Preserve correct behavior, do not hard-code invoice values, and return one reusable instruction.",
        }
        context = RuleGenerationContext(
            field_key=component,
            display_label="Extraction instruction",
            short_rule="Rewrite the candidate instruction using diagnostic feedback.",
            detailed_rule=[candidate.get(component, ""), "Feedback: " + side_info],
            old_value="current instruction",
            new_value="improved instruction",
            field_path=component,
            invoice_payload=prompt,
            final_response={"instruction": "one improved reusable extraction instruction"},
        )
        try:
            if hasattr(self.generator, "generate_with_metadata"):
                generated = self.generator.generate_with_metadata(context=context)
                if isinstance(generated, ProviderGenerationResult):
                    sentence, request_id = generated.sentence, generated.request_id
                    self.usage.add_summary(generated.usage)
                else:
                    sentence, request_id, _ = generated
                    self.usage.add_call(TokenUsage(call_type="reflection"))
            else:
                sentence, request_id, _ = self.generator.generate(context=context)
                self.usage.add_call(TokenUsage(call_type="reflection"))
        except TypeError as exc:
            if "context" not in str(exc) and "unexpected keyword" not in str(exc):
                raise
            sentence, request_id, _ = self.generator.generate(
                field_key=component, display_label="Extraction instruction",
                short_rule=context.short_rule, detailed_rule=context.detailed_rule,
                old_value=context.old_value, new_value=context.new_value,
                field_path=component, invoice_payload=prompt,
                final_response=context.final_response,
            )
            self.usage.add_call(TokenUsage(call_type="reflection"))
        return ReflectionProposal(
            new_texts={component: sentence},
            prompts={component: json.dumps(prompt, ensure_ascii=False)},
            raw_lm_outputs={component: sentence},
            metadata={"oci_request_id": request_id, "usage": self.usage.model_dump()},
        ), self
