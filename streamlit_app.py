import hashlib
import json
import os

import httpx
import streamlit as st
from dotenv import load_dotenv
from app.model_registry import resolve_rule_generation_model, rule_generation_settings

load_dotenv()

st.set_page_config(page_title="OCI extraction rule updater", layout="wide")
st.title("OCI invoice extraction-rule updater")
api_url = st.sidebar.text_input("FastAPI URL", os.getenv("API_URL", "http://127.0.0.1:8000"))
request_timeout_seconds = float(os.getenv("STREAMLIT_REQUEST_TIMEOUT_SECONDS", "1800"))
connect_timeout_seconds = float(os.getenv("STREAMLIT_CONNECT_TIMEOUT_SECONDS", "30"))

for key, default in (
    ("document_id", None),
    ("extracted_json", {}),
    ("corrected_json", {}),
    ("extraction_diagnostics", {}),
    ("extraction_usage", None),
    ("pdf_signature", None),
    ("candidate_data", None),
    ("update_job_id", None),
    ("update_job_data", None),
    ("candidate_selections", {}),
    ("promotion_message", None),
    ("sse_last_event_id", 0),
    ("sse_status", "not_connected"),
    ("sse_decision_summaries", {}),
    ("invoice_payload_editor", "{}"),
    ("corrected_payload_editor", "{}"),
    ("rule_generation_model", "gpt-oss-20b"),
    ("rule_generation_model_label", "GPT-OSS 20B"),
    ("reasoning_effort_label", "Low"),
):
    st.session_state.setdefault(key, default)

uploaded_pdf = st.file_uploader("Invoice PDF", type=["pdf"])
if uploaded_pdf:
    signature = hashlib.sha256(uploaded_pdf.getvalue()).hexdigest()
    if signature != st.session_state.pdf_signature:
        st.session_state.pdf_signature = signature
        st.session_state.document_id = None
        st.session_state.extracted_json = {}
        st.session_state.corrected_json = {}
        st.session_state.extraction_diagnostics = {}
        st.session_state.extraction_usage = None
        st.session_state.candidate_data = None
        st.session_state.update_job_id = None
        st.session_state.update_job_data = None
        st.session_state.candidate_selections = {}
        st.session_state.promotion_message = None
        st.session_state.sse_last_event_id = 0
        st.session_state.sse_status = "not_connected"
        st.session_state.sse_decision_summaries = {}
        st.session_state.invoice_payload_editor = "{}"
        st.session_state.corrected_payload_editor = "{}"

extract_clicked = st.button("Extract invoice", disabled=uploaded_pdf is None, icon=":material/document_scanner:")
if extract_clicked and uploaded_pdf:
    try:
        response = httpx.post(
            f"{api_url.rstrip('/')}/v1/invoices/extract",
            files={"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")},
            timeout=httpx.Timeout(request_timeout_seconds, connect=connect_timeout_seconds),
        )
        if response.is_error:
            st.error(response.text)
        else:
            data = response.json()
            extracted = data.get("extracted_json", {})
            st.session_state.document_id = data["document_id"]
            st.session_state.extracted_json = extracted
            st.session_state.corrected_json = json.loads(json.dumps(extracted))
            st.session_state.extraction_diagnostics = data.get("diagnostics", {})
            st.session_state.extraction_usage = data.get("usage") or (data.get("diagnostics") or {}).get("usage")
            st.session_state.candidate_data = None
            st.session_state.invoice_payload_editor = json.dumps(extracted, ensure_ascii=False, indent=2)
            st.session_state.corrected_payload_editor = json.dumps(extracted, ensure_ascii=False, indent=2)
            st.rerun()
    except Exception as exc:
        st.error(f"Extraction failed: {exc}")

def _token_display(value):
    if value is None:
        return "Unavailable from OCI"
    if isinstance(value, bool):
        return "Unavailable from OCI"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "Unavailable from OCI"


def render_usage(title, usage, *, unavailable_message="Unavailable from OCI"):
    """Render provider-reported usage without estimating missing categories."""
    if not usage:
        st.caption(f"{title}: Not available")
        return
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    usage = usage or {}
    st.markdown(f"**{title}**")
    columns = st.columns(3)
    values = [
        ("Calls", usage.get("calls")),
        ("Input tokens", usage.get("input_tokens")),
        ("Output tokens", usage.get("output_tokens")),
        ("Cached tokens", usage.get("cached_tokens")),
        ("Reasoning tokens", usage.get("reasoning_tokens")),
        ("Total tokens", usage.get("total_tokens")),
    ]
    for index, (label, value) in enumerate(values):
        columns[index % 3].metric(label, _token_display(value))
    semantics = usage.get("output_tokens_semantics")
    if semantics == "may_include_reasoning_tokens":
        st.caption("Output tokens may include hidden reasoning tokens.")
    reasoning_status = usage.get("reasoning_tokens_status")
    if usage.get("reasoning_tokens") is None and reasoning_status:
        st.caption(f"Reasoning-token status: {reasoning_status}")
    with st.expander(f"{title} diagnostics"):
        st.json({
            key: usage.get(key)
            for key in (
                "reported_calls", "unknown_calls", "missing_categories",
                "reasoning_tokens_reported", "reasoning_tokens_status",
                "output_tokens_semantics",
            )
            if key in usage
        })


left, right = st.columns(2)
with left:
    st.subheader("Original extracted JSON")
    st.text_area(
        "Original extracted JSON",
        key="invoice_payload_editor",
        height=460,
        disabled=True,
        label_visibility="collapsed",
    )
    if st.session_state.extraction_diagnostics:
        st.caption("Extraction diagnostics")
        st.json(st.session_state.extraction_diagnostics)
    if st.session_state.extraction_usage:
        render_usage("PDF extraction usage", st.session_state.extraction_usage)
with right:
    st.subheader("User-corrected JSON")
    corrected_text = st.text_area(
        "User-corrected JSON",
        key="corrected_payload_editor",
        height=460,
        label_visibility="collapsed",
    )

st.divider()
st.subheader("Rule-candidate settings")
dry_run = st.checkbox("Dry run (do not persist rules or audit)")
allow_partial = st.checkbox("Allow partial updates")
model_labels = {
    "GPT-OSS 20B": "gpt-oss-20b",
    "GPT-4o": "gpt-4o",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
}
selected_label = st.segmented_control(
    "Sentence-generation model",
    list(model_labels),
    key="rule_generation_model_label",
) or "GPT-OSS 20B"
st.session_state.rule_generation_model = model_labels[selected_label]
st.caption("PDF extraction model: Gemini 2.5 Flash")
st.caption(f"PDF extraction region: {os.getenv('OCI_EXTRACTION_REGION') or os.getenv('OCI_EXC_REGION', 'not configured')}")
st.caption(f"PDF extraction project: {'configured' if os.getenv('OCI_EXTRACTION_PROJECT_ID') or os.getenv('PROJECT_ID') else 'not configured'}")
st.caption(f"Sentence-generation model: {selected_label}")
selected_model = resolve_rule_generation_model(st.session_state.rule_generation_model)
sentence_settings = rule_generation_settings(selected_model)
sentence_mode = sentence_settings.get("serving_mode") or "on_demand"
endpoint_configured = bool(sentence_settings.get("endpoint_id"))
st.caption(f"Effective model: {selected_model.model_id}")
st.caption(f"Sentence-generation region: {sentence_settings.get('region') or 'not configured'}")
st.caption(f"Sentence-generation serving mode: {sentence_mode}")
reasoning_labels = {"None": "none", "Minimal": "minimal", "Low": "low", "Medium": "medium", "High": "high"}
reasoning_supported = selected_model.model_id == "openai.gpt-oss-20b"
reasoning_label = st.segmented_control(
    "Reasoning effort",
    list(reasoning_labels),
    key="reasoning_effort_label",
    disabled=not reasoning_supported,
) or "Low"
st.session_state.reasoning_effort = reasoning_labels[reasoning_label]
if sentence_mode == "dedicated" and not endpoint_configured:
    st.error(f"{selected_label} cannot be used until its dedicated endpoint ID is configured.")

generate_clicked = st.button(
    "Generate extraction-rule candidates",
    type="primary",
    disabled=not st.session_state.document_id or (sentence_mode == "dedicated" and not endpoint_configured),
    icon=":material/auto_awesome:",
)
if generate_clicked:
    try:
        payload = {
            "invoice_payload": json.loads(st.session_state.invoice_payload_editor),
            "final_response": json.loads(corrected_text),
            "dry_run": dry_run,
            "allow_partial": allow_partial,
            "document_id": st.session_state.document_id,
            "rule_generation_model": st.session_state.rule_generation_model,
            "reasoning_effort": st.session_state.reasoning_effort,
        }
        with st.spinner("Creating normal-generation job..."):
            response = httpx.post(
                f"{api_url.rstrip('/')}/v1/extraction-rules/update",
                json=payload,
                timeout=httpx.Timeout(30, connect=10),
            )
        if response.is_error:
            st.error(f"API error {response.status_code}: {response.text}")
        else:
            data = response.json()
            st.session_state.update_job_id = data.get("job_id")
            st.session_state.update_job_data = data
            st.session_state.candidate_data = None
            st.session_state.candidate_selections = {}
            st.session_state.promotion_message = None
            st.session_state.sse_last_event_id = 0
            st.session_state.sse_status = "not_connected"
            st.session_state.sse_decision_summaries = {}
            st.success("Normal OCI candidate-generation job created")
    except json.JSONDecodeError as exc:
        st.error(f"Corrected JSON is invalid: {exc}")
    except Exception as exc:
        st.error(f"Request failed: {exc}")

def render_strategy(strategy, title, key_suffix):
    if not strategy:
        return
    st.subheader(title)
    st.dataframe(strategy.get("changes", []), width="stretch")
    metadata = strategy.get("metadata") or {}
    evaluation = strategy.get("evaluation") or {}
    if evaluation:
        st.caption(
            f"Evaluation: {evaluation.get('candidate_status', 'unavailable')} · "
            f"score: {_token_display(evaluation.get('score'))}"
        )
    st.info(f"Persistence: {metadata.get('persistence_status', 'not_persisted')}")
    model = metadata.get("model")
    if model:
        st.caption(f"Sentence-generation model: {model}")
    usage = strategy.get("usage")
    if usage:
        render_usage("Rule-generation usage", usage)
    for change in strategy.get("changes", []):
        generation = change.get("generation") or {}
        if change.get("generated_sentence"):
            st.markdown(f"**{change.get('FIELD_KEY', 'Field')} instruction**")
            st.write(change["generated_sentence"])
            st.caption(f"Correction kind: {change.get('correction_kind') or 'not classified'}")
            st.caption(f"LLM calls: {1 if change.get('oci_request_id') else 0}")
            reason = change.get("reason")
            if reason:
                with st.expander("Business explanation", expanded=True):
                    st.write(reason)
        elif change.get("status") in {"generation_failed", "unavailable"}:
            reason = change.get("reason", "candidate generation failed")
            st.error(f"{change.get('FIELD_KEY', 'Field')}: {reason}")
        if generation:
            with st.expander("Generation diagnostics", expanded=False):
                st.write({
                    key: generation.get(key)
                    for key in ("attempts", "application_output_limit_sent", "provider_managed_limit",
                                "finish_reasons", "request_ids", "reason")
                    if key in generation
                })
                if generation.get("reason") == "provider_output_truncated_after_repair":
                    st.warning(
                        "OCI reached its provider-managed output limit before returning a complete rule. "
                        "The rule was not saved."
                    )
        evidence = change.get("evidence", [])
        competing = change.get("competing_evidence", [])
        if evidence or competing:
            with st.expander(f"Evidence used for {change.get('FIELD_KEY', 'field')}"):
                st.write({"previous_value": change.get("old_value"), "corrected_value": change.get("new_value"),
                          "confidence": change.get("confidence"), "demonstrations_used": change.get("demonstrations_used", 0)})
                if evidence:
                    st.dataframe(evidence, width="stretch", hide_index=True)
                if competing:
                    st.caption("Competing evidence")
                    st.dataframe(competing, width="stretch", hide_index=True)
    st.download_button(f"Download {title} preview rules", json.dumps(strategy.get("updated_rules", []), ensure_ascii=False, indent=2),
                       file_name=f"{key_suffix}_preview_rules.json", mime="application/json", key=f"download-{key_suffix}")


def _eligible(change):
    return bool(
        change
        and change.get("candidate_id")
        and change.get("status") == "preview"
        and change.get("candidate_status") not in {"rejected", "generation_failed", "unavailable"}
        and change.get("promotion_eligible") is True
        and change.get("persistence_status") == "awaiting_approval"
    )


def render_candidate_approval(job):
    normal_changes = {item.get("FIELD_KEY"): item for item in (job.get("normal_result") or {}).get("changes", [])}
    fields = list(normal_changes.keys())
    if not fields:
        return
    st.subheader("Approve candidates")
    st.caption("Preview generation never changes extraction_rules.json. Select candidates, then explicitly approve them.")
    selected_ids = []
    for field_key in fields:
        normal = normal_changes.get(field_key)
        options = ["No change"]
        if _eligible(normal):
            options.append("Normal OCI")
        choice = st.segmented_control(
            f"Candidate for {field_key}", options, default="No change",
            key=f"candidate-selection-{field_key}",
        ) or "No change"
        if choice == "Normal OCI" and normal:
            selected_ids.append(normal["candidate_id"])
    if selected_ids:
        st.warning(f"Selected candidates: {len(selected_ids)}. Production rules are still unchanged.")
    approve = st.button(
        "Approve selected candidates and update extraction rules",
        disabled=not selected_ids,
        type="primary",
        icon=":material/publish:",
        key="approve-selected-candidates",
    )
    if approve:
        response = httpx.post(
            f"{api_url.rstrip('/')}/v1/extraction-rules/promote-batch",
            json={"candidate_ids": selected_ids, "expected_rule_version": "v1", "confirm": True, "dry_run": False},
            timeout=httpx.Timeout(60, connect=connect_timeout_seconds),
        )
        if response.is_error:
            st.error(f"Approval failed: {response.text}")
        else:
            result = response.json()
            if result.get("status") == "promoted":
                st.success("Approved candidates were persisted to extraction_rules.json")
                st.session_state.promotion_message = result
                st.session_state.candidate_selections = {}
                st.rerun()
            else:
                st.error(result.get("reason", "Approval was rejected"))


def render_update_job(job):
    normal = job.get("normal_result")
    if normal:
        metadata = normal.get("metadata") or {}
        if metadata.get("reason") == "No changed mapped fields were detected":
            st.subheader("Generative OCI")
            st.info("No rule candidate was generated. No changed mapped fields were detected.")
            st.caption("No OCI sentence-generation call was made.")
            st.caption("Rule-generation usage: Not applicable.")
            return
        render_strategy(normal, "Generative OCI", "generative")
    elif job.get("normal_status") in {"queued", "running"}:
        st.subheader("Generative OCI")
        st.info("Normal OCI generation is running...")


def consume_update_events(job_id):
    """Read currently available safe SSE events without blocking the UI rerun."""
    events = []
    try:
        with httpx.stream(
            "GET",
            f"{api_url.rstrip('/')}/v1/extraction-rules/update-jobs/{job_id}/events",
            headers={"Last-Event-ID": str(st.session_state.sse_last_event_id)},
            timeout=httpx.Timeout(2, connect=10),
        ) as response:
            if response.is_error:
                st.session_state.sse_status = "fallback"
                return events
            st.session_state.sse_status = "connected"
            current = {}
            for line in response.iter_lines():
                if not line:
                    if current.get("event") and current.get("data"):
                        try:
                            payload = json.loads(current["data"])
                        except json.JSONDecodeError:
                            payload = {}
                        events.append((current["event"], payload))
                    current = {}
                    if len(events) >= 20:
                        break
                elif line.startswith("id:"):
                    try:
                        st.session_state.sse_last_event_id = int(line[3:].strip())
                    except ValueError:
                        pass
                elif line.startswith("event:"):
                    current["event"] = line[6:].strip()
                elif line.startswith("data:"):
                    current["data"] = line[5:].strip()
    except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError):
        if st.session_state.sse_status != "connected":
            st.session_state.sse_status = "fallback"
    return events


if st.session_state.update_job_id:
    @st.fragment(run_every="2s")
    def poll_update_job():
        try:
            sse_events = consume_update_events(st.session_state.update_job_id)
            for event_type, event_data in sse_events:
                if event_type == "decision_summary" and event_data.get("field_key"):
                    st.session_state.sse_decision_summaries[event_data["field_key"]] = event_data.get("summary")
            response = httpx.get(f"{api_url.rstrip('/')}/v1/extraction-rules/update-jobs/{st.session_state.update_job_id}",
                                 timeout=httpx.Timeout(30, connect=10))
            if response.is_error:
                st.error(f"Update job status failed: {response.text}")
                return
            job = response.json()
            normal_result = job.get("normal_result") or {}
            for change in normal_result.get("changes", []):
                if not change.get("reason") and change.get("FIELD_KEY") in st.session_state.sse_decision_summaries:
                    change["reason"] = st.session_state.sse_decision_summaries[change["FIELD_KEY"]]
            st.session_state.update_job_data = job
            st.caption(f"Generation phase: {job.get('phase')}")
            detection = job.get("change_detection") or {}
            if detection:
                st.caption(f"Change detection: {detection.get('reason', 'unknown')}")
                ignored_pages = detection.get("ignored_page_only_fields") or []
                ignored_unchanged = detection.get("ignored_unchanged_fields") or []
                ignored_unmapped = detection.get("ignored_unmapped_fields") or []
                if ignored_pages or ignored_unchanged or ignored_unmapped:
                    with st.expander("Change-detection details"):
                        st.json({
                            "changed_fields": detection.get("changed_fields", []),
                            "ignored_page_only_fields": ignored_pages,
                            "ignored_unchanged_fields": ignored_unchanged,
                            "ignored_unmapped_fields": ignored_unmapped,
                        })
            st.caption(
                f"Sentence-generation OCI call: {'made' if job.get('oci_sentence_generation_called') else 'not made'} "
                f"({job.get('oci_sentence_generation_call_count', 0)} call(s))"
            )
            st.caption(
                f"PDF extraction model: {job.get('extraction_model', 'google.gemini-2.5-flash')} · "
                f"Sentence-generation model: {job.get('effective_model', job.get('requested_model', 'gpt-oss-20b'))}"
            )
            progress = job.get("progress", {})
            st.info(f"Normal fields: {progress.get('normal_completed_fields', 0)} / {progress.get('normal_total_fields', 0)}")
            if job.get("error"):
                st.error((job["error"] or {}).get("reason", "Update job failed"))
            render_update_job(job)
            render_candidate_approval(job)
        except Exception as exc:
            st.warning(f"Update status request failed: {exc}")

    poll_update_job()
