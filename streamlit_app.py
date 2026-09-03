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
    ("supplier_key", ""),
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


def extraction_table_rows(payload):
    """Project public extraction JSON into stable, display-only table rows."""
    rows = []
    for field, node in (payload or {}).items():
        if isinstance(node, dict) and {"value", "Page"}.issubset(node):
            rows.append({"Field": field, "Value": node.get("value"), "Page": node.get("Page")})
        elif isinstance(node, list):
            for index, item in enumerate(node, start=1):
                if isinstance(item, dict) and {"value", "Page"}.issubset(item):
                    rows.append({"Field": f"{field}[{index}]", "Value": item.get("value"), "Page": item.get("Page")})
                elif isinstance(item, dict):
                    for child, child_node in item.items():
                        if isinstance(child_node, dict) and {"value", "Page"}.issubset(child_node):
                            rows.append({"Field": f"{field}[{index}].{child}", "Value": child_node.get("value"), "Page": child_node.get("Page")})
                        else:
                            rows.append({"Field": f"{field}[{index}].{child}", "Value": child_node, "Page": None})
                else:
                    rows.append({"Field": f"{field}[{index}]", "Value": item, "Page": None})
        else:
            rows.append({"Field": field, "Value": node, "Page": None})
    return rows


left, right = st.columns(2)
with left:
    st.subheader("Original extracted fields")
    if st.session_state.extracted_json:
        st.dataframe(
            extraction_table_rows(st.session_state.extracted_json),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Upload and extract an invoice to view its fields.")
    with st.expander("Original extraction JSON", expanded=False):
        st.code(st.session_state.invoice_payload_editor, language="json")
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
    try:
        corrected_preview = json.loads(corrected_text or "{}")
    except json.JSONDecodeError:
        corrected_preview = {}
    if corrected_preview:
        with st.expander("User-corrected fields", expanded=False):
            st.dataframe(extraction_table_rows(corrected_preview), width="stretch", hide_index=True)

st.divider()
st.subheader("Rule-candidate settings")
st.text_input(
    "Supplier scope key (optional)",
    key="supplier_key",
    help="Promoted candidates are saved to this supplier overlay when provided.",
)
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
selected_model = resolve_rule_generation_model(st.session_state.rule_generation_model)
sentence_settings = rule_generation_settings(selected_model)
sentence_mode = sentence_settings.get("serving_mode") or "on_demand"
endpoint_configured = bool(sentence_settings.get("endpoint_id"))
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
            "supplier_key": st.session_state.supplier_key.strip() or None,
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
    # Render only changed fields; the unchanged rule snapshot is not a
    # generated candidate and is intentionally omitted from the main UI.
    metadata = strategy.get("metadata") or {}
    evaluation = strategy.get("evaluation") or {}
    if evaluation:
        st.caption(
            f"Evaluation: {evaluation.get('candidate_status', 'unavailable')} · "
            f"score: {_token_display(evaluation.get('score'))}"
        )
    st.info(f"Persistence: {metadata.get('persistence_status', 'not_persisted')}")
    usage = strategy.get("usage")
    if usage:
        render_usage("Rule-generation usage", usage)
    changes = strategy.get("changes", [])
    summary_rows = []
    for change in changes:
        if not change.get("generated_sentence") and change.get("status") not in {"generation_failed", "unavailable"}:
            continue
        change_usage = change.get("usage") or {}
        summary_rows.append({
            "Field": change.get("FIELD_KEY", "Field"),
            "Old value": change.get("old_value"),
            "New value": change.get("new_value"),
            "Correction": change.get("observed_correction") or change.get("correction_kind") or "not classified",
            "Status": change.get("status", "unknown"),
            "OCI calls": change_usage.get("calls", change.get("oci_calls", 0)),
        })
    if summary_rows:
        st.dataframe(summary_rows, width="stretch", hide_index=True)
    for change in changes:
        generation = change.get("generation") or {}
        if change.get("generated_sentence"):
            st.markdown(f"**Field:** {change.get('FIELD_KEY', 'Field')}")
            st.write(f"Old value: {change.get('old_value')}")
            st.write(f"New value: {change.get('new_value')}")
            st.caption(f"Observed correction: {change.get('observed_correction') or 'not classified'}")
            st.markdown("**Generated sentence**")
            st.write(change["generated_sentence"])
            st.caption(f"Correction kind: {change.get('correction_kind') or 'not classified'}")
            st.caption(f"Status: {change.get('status', 'unknown')} · Promotion eligible: {change.get('promotion_eligible', False)}")
            change_usage = change.get("usage") or {}
            st.caption(f"LLM calls: {change_usage.get('calls', change.get('oci_calls', 1 if change.get('oci_request_id') else 0))}")
            if change_usage:
                render_usage(f"Usage for {change.get('FIELD_KEY', 'field')}", change_usage)
            intent = change.get("correction_intent") or change.get("intent") or {}
            if intent:
                with st.expander("Correction intent", expanded=False):
                    st.write({key: intent.get(key) for key in ("behavior", "label_policy", "transform_policy", "null_policy", "scope") if intent.get(key)})
            if change.get("rule_diff"):
                with st.expander("Rule changes", expanded=False):
                    st.write(change["rule_diff"])
            reason = change.get("reason")
            if reason:
                with st.expander("Business explanation", expanded=True):
                    st.write(reason)
        elif change.get("status") in {"generation_failed", "unavailable"}:
            reason = change.get("reason", "candidate generation failed")
            st.error(f"{change.get('FIELD_KEY', 'Field')}: {reason}")
        if generation:
            if generation.get("reason") == "provider_output_truncated_after_repair":
                st.warning("OCI reached its provider-managed output limit before returning a complete rule. The rule was not saved.")
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
    st.caption("Preview only. Production rules are unchanged until explicit approval.")


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
            json={"candidate_ids": selected_ids, "expected_rule_version": "v1", "confirm": True,
                  "promotion_scope": "supplier" if st.session_state.supplier_key.strip() else "global",
                  "supplier_key": st.session_state.supplier_key.strip() or None, "dry_run": False},
            timeout=httpx.Timeout(60, connect=connect_timeout_seconds),
        )
        if response.is_error:
            st.error(f"Approval failed: {response.text}")
        else:
            result = response.json()
            if result.get("status") == "promoted":
                scope = result.get("promotion_scope", "supplier")
                st.success(f"Approved candidates were persisted to the {scope} rule scope.")
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
            job_usage = job.get("sentence_generation_usage") or (job.get("usage") or {}).get("normal") or {}
            usage_calls = int(job_usage.get("calls") or 0) if isinstance(job_usage, dict) else 0
            reported_calls = int(job.get("oci_sentence_generation_call_count") or 0)
            if job.get("status") == "completed" and usage_calls > 0 and reported_calls == 0:
                st.warning("Runtime metadata mismatch: usage reports OCI calls but the job call counter is zero. Restart FastAPI and refresh this job.")
            st.caption(
                f"Sentence-generation OCI call: {'made' if usage_calls > 0 or reported_calls > 0 else 'not made'} "
                f"({max(usage_calls, reported_calls)} call(s))"
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
