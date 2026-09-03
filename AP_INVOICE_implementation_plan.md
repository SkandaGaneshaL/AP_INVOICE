# AP_INVOICE — Implementation Plan (No Code)

**Scope:** Improve extraction accuracy and correction-to-rule sentence quality using **google.gemini-2.5-flash** (PDF extract + rule merge) and **openai.gpt-oss-20b** (one sentence per field). Zero GEPA / MIPRO / iterative rollouts. Token-cheap. Generic across all suppliers, documents, and field keys. Logical / LLM-application logic only — no layout graphs, operator programs, regex induction, or DSA pipelines.

**Repo analyzed:** `SkandaGaneshaL/AP_INVOICE`  
**Evidence analyzed:** `2026-09-03T05-18_export.csv` (6 real correction runs)  
**Date:** 3 September 2026

---

## 0. What this system is actually doing (today)

Two models, two jobs, currently tangled:

| Job | Intended model | What happens now |
|---|---|---|
| Extract fields from PDF | `google.gemini-2.5-flash` | One giant prompt of *all* `DETAILED_RULE` bullets + PDF. Works, but rules are noisy and not supplier-scoped. |
| User edits a field | — | `invoice_payload.json` vs `final_response.json` diff. |
| Write **one new rule sentence** from that edit | `openai.gpt-oss-20b` @ **medium** reasoning | Model mostly **paraphrases the existing rule**, invents fallback hops, ignores the actual old→new transform. |
| Merge sentence into `DETAILED_RULE` | **Not Gemini.** Regex conflict merger in `rule_merger.py` | Appends/drops by overlapping English words. Does not rewrite the contradictory bullet (e.g. DD/MM vs MM/DD). |
| Promote | Human approval | Gates reject most candidates because PyMuPDF “evidence” is empty. |

`SupplierRuleStore` exists (`data/rules/suppliers/`) and is **not wired** into extract or update.

---

## 1. Why accuracy is low — root causes from the CSV

Six corrections, same invoice (Lippert). Two previewed, four rejected. **None of the six sentences encode the user’s actual intent.**

| FIELD_KEY | Old → New | What the user meant | What gpt-oss-20b wrote | Why it failed |
|---|---|---|---|---|
| PONumber | `LB9517259` → `9517259` | Strip leading alphabetic prefix from the **labeled PO** | Restate existing “extract from labeled PO; if empty return null” | Previewed, but **did not learn the transform**. False hop (“if empty”) leaked through. |
| InvoiceDate | `23/06/2026` → `06/23/2026` | Change output format **DD/MM/YYYY → MM/DD/YYYY** | Copied existing rule **including “format as DD/MM/YYYY”** | Rejected: empty evidence. Sentence **contradicts the correction**. |
| VendorNameEnglish | `LIPPERT COMPONENTS, INC` → `LIPPERT COMPONENTS` | Drop legal suffix (INC / INC.) | Generic “extract English name from header” | Rejected: empty evidence. No suffix policy. |
| PayeeName | `LIPPERT COMPONENTS, INC.` → `LIPPERT COMPONENTS` | Same suffix drop | **Invented** “Recipient label, else fall back to Beneficiary” | Previewed. Classic **false hop**. Hop regex misses `fall back`. |
| InvoiceNumber | `PSI-0009280560` → `0009280560` | Strip leading `PSI-` (or keep numeric core) | Invented “numeric string adjacent to the date; if absent search header for ‘number’” | Rejected. Hallucinated spatial heuristic + hop. |
| VendorName | `LIPPERT COMPONENTS, INC` → `LIPPERT COMPONENTS` | Suffix drop, keep spelling | “If missing, use CompanyName” | Rejected. False hop to another field. |

### 1.1 The eight bugs that produce this

1. **The model is asked to rewrite a rule, but is given the old rule as the main signal.**  
   `build_sentence_payload` sends `short_rule` + gist of the first two `DETAILED_RULE` bullets. gpt-oss-20b, especially at medium reasoning, treats that as the answer and paraphrases it. The correction is a side note.

2. **User correction does not outrank the existing rule.**  
   InvoiceDate is the smoking gun: existing `SHORT_RULE` = “format strictly DD/MM/YYYY”. User changed the value to MM/DD. The model obeyed the old rule.

3. **Payload is the wrong shape for a 20B reasoning model.**  
   Sent: field_key, labels, old/new values, a 4-value `correction_kind`, two label lists.  
   **Not sent:** observed transform, snippets, “what already exists vs what is missing”, “do not restate”, supplier, 2–3 canonical examples of *this kind of delta*.  
   Anthropic’s rule: *smallest set of high-signal tokens*. Today you send low-signal tokens (old rule gist, unrelated Order/Ship competing hits) and omit the high-signal ones (the delta).

4. **`correction_kind` is a regex toy that mislabels almost everything.**  
   `classify_correction` in `correction_kind.py`: if any positive label AND any competing label exist → `label_disambiguation`. Evidence builder **always** injects `Order No./Ship No.` as competing evidence on **every** field. So PONumber (a prefix-strip) is tagged as label disambiguation. Date reformat is sometimes `format_policy`, suffix-strip falls through to `preserve_literal`. The 20B model is steered by a wrong category.

5. **Evidence builder is classical DSA and it is wrong for this job.**  
   `evidence.py` is PyMuPDF word grouping + regex labels + `_exact_token`.  
   - It cannot see “Customer PO” vs “Invoice Date” unless the PDF text layer cooperates.  
   - `_transformation` only knows 4 ops (alpha prefix, thousands, date repattern, `Token Rest`). It cannot see `PSI-`, `, INC.`, `INC.`, trailing dots, legal-form suffixes, or date order swap when parse fails.  
   - It dumps Order/Ship references onto every field (context poisoning).  
   - Gate: `supported = packet is None or bool(packet.evidence)`. Empty evidence → score 0 → **rejected**, even if the sentence is usable. That is why 4/6 died.

6. **Reasoning effort is the token firehose, and it makes quality worse.**  
   gpt-oss-20b is trained so `Reasoning: medium|high` **lengthens the CoT**. OpenAI: use low effort for tasks that do not need complex reasoning. Writing one 25-word imperative sentence is that task.  
   CSV: 343–1396 **output** tokens for a 40-word sentence. `output_tokens_semantics = may_include_reasoning_tokens`. 4 of 6 fields used **2 calls** (repair). Repair strips the payload even further, so the second attempt hallucinates more hops.

7. **Validators ban the wrong things and miss the right things.**  
   - Ban: old/new literal in the sentence (good).  
   - Ban: `{`, `}`, multi-sentence, words `otherwise|fallback|then use|instead use|if not found|regex`.  
   - Miss: `fall back`, `if missing`, `if absent`, `if empty`, `or the nearest`, `search for`, `adjacent to the date`.  
   PayeeName’s “fall back to Beneficiary” **passed**. That is your “false hops” complaint.

8. **Merge is not an LLM, so contradictions survive.**  
   `rule_merger.py` keeps both “Return DD/MM/YYYY” and a new format sentence, or drops by word overlap. User asked for Gemini-2.5-flash to **rewrite** `DETAILED_RULE`. That is the correct tool for merge. It is not used.

### 1.2 Why “high reasoning” cannot save a 20B model here

gpt-oss-20b (Harmony format):
- CoT lives in the `analysis` channel; final answer in `final`.
- Medium/high CoT **invents extra policy** (hops, spatial heuristics) because the task looks under-specified.
- Structured Outputs + Harmony is fragile if the schema is only `{"sentence": "..."}` — the model has no slot to put “the transform”, so it either restates the old rule or rambling-hops.
- Instruction-following is good on the **final** channel, weak on analysis. Don’t pay for analysis you will throw away.

**Fix is not a bigger CoT. Fix is a better JSON contract + `Reasoning: low` + no repair call.**

---

## 2. Names you must change in this repository

You asked for names only. Grouped by action.

### 2.1 Delete (algorithmic / GEPA / DSA — stop using these)

These implement the “operator program / layout graph / MDL / transform induction” stack you want gone:

| Path | Why |
|---|---|
| `app/layout_graph.py` | Spatial graph over PDF words. DSA. |
| `app/intent_lattice.py` | Lattice search over intents. |
| `app/transform_induction.py` | Regex/edit-script induction of transforms. |
| `app/operators.py` | `FieldProgram`, `SelectOp`, `DisambiguateOp`, `TransformOp`, MDL candidates. |
| `app/rule_compiler.py` | Compiles programs → sentences. |
| `app/candidate_binder.py` | Binds candidates with scores. |
| `app/type_inference.py` | Date/number parsers used as a substitute for LLM intent. |
| `app/sentence_templates.py` | Hard-coded templates keyed by operator names. |
| `app/gepa_adapter.py` | GEPA. Token burner. |
| `app/gepa_jobs.py` | GEPA jobs. |
| `app/oci_reflection.py` | GEPA reflection LM. |
| `app/context_packer.py` | Packer for the old stack. |
| `tests/test_algorithm_first.py` | Tests the DSA path. |
| `tests/test_gepa_async.py` | GEPA. |
| `tests/test_gepa_detached_runtime.py` | GEPA. |

Unwire every import of `FieldProgram`, `GepaRuleOptimizer`, `ExtractionRuleGEPAAdapter`, `OciReflectionLM`, `render_sentence`, `layout_graph`.

### 2.2 Rewrite in place (keep the filename, change the contract)

| Path | Change the name/contract of |
|---|---|
| `prompts/sentence_developer.txt` | Today a 1-line stub. Replace with Harmony **developer** message: role, `Reasoning: low`, no-hop constitution, delta-only job. |
| `prompts/merge_developer.txt` | Point this at **Gemini Flash**, not 20B. Merge/rewrite `DETAILED_RULE`, max 6 bullets, correction wins over old rule. |
| `prompts/extract_pass_a.txt` / `extract_pass_b.txt` | Currently 1 line each. Make them real: Pass A = identity/header fields; Pass B = line items. Same Gemini Flash, two cheap calls only if line items exist. |
| `app/sentence_payload.py` → keep name, change function `build_sentence_payload` | Emit the **delta JSON** in §4, not the current gist. |
| `app/sentence_validators.py` → `validate_sentence` | Expand hop lexicon; keep literal-value ban; add “does not mention other FIELD_KEYs”. |
| `app/sentence_gates.py` → `evaluate_sentence_gates` | Stop requiring PyMuPDF evidence. Gate on: schema, hops, literals, “encodes the delta”, “does not restate short_rule”. |
| `app/correction_kind.py` | Either delete, or reduce to a **label** produced by the 20B structured output. Do not regex-classify. |
| `app/prompt_builder.py` → `RulePromptBuilder.normal_payload` | Delta-only. Remove `feedback()` dumping of competing Order/Ship lines. |
| `app/oci_provider.py` / `app/oci_native_rule_provider.py` | Force `reasoning_effort=LOW` (or `none` if OCI supports it). Put `Reasoning: low` in the system/developer text. Cap `max_tokens` for the **final** channel. Schema = CorrectionIntent, not `{sentence}`. Disable the repair loop or replace it with a **local assembler** (§5). |
| `app/rule_merger.py` | `merge_rule_sentences` must call Gemini Flash. Keep the current function as a **offline fallback** only if Flash is down. |
| `app/evidence.py` | Replace PyMuPDF regex hunt with a **Gemini Flash evidence call** (or skip evidence entirely for sentence gen — see §5). If you keep a locator, it must be multimodal and field-scoped. |
| `app/service.py` → `UpdateRulesService` | Pipeline: diff → per-field 20B sentence → Flash merge → preview. Remove GEPA branch, `FieldProgram` (currently referenced **without import** — latent bug). |
| `app/extraction_prompt.py` → `InvoiceExtractionPromptBuilder` | Compact per-field rules + supplier overlay. Do not concatenate 8 PONumber bullets into the extract prompt every time. |
| `app/pdf_extractor.py` | Native PDF to Flash; add `source_label` + `confidence` to the extract schema; delete `_apply_explicit_normalizations` (hard-coded InvoiceNumber prefix strip). |
| `app/supplier_store.py` | Wire `resolve()` into extract **and** into sentence/merge. Persist promoted field rules under `data/rules/suppliers/<supplier_key>.json`. |
| `app/model_registry.py` | Default `reasoning_effort` = `low`. Never `medium`/`high` for this task. |
| `app/models.py` → `UpdateRequest` | Default `reasoning_effort="low"`, `enable_gepa=False`. Add `supplier_key`. Add `CorrectionIntent`. |
| `data/extraction_rules.json` | Keep as **global baseline only**. Do not keep growing it with supplier-specific quirks. |
| `streamlit_app.py` | Hide GEPA. Default reasoning Low. Show delta-intent + “what will be merged” before approval. |

### 2.3 Keep, lightly touch

| Path | Role |
|---|---|
| `app/comparator.py` | Diff `invoice_payload` vs `final_response`. Keep. Do not put business transforms here. |
| `app/rule_repository.py` | Global rules. |
| `app/audit.py` / `app/feedback_repository.py` | Approval + gold traces (Hamel flywheel). |
| `app/update_jobs.py` / `app/main.py` | Job/SSE shell. Strip GEPA status. |
| `app/oci_pdf_client.py` | Flash transport for PDF. |
| `app/model_output.py` | Parser. Extend for CorrectionIntent JSON. |
| `app/schema_contract.py` / `app/normalization.py` | Keep only as **post-LLM** light canonicalization (whitespace), not as the source of truth. |
| `app/merge.py` `append_rule` | Promotion still appends after human OK — but after Flash merge, not raw append. |

### 2.4 Config / env names to change

| Current | Set to |
|---|---|
| `OCI_RULE_GENERATION_REASONING_EFFORT` (UI often `medium`) | `low` |
| `OCI_MODEL_OUTPUT_REPAIR_ATTEMPTS` (1 → 2 calls) | `0` |
| `OCI_EXTRACTION_MODEL_ID` | `google.gemini-2.5-flash` |
| `OCI_RULE_GENERATION_MODEL_ID` / `OCI_GPT_RULE_MODEL_ID` | `openai.gpt-oss-20b` |
| New: `OCI_RULE_MERGE_MODEL_ID` | `google.gemini-2.5-flash` |
| `enable_gepa` | always false; delete the flag later |
| New: `STORE_FEEDBACK_EVIDENCE` | `true` (gold traces, not sent to 20B every time) |

---

## 3. Target architecture (LLM application logic, not agents)

Chip Huyen / Anthropic “Building effective agents”: **this is a workflow, not an agent.** No tool loop. No ReAct. Two (optionally three) single-shot LLM calls per corrected field.

```
PDF ──► Gemini 2.5 Flash (extract) ──► invoice_payload.json
                                              │
User edits in UI ──► final_response.json      │
                                              ▼
                         comparator.find_changes  (no LLM)
                                              │
                    for each changed field_key │
                                              ▼
              compact CorrectionDelta JSON ──► gpt-oss-20b
              Harmony, Reasoning: low          │
              Structured Output                ▼
                                   CorrectionIntent
                                   {behavior, transform, sentence, ...}
                                              │
                         local gates (no LLM) │
                         if sentence fails: assemble from slots
                                              ▼
              {existing DETAILED_RULE, new sentence, intent}
                                              │
                                              ▼
                         Gemini 2.5 Flash (merge/rewrite)
                                              │
                                              ▼
                    preview DETAILED_RULE (max 6 bullets)
                                              │
                         human approve ──► persist
                              │
                    global? no ──► data/rules/suppliers/<key>.json
                    global yes ──► data/extraction_rules.json
```

Extraction and merge share Flash. Sentence generation is **only** 20B. That matches your constraint and uses each model for what it is good at:

- Flash: vision + long context + “rewrite this list of bullets”.
- 20B: cheap, instruction-following, structured JSON, **one thought**.

Do **not** put the PDF in the 20B context. Do **not** put the full extract JSON in the 20B context.

---

## 4. The JSON you must send to gpt-oss-20b

This is the whole game. Today’s payload teaches the model the **old** rule. The new payload teaches the **delta**.

### 4.1 CorrectionDelta (user → 20B)

Keep under ~200 tokens. One object, one field.

```json
{
  "task": "write_one_extraction_rule_sentence",
  "field_key": "InvoiceDate",
  "display_label": "Invoice Date",
  "supplier_scope": "this_supplier_only",
  "existing_behavior_one_line": "Take Invoice/Issue/Bill Date; emit DD/MM/YYYY; ignore due/delivery dates.",
  "delta": {
    "old_value": "23/06/2026",
    "new_value": "06/23/2026",
    "observed_change": "same calendar date, different print order",
    "do_not_restate": "Do not repeat the existing DD/MM/YYYY policy. The user just overrode it."
  },
  "constraints": [
    "One imperative sentence, <= 30 words.",
    "Reusable for any invoice of this supplier. Never copy old_value or new_value.",
    "No other fields. No fallbacks, hops, 'if missing', 'otherwise', 'search for'.",
    "If the existing_behavior already covers the delta, return sentence=null and noop=true."
  ]
}
```

What **not** to send:
- Full `DETAILED_RULE` list
- Full invoice JSON
- Competing Order/Ship hits
- Historical GEPA traces
- Bounding boxes
- Other field keys
- The entire prompt constitution (that belongs in the **developer** message, cached)

Optional high-signal add (only if cheap and field-local):

```json
"anchor": {
  "source_label_on_document": "Invoice Date",
  "raw_as_printed": "23/06/2026"
}
```

Get `anchor` from Flash at extract time (see §7), not from PyMuPDF.

### 4.2 CorrectionIntent (20B → you)

Do **not** ask only for `{"sentence":"..."}`. Ask for slots + sentence. One call. Schema-constrained.

```json
{
  "noop": false,
  "behavior": "labeled_value",
  "label_policy": "Use the field labeled Invoice Date, Issue Date, or Bill Date. Ignore due, delivery, payment dates.",
  "transform_policy": "Emit MM/DD/YYYY regardless of how the date is printed.",
  "null_policy": "If the label is absent: Not present. If the label exists but unreadable: null.",
  "scope": "this_supplier_only",
  "sentence": "Extract the invoice issue date from Invoice Date / Issue Date / Bill Date and emit it as MM/DD/YYYY."
}
```

Closed enums keep a 20B model from inventing hops:

- `behavior`: `labeled_value | section_value | header_entity | line_item | compute_forbidden`
- `transform_policy` may be `none` or a short clause (`strip_leading_alpha_prefix`, `strip_legal_suffix`, `reformat_date`, `keep_literal`, `numeric_core_only`, …) — **generated as text**, not picked from a hardcoded operator table.
- `noop: true` when the edit is a one-off typo, not a rule.

Then **compile** `sentence` locally if the model’s sentence fails gates:

```
sentence = join_nonempty(label_policy, transform_policy)
```

That kills the repair LLM call (the 2nd 500–1400 token hit in the CSV).

### 4.3 MergePayload (you → Flash)

```json
{
  "field_key": "InvoiceDate",
  "display_label": "Invoice Date",
  "existing_detailed_rule": ["...current bullets..."],
  "new_sentence": "Extract the invoice issue date ... emit MM/DD/YYYY.",
  "intent": { "...CorrectionIntent..." },
  "merge_policy": [
    "The new_sentence and transform_policy WIN over any contradicting existing bullet.",
    "Rewrite, do not append duplicates.",
    "Keep generic: no invoice-specific values.",
    "At most 6 one-sentence bullets.",
    "No fallback hops to other fields.",
    "Preserve null/not-present policy unless the correction changed it."
  ]
}
```

Flash returns:

```json
{
  "updated_detailed_rule": ["...", "..."],
  "dropped_bullets": ["Return DD/MM/YYYY."],
  "conflict_resolved": true,
  "short_rule": "format strictly MM/DD/YYYY"
}
```

`SHORT_RULE` must be updated too — today it is never updated, so the next extract still says DD/MM.

---

## 5. How to make gpt-oss-20b accurate **and** cheap

### 5.1 Harmony + reasoning (this is the token lever)

gpt-oss is post-trained on Harmony: roles `system > developer > user > assistant`, channels `analysis` / `final`.

Put in the **developer** (or system) message, literally:

```
Reasoning: low
```

Also send OCI `reasoning_effort=LOW`. Do both. The model card says effort is set by the system sentence; your code currently only sets the OCI field.

| Effort | When | Expected extra CoT |
|---|---|---|
| **low** (default for this app) | One-sentence rule from a delta JSON | tens–low hundreds of tokens |
| medium | Never for this task | CSV: 343–1396 output tokens |
| high | Never | GEPA-like cost without GEPA gains |

OpenAI: *“adjust reasoning effort for tasks that don’t require complex reasoning and/or target very low latency final outputs.”* This is that task.

Also set **verbosity low** in the developer text (`Verbosity: low` if the serving stack honors Harmony extras). Your code already tried OCI `verbosity` and had to drop it — put it in text instead.

### 5.2 Constitution (cached, stable prefix)

One developer message, identical every call so OCI prefix-cache can hit:

1. You write **one** reusable extraction-rule sentence for **one** field.
2. The user JSON is a **delta**. Your job is the **missing behavior**, not a paraphrase of `existing_behavior_one_line`.
3. Never copy `old_value` / `new_value`.
4. Never mention another field_key.
5. Never write a hop: no otherwise / fallback / fall back / if missing / if absent / if empty / if not found / search for / then use / nearest heading.
6. If the delta is already covered, `noop=true`, `sentence=null`.
7. Return **only** the CorrectionIntent JSON on the **final** channel.

Anthropic: start minimal, add a rule only when a failure mode is real. The hop ban is a real failure mode (CSV rows 4, 5, 6).

### 5.3 Two or three frozen demonstrations (not GEPA)

Not optimization. Not rollouts. Three **static** few-shots in the developer message (still cacheable):

1. **Format override:** `23/06/2026` → `06/23/2026` ⇒ sentence about emitting MM/DD/YYYY.  
2. **Prefix strip:** `LB9517259` → `9517259` ⇒ “from the labeled PO, drop a leading alphabetic prefix when the remainder is the identifier core.”  
3. **Legal suffix:** `ACME, INC.` → `ACME` ⇒ “use the legal entity name without corporate suffix (Inc, LLC, Ltd, GmbH, …).”

These are **types**, not Lippert values. Swap the examples for invented ones so the model cannot memorize `9517259`.

This is BootstrapFewShot’s *idea* (show successful traces) without the optimizer. Token cost: ~250 tokens, cached.

### 5.4 Constrained decoding

Keep OCI `JsonSchemaResponseFormat` / `is_strict=True`, but **widen the schema** to CorrectionIntent.  
Caveat from the wild: Harmony + GBNF/grammar sometimes garbles gpt-oss (LM Studio #1555). If OCI strict schema starts swallowing the `final` channel, fall back to: Harmony `final` + “JSON only” + local `json.loads`, still **one** call.

Do not ask the model for a `reason` field that restates values — your `_validated_reason` already scrubs them and then substitutes a **fake** “Selected evidence associated with the explicit X label…” which is why every CSV row has the same lying `decision_summary`.

Replace that template with: `intent.behavior + intent.transform_policy` (no values).

### 5.5 Local assembler = 0-token repair

CSV shows 2 calls whenever validation failed. Stop calling the model again.

```
parse CorrectionIntent
if sentence fails gates:
    sentence = assemble(label_policy, transform_policy)  # local
if still fails:
    mark generation_failed — do not invent
```

### 5.6 Token budget (target vs CSV)

| | Today (CSV) | Target |
|---|---|---|
| Input | 307–571 | ~250 system (cached) + ~180 user |
| Output (incl. CoT) | 343–1396 | ~80–200 |
| Calls / field | 1–2 | **1** |
| Total / field | 650–1967 | **~400–600** |

Prefix-cache the developer constitution. Shopify gisting / Anthropic compaction: you do **not** need learned gist-tokens. You need to **not send** `DETAILED_RULE` and competing evidence.

### 5.7 What inputs 20B actually needs (and nothing else)

Must:
- field_key, display_label
- one-line existing behavior (gisted by **you**, not the first two bullets blindly)
- old_value, new_value
- a one-line `observed_change` (string diff description you compute **without** an LLM — “prefix removed”, “suffix removed”, “date order swapped”, “whitespace”, “wholly different string”)
- constraints list (short)

Nice if cheap:
- `source_label_on_document` from extract-time Flash citation

Never:
- PDF bytes, page images
- sibling fields
- GEPA history
- Order/Ship competing list
- `reasoning_effort=medium|high`

`observed_change` without an LLM: compare strings.

- `new in old` and old−new is `[A-Za-z].*` → “leading alphabetic/token prefix removed”
- `new in old` and old−new is `,?\s*(INC|LTD|LLC|GMBH|PTE|PLC|CO)\.?` → “legal suffix removed”
- both look like dates, same digits different order → “date format reordered”
- edit distance small, punctuation only → “punctuation/canonical form”
- else → “value replaced; infer the general policy from the pair, do not memorize”

That is a **string observation**, not an operator engine. Do not rebuild `transform_induction.py`.

---

## 6. False hops — kill them at three layers

You cannot prompt-engineer this away with a 20B model if you also **show it** existing rules full of “If absent, return null / If no X, use Y”.

| Layer | What |
|---|---|
| Prompt | Explicit ban list in developer constitution. Few-shot of a **rejected** hop: “BAD: if missing, use CompanyName”. |
| Schema | No field named `fallback`. `null_policy` is a closed string, not free prose. |
| `validate_sentence` | Expand `_HOPS` to: `otherwise`, `fallback`, `fall back`, `then use`, `instead`, `if not found`, `if missing`, `if absent`, `if empty`, `search for`, `nearest heading`, `or the`, `regex`, `adjacent to the date`, `CompanyName`, and **any other FIELD_KEY** in `extraction_rules.json`. |

Null policy is allowed **only** as a dedicated `null_policy` slot, not inside `sentence`. If you need it in the merged rule, Flash adds it once, not the 20B sentence.

---

## 7. Extraction accuracy (Gemini 2.5 Flash) — generic, supplier-aware

Sentence generation cannot fix a bad extract. Flash is already the right model (native PDF, ~82–93% on clean invoices in 2025 native-PDF benches; citations + schema). The misses are **prompt and memory**, not the model.

### 7.1 Techniques to apply (LLM-app, not DSA)

1. **Native PDF in, structured JSON out** (Gemini document understanding). Do not pre-OCR with PyMuPDF for the extract path.
2. **Per-field rule as the schema description**, not an 8-bullet essay. Flash should see:
   - global 3-line constitution (extract as written, no math, no guess)
   - for each field: `DISPLAY_LABEL`, compact `SHORT_RULE`, and **at most 2** detailed bullets
   - supplier overlay bullets **for this supplier only**
3. **Grounding fields in the schema** (Box / Vertex pattern):

   ```json
   "InvoiceNumber": {
     "value": "PSI-0009280560",
     "Page": 1,
     "source_label": "Invoice No.",
     "confidence": 0.86
   }
   ```

   Later, sentence-gen uses `source_label`. No second PDF hunt.

4. **Extract as-written. Apply policy in the same call only if the field rule says so.**  
   Do not have a Python `if field == InvoiceNumber: strip prefix`. That is the hardcoded path in `_apply_explicit_normalizations`. Policy lives in the rule text Flash already reads.

5. **Two-pass only when needed** (your `extract_pass_a` / `extract_pass_b`):
   - Pass A: header/identity (vendor, dates, amounts, PO, invoice #)
   - Pass B: line items, only if Pass A or a cheap visual cue says a table exists  
   Same model, two focused prompts. Shopify/Anthropic: narrow tasks beat one giant prompt.

6. **Supplier overlay, not a global snowball.**  
   Lippert wants `LIPPERT COMPONENTS` without `INC`. A German GmbH invoice must **keep** GmbH. If you write “always strip INC” into `data/extraction_rules.json`, you poison every other supplier.  
   Persist under `data/rules/suppliers/lippert_components.json`. `SupplierRuleStore.resolve()` already sketches this — **use it**.

7. **Confidence routing (IDP industry default, Hyperscience / Azure DI).**  
   Auto-accept field if `confidence >= τ` (per-field τ: amounts higher than addresses). Else highlight for the human. The correction you already capture becomes the gold label.

8. **Do not compute tax/net/gross in the model unless the rule forbids it — and your current rules already forbid it.** Keep that. Flash will still try to “be helpful”; the constitution must say **compute_forbidden** in the field schema description.

9. **Language:** keep your `<original> || <English>` convention; it is a good bilingual contract. Do not translate identifiers.

### 7.2 What not to add for extraction

- LayoutLMv3 / Donut / GNN spatial graphs  
- HyDE, Self-RAG, CRAG, RAPTOR, GraphRAG (no retrieval corpus; the document **is** the context)  
- Multi-agent extractors  
- Regex template per supplier  
- GEPA over the extract prompt  

Flash + compact rules + supplier overlay + citations **is** the 2026 IDP pattern (Tradeshift: “custom prompts per high-value supplier, no template”; Azure DI: human review on low confidence).

---

## 8. Supplier-specific implementation (the missing product piece)

`supplier_store.py` is the right idea. It is unused.

**Identity:** `supplier_key(VendorName)` after first extract (or user-selected supplier). Unstable names: also key by tax ID / GST if present.

**Store shape:**

```
data/rules/suppliers/<supplier_key>.json
{
  "InvoiceDate": {
    "SHORT_RULE": "format strictly MM/DD/YYYY",
    "DETAILED_RULE": ["...merged bullets..."],
    "updated_at": "...",
    "source_candidate_id": "..."
  }
}
```

**Read path (extract):** `global_rules ⊕ supplier_overlay`. Flash sees the merged compact rules.

**Write path (promote):** default **supplier only**. Promoting to global should be an explicit second button (“apply to all suppliers”) because a Lippert suffix rule is wrong for a sole proprietor.

This is Letta/MemGPT “memory outside the window”, implemented as JSON files — no vector DB.

---

## 9. Latest techniques — what to use vs ignore

Mapped to **this** workflow. “Latest” does not mean “spend tokens.”

### 9.1 Use (LLM application logic)

| Technique | Source | How you use it |
|---|---|---|
| **Context engineering: smallest high-signal tokens** | Anthropic Eng, Sep 2025 | Delta JSON, not full rules + competing hits. |
| **Right-altitude prompts** | Anthropic “Building effective agents” | Constitution in developer message; no 40-bullet if-else. |
| **Write / Select / Compress / Isolate** | LangChain + Anthropic | Write: supplier JSON memory. Select: only this field. Compress: 1-line existing_behavior. Isolate: 20B never sees PDF. |
| **Prefix / prompt caching** | Anthropic, OpenAI, Shopify | Frozen developer constitution + 3 few-shots. |
| **Gisting (operational, not NeurIPS weights)** | Shopify Eng 2026 | Pre-compress `DETAILED_RULE` → one line *before* 20B. You already have `_gist`; gist the **policy**, not bullet[0:2]. |
| **Structured Outputs / constrained decoding** | OpenAI, Outlines, Instructor | CorrectionIntent schema on 20B; merge schema on Flash. |
| **Harmony Reasoning: low** | gpt-oss model card | System sentence + OCI field. |
| **Local slot-filling assembler** | Compiler pattern (DSPy’s spirit, no optimizer) | Repair without a 2nd LLM call. |
| **Human-in-the-loop IDP + per-field confidence** | Azure DI, Hyperscience, Tradeshift | You already have preview/promote. Gate on quality, not PyMuPDF. |
| **Supplier-specific prompts, not templates** | Tradeshift 2026 | Overlay JSON. |
| **Error taxonomy → unit evals** | Hamel Husain evals, Eugene Yan | See §10. |
| **Trace logging** | Honeycomb “hard stuff”, Datadog, Langfuse pattern | Store payload, intent, sentence, merge diff, tokens. |
| **Constitutional constraints** | Constitutional AI idea, applied as rules | Hop ban, no-hardcode, no-other-field. |
| **Two-pass extract (header / lines)** | Anthropic “narrow tools”, Shopify River “narrow swarms” | `extract_pass_a` / `b`. |
| **Citations in extract schema** | Gemini bounding-box + Box/Vertex IDP | `source_label`, `Page`, optional snippet. |
| **Approval workflow as the learning loop** | Your promote API | Gold set grows; 20B still one-shot. |

### 9.2 Do not use (for this constraint set)

| Technique | Why not |
|---|---|
| **GEPA / MIPROv2 / COPRO / GRPO / DPO loops** | Multi-rollout token burn. You already removed GEPA for this reason. |
| **ReAct / tool-using agent for sentence gen** | One JSON in, one JSON out. Tools add hops. |
| **Self-Consistency, ToT, Least-to-Most, Reflexion** | Iteration = tokens. |
| **CRAG / Self-RAG / RAPTOR / GraphRAG / HyDE / ColBERT** | No external corpus. The invoice is the document. |
| **Layout graphs, HNSW, FAISS, operator MDL, edit scripts** | The current `app/layout_graph.py`… stack. You asked it gone. |
| **Fine-tune 20B on your invoices** | Data too small; supplier drift; ops cost. Revisit only after 1k+ gold deltas. |
| **High reasoning / visible CoT to users** | OpenAI: CoT of gpt-oss can disobey and is unsafe to show. You already hide it; stop paying for it. |

### 9.3 Papers/blogs that actually map (from your lists)

Worth implementing the **pattern**, not the repo:

- Anthropic: Building effective agents; Effective context engineering; Writing tools for agents  
- OpenAI: gpt-oss model card (Harmony, effort); Structured Outputs  
- Hamel: evals + field guide (failure taxonomy)  
- Jason Liu: systematically improving RAG — use the **eval flywheel**, not the retriever  
- Eugene Yan: LLM patterns (guardrails, cache, feedback)  
- Honeycomb: instrument the LLM product  
- Shopify: gisting, River (narrow task agents)  
- Tradeshift / Azure DI: HITL invoice IDP  
- Gemini docs: native PDF + response schema  
- Instructor / Outlines: schema-first extraction (pattern; you already have OCI json_schema)

Ignore as **implementation** for sentence-gen: LangGraph multi-agent, CrewAI, AutoGen, MemGPT loops, DSPy teleprompters.

---

## 10. Evals (this is how you know it got accurate)

Hamel: you cannot prompt your way out of a missing taxonomy. Build **unit tests on traces**, not another LLM judge first.

### 10.1 Failure taxonomy (from your CSV + likely next errors)

1. `restate_existing` — sentence ≈ short_rule  
2. `false_hop` — conditional other-source  
3. `hardcoded_value` — contains old/new  
4. `wrong_transform` — date still DD/MM after MM/DD correction  
5. `cross_field` — mentions another FIELD_KEY  
6. `noop_missed` — wrote a rule for a typo  
7. `merge_contradiction` — old and new bullets both live  
8. `global_poison` — supplier quirk written globally  

### 10.2 Deterministic gates (pytest, no LLM)

For every gold row `(field_key, old, new, existing_rule, expected_properties)`:

- sentence word count ≤ 30  
- no hop lexicon  
- no old/new literals  
- no other field_keys  
- if expected_transform = date_order: sentence mentions the **new** pattern, not the old  
- merge result does not contain the contradicted bullet  
- one 20B call, input_tokens < 500, output_tokens < 250, reasoning_effort=low  

### 10.3 Field-level extract eval

Hold out N invoices × gold JSON. Metrics: exact match per field, format-tolerant match for money/dates, straight-through rate (all fields ≥ τ). Stratify by supplier and by “seen overlay vs new supplier”.

Do **not** use LLM-as-judge until the judge itself is measured on 50 human-labeled rows (Hamel). For this task, string/canonical match is enough.

### 10.4 Canary set

Lock the Lippert CSV six as canaries. They must all produce:

- PONumber: prefix-strip policy, no “if empty” in the *sentence*  
- InvoiceDate: MM/DD/YYYY, and Flash **drops** DD/MM  
- Vendor*/PayeeName: legal-suffix policy, **no** Recipient/CompanyName hop  
- InvoiceNumber: leading `PSI-`-style token strip, **no** “adjacent to the date”

---

## 11. End-to-end flow (what to implement, in order)

### Phase A — Stop the bleeding (1–2 days, biggest accuracy/token win)

1. Default reasoning **low**; UI default low; disable medium/high for this endpoint or warn.  
2. `OCI_MODEL_OUTPUT_REPAIR_ATTEMPTS=0`.  
3. Expand hop validator; reject `fall back` / `if missing` / other field names.  
4. Change `evaluate_sentence_gates` so empty PyMuPDF evidence does **not** auto-reject.  
5. Delete the fake `decision_summary` template.  
6. Change sentence payload to **delta-only** (§4.1).  
7. Developer message: `Reasoning: low` + constitution + 3 few-shots.  
8. Schema → CorrectionIntent; local assembler on sentence fail.

This alone should: cut tokens ~3–5×, stop false hops, stop restating DD/MM.

### Phase B — Flash merge + SHORT_RULE update (2–3 days)

1. New merge call: Gemini Flash, `prompts/merge_developer.txt`.  
2. Correction wins; max 6 bullets; return new `SHORT_RULE`.  
3. Preview in UI: before/after bullets, dropped bullets.  
4. Promotion writes **supplier overlay** by default.

### Phase C — Wire supplier memory + compact extract (3–5 days)

1. `SupplierRuleStore.resolve` on every extract.  
2. Compact extract prompt: constitution + short rules + overlay, not the full 333-line JSON.  
3. Add `source_label` + `confidence` to extract schema.  
4. Remove `_apply_explicit_normalizations`.  
5. Optional Pass B for line items.

### Phase D — Evidence without DSA (optional)

If you still want “why this correction” in the UI: one Flash call, PDF + `{field, old, new}` → `{source_label, snippet, page}`. Not used as a reject gate. Not PyMuPDF.

### Phase E — Eval flywheel (ongoing)

Gold JSON from promotions. Canary CSV. Token dashboards. No optimizer.

---

## 12. Exact responsibilities of each model (do not blur)

| Model | Allowed to do | Forbidden |
|---|---|---|
| **gemini-2.5-flash** | Read PDF, extract JSON, cite label/page, merge/rewrite `DETAILED_RULE`+`SHORT_RULE`, optional evidence locator | Generate the one-sentence delta (that is 20B’s job); run CoT-heavy reflection loops |
| **gpt-oss-20b** | Consume CorrectionDelta, emit CorrectionIntent + sentence | See PDF; see other fields; merge the rule list; medium/high reasoning; repair loops |

---

## 13. Application improvements (non-model)

- **Preview is lying today:** status `preview` does not mean the sentence matches the correction (PONumber, PayeeName). UI should show: `intent.transform_policy`, hop flags, “existing bullet this will replace”.  
- **Batch promote** already exists; add “promote to supplier” vs “promote globally”.  
- **Do not send `invoice_payload` full blob into 20B** — only the one field’s old/new.  
- **Idempotency:** same `(supplier, field, old, new, rule_hash)` → return cached candidate (Eugene Yan caching). 0 tokens.  
- **Instrument:** per field, per call: model, effort, input, output, cache_hit, gate_fail_reason (Honeycomb).  
- **Fix `service.py` `FieldProgram` NameError** when `rule.PROGRAM` is set — goes away when operators die.  
- **Global `DETAILED_RULE` for PONumber is already 8 bullets of hops.** That is why 20B emits hops: it is imitating *your* rule style. Flash merge should compress PONumber to 3 bullets: labeled PO only; empty → null; do not take Order/Ship. Transform (prefix strip) lives in the **supplier** overlay if only Lippert wants it.

---

## 14. What “logical not hardcoded” means in practice

Hardcoded (delete):
- `if field_key == InvoiceNumber: strip [A-Za-z]+`
- `CorrectionKind` regex tree
- `TEMPLATES[...]` in `sentence_templates.py`
- “always competing Order No” in `evidence.py`
- GEPA seed special-case for InvoiceNumber in `prompt_builder.gepa_seed`

Logical (keep):
- String observations: “new is a suffix of old”, “both parse as dates”  
- Schema enums that **name** behaviors without binding them to a field  
- Prompts that say “describe the reusable policy that explains this pair”  
- Supplier overlay as data, not `if supplier == Lippert`

A 20B model **can** induce “strip legal suffix” from one `(ACME, INC. → ACME)` example **if** you tell it that is the job. It **cannot** if you also paste an 8-line existing rule and turn reasoning to medium.

---

## 15. Decision summary

The CSV is not a model-quality problem. It is a **context and contract** problem:

1. You pay for medium CoT you then discard.  
2. You send the old rule and get the old rule back.  
3. You reject good-enough sentences because regex evidence is empty — and you **accept** bad sentences that hop.  
4. You never let Flash rewrite the contradicting bullet.  
5. You never scope learning to the supplier.

Do not replace 20B. Do not add GEPA. Do not add agents.

**Change the names in §2, ship the delta JSON in §4, lock Reasoning: low, merge with Flash, overlay per supplier, eval the taxonomy in §10.** That is the whole plan.
