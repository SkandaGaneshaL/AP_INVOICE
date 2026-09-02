from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

from .operators import CandidateValue, CompetingHit, EvidenceBundle, EvidenceHit, FieldProgram


@dataclass
class LayoutNode:
    text: str
    bbox: list[float]
    page: int
    kind: str = "WORD"
    section: str = "header"
    node_id: int = -1


@dataclass
class LayoutGraph:
    nodes: list[LayoutNode] = field(default_factory=list)
    edges: list[tuple[int, int, str]] = field(default_factory=list)
    line_nodes: list[list[int]] = field(default_factory=list)
    block_nodes: list[list[int]] = field(default_factory=list)
    section_nodes: dict[str, list[int]] = field(default_factory=dict)
    table_nodes: list[list[int]] = field(default_factory=list)
    cell_nodes: list[list[int]] = field(default_factory=list)

    def lines(self) -> list[list[LayoutNode]]:
        return [[self.nodes[index] for index in line] for line in self.line_nodes] or self._fallback_lines()

    def _fallback_lines(self) -> list[list[LayoutNode]]:
        result: list[list[LayoutNode]] = []
        for node in sorted(self.nodes, key=lambda item: (item.page, item.bbox[1], item.bbox[0])):
            target = next((line for line in result if line and line[0].page == node.page and abs(line[0].bbox[1] - node.bbox[1]) <= 5), None)
            if target is None:
                result.append([node])
            else:
                target.append(node)
        for line in result:
            line.sort(key=lambda item: item.bbox[0])
        return result


def _words(source: Any) -> list[tuple[int, str, list[float]]]:
    result = []
    if isinstance(source, list):
        for item in source:
            if isinstance(item, dict):
                bbox = item.get("bbox", [item.get(str(index), 0) for index in range(4)])
                result.append((int(item.get("page", 1)), str(item.get("text", "")), list(bbox)))
            elif len(item) >= 5:
                result.append((int(item[5]) if len(item) > 5 and isinstance(item[5], (int, str)) and str(item[5]).isdigit() else 1, str(item[4]), list(item[:4])))
        return result
    try:
        for page_number, page in enumerate(source, 1):
            for word in page.get_text("words") or []:
                result.append((page_number, str(word[4]), list(word[:4])))
    except Exception:
        return []
    return result


def _section(text: str, y: float, page_height: float = 1000) -> str:
    lower = text.casefold()
    if any(token in lower for token in ("bill to", "buyer")): return "bill_to"
    if any(token in lower for token in ("ship to", "delivery")): return "ship_to"
    if any(token in lower for token in ("remit", "payment")): return "remit_to"
    if any(token in lower for token in ("vendor", "supplier", "from")): return "vendor"
    if y < page_height * .25: return "header"
    return "body"


def build_layout_graph(pdf_or_page_words: Any) -> LayoutGraph:
    graph = LayoutGraph()
    for page, text, bbox in _words(pdf_or_page_words):
        node = LayoutNode(text=text, bbox=bbox, page=page, section=_section(text, bbox[1]), node_id=len(graph.nodes))
        graph.nodes.append(node)
    lines: list[list[int]] = []
    for index in sorted(range(len(graph.nodes)), key=lambda i: (graph.nodes[i].page, graph.nodes[i].bbox[1], graph.nodes[i].bbox[0])):
        node = graph.nodes[index]
        target = next((line for line in lines if graph.nodes[line[0]].page == node.page and abs(graph.nodes[line[0]].bbox[1] - node.bbox[1]) <= 5), None)
        if target is None:
            lines.append([index])
        else:
            target.append(index)
    for line in lines:
        line.sort(key=lambda i: graph.nodes[i].bbox[0])
    graph.line_nodes = lines
    # Blocks are bounded groups of nearby lines.  This lightweight geometry
    # representation is sufficient for deterministic section/table priors and
    # avoids making a learned layout model part of the correction path.
    blocks: list[list[int]] = []
    for line in lines:
        if blocks and graph.nodes[blocks[-1][-1]].page == graph.nodes[line[0]].page and \
                graph.nodes[line[0]].bbox[1] - graph.nodes[blocks[-1][-1]].bbox[3] <= 18:
            blocks[-1].extend(line)
        else:
            blocks.append(line[:])
    graph.block_nodes = blocks
    graph.section_nodes = {}
    for block in blocks:
        section = graph.nodes[block[0]].section
        graph.section_nodes.setdefault(section, []).extend(block)
    # Treat dense lower-page rows as a table region; cell construction remains
    # conservative because OCR/PDF coordinates are often imperfect.
    graph.table_nodes = [line[:] for line in lines if len(line) >= 3 and graph.nodes[line[0]].bbox[1] > 250]
    graph.cell_nodes = [line[:] for line in graph.table_nodes]
    for line in lines:
        for left, right in zip(line, line[1:]):
            graph.edges.append((left, right, "reading_order_next"))
    for section, indexes in graph.section_nodes.items():
        for index in indexes:
            graph.edges.append((index, index, "belongs_to_section:" + section))
    for row in graph.table_nodes:
        for index in row:
            graph.edges.append((index, index, "same_cell"))
    for left_index, left in enumerate(graph.nodes):
        for right_index, right in enumerate(graph.nodes):
            if left_index == right_index or left.page != right.page:
                continue
            same_row = abs(left.bbox[1] - right.bbox[1]) <= 8
            if same_row:
                graph.edges.append((left_index, right_index, "same_row"))
                if left.bbox[2] <= right.bbox[0]: graph.edges.append((left_index, right_index, "left_of"))
                if left.bbox[0] >= right.bbox[2]: graph.edges.append((left_index, right_index, "right_of"))
            elif left.bbox[3] <= right.bbox[1]:
                graph.edges.append((left_index, right_index, "above"))
            elif right.bbox[3] <= left.bbox[1]:
                graph.edges.append((left_index, right_index, "below"))
            if left.bbox[0] <= right.bbox[2] and right.bbox[0] <= left.bbox[2] and abs(left.bbox[1] - right.bbox[1]) <= 8:
                graph.edges.append((left_index, right_index, "same_cell"))
    return graph


def find_kv_candidates(graph: LayoutGraph, label_aliases: list[str] | FieldProgram | None = None,
                       field_type: Any = None, section_prior: str | None = None, **kwargs) -> list[CandidateValue]:
    if isinstance(label_aliases, FieldProgram):
        aliases = list(label_aliases.select.label_aliases)
        section_prior = section_prior or label_aliases.section_prior or label_aliases.select.section_prior
    else:
        aliases = list(label_aliases or kwargs.get("field_label_aliases") or [])
    aliases = [str(alias).casefold() for alias in aliases if str(alias).strip()]
    candidates: list[CandidateValue] = []
    graph_lines = graph.lines()
    for line_index, line in enumerate(graph_lines):
        text = " ".join(node.text for node in line).strip()
        lower = text.casefold()
        alias = next((alias for alias in aliases if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", lower)), None)
        if not alias:
            continue
        match = re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)\s*[:#-]?\s*(.*)$", text, re.I)
        if match and match.group(1).strip():
            value = match.group(1).strip()
        else:
            # PDF/OCR extraction frequently puts a label and value on
            # adjacent lines. Accept only the immediate next line and keep
            # the relationship local and deterministic.
            if line_index + 1 >= len(graph_lines):
                continue
            next_line = graph_lines[line_index + 1]
            value = " ".join(node.text for node in next_line).strip()
            if not value or any(re.search(r"\b" + re.escape(other) + r"\b", value, re.I) for other in aliases):
                continue
            line = line + next_line
        bbox = [min(node.bbox[0] for node in line), min(node.bbox[1] for node in line), max(node.bbox[2] for node in line), max(node.bbox[3] for node in line)]
        section = line[0].section
        confidence = .98 if not section_prior or section == section_prior else .60
        candidates.append(CandidateValue(value=value, raw_value=value, canonical_value=value, page=line[0].page, bbox=bbox, source_label=alias, evidence_text=text[:240], confidence=confidence))
    return candidates


def find_evidence_for_correction(graph: LayoutGraph, old_value: Any, new_value: Any, current_program: FieldProgram | None = None) -> EvidenceBundle:
    program = current_program or FieldProgram()
    candidates = find_kv_candidates(graph, program)
    old_text, new_text = str(old_value or "").casefold(), str(new_value or "").casefold()
    selected, competing = [], []
    for candidate in candidates:
        value = str(candidate.value or "").casefold()
        hit = EvidenceHit.model_validate({**candidate.model_dump(), "score": candidate.confidence})
        transformed = value
        try:
            from .transform_induction import apply_program
            transformed = str(apply_program(program, candidate.value) or "").casefold()
        except Exception:
            pass
        (selected if value == old_text or value == new_text or transformed == new_text or
         old_text in value or new_text in value else competing).append(hit)
    return EvidenceBundle(evidence_hits=selected, competing_hits=[CompetingHit.model_validate(hit.model_dump()) for hit in competing],
                          section=selected[0].source_label if selected else None, score=max((item.score for item in selected), default=0),
                          debug_trace=["label alias matching", "same-row candidate selection"])


def find_evidence_hits(graph: LayoutGraph, old_value: Any, new_value: Any, field_program: FieldProgram) -> tuple[list[EvidenceHit], list[EvidenceHit]]:
    bundle = find_evidence_for_correction(graph, old_value, new_value, field_program)
    return bundle.evidence_hits, bundle.competing_hits
