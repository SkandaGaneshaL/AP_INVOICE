from app.layout_graph import build_layout_graph, find_kv_candidates
from app.operators import CorrectionExample, FieldProgram, SelectOp, TransformOp
from app.transform_induction import apply_program, induce_transform_candidates
from app.type_inference import FieldType, infer_field_type
from app.rule_compiler import compile_rule_update


def words(*rows):
    result = []
    for y, row in enumerate(rows):
        x = 0
        for text in row:
            width = max(10, len(text) * 5)
            result.append({"page": 1, "text": text, "bbox": [x, y * 20, x + width, y * 20 + 10]})
            x += width + 5
    return result


def test_identifier_prefix_transform_is_generic():
    kind = infer_field_type("arbitrary_key", "Reference", "HK 9497384", "9497384", "Customer PO HK 9497384")
    candidates = induce_transform_candidates(kind, "HK 9497384", "9497384", [], [])
    assert candidates[0].program.transform[0].op == "strip_leading_alpha_token"
    assert apply_program(candidates[0].program, "HK 9497384") == "9497384"


def test_other_generic_identifier_prefix_is_supported():
    candidates = induce_transform_candidates(FieldType.IDENTIFIER, "PSI-0009280390", "0009280390", [], [])
    assert any(item.program.transform[0].op == "strip_leading_alpha_token" for item in candidates)


def test_layout_graph_selects_labeled_value():
    graph = build_layout_graph(words(("Customer", "PO", "HK", "9497384"), ("Order", "No.", "PS-123")))
    candidates = find_kv_candidates(graph, ["Customer PO"])
    assert candidates and candidates[0].value == "HK 9497384"


def test_compiler_preserves_existing_selection_and_adds_transform():
    existing = FieldProgram(select=SelectOp(label_aliases=["Customer PO"]), type="IDENTIFIER",
                            transform=[TransformOp(op="identity")])
    candidates = induce_transform_candidates(FieldType.IDENTIFIER, "HK 9497384", "9497384", [], [])
    result = compile_rule_update(existing, CorrectionExample(field_key="x", field_type="IDENTIFIER",
        old_value="HK 9497384", new_value="9497384", label_text="Customer PO"), candidates, None)
    assert result.selected_operator == "strip_leading_alpha_token"
    assert result.program.select.label_aliases == ["Customer PO"]
    assert any(item.op == "strip_leading_alpha_token" for item in result.program.transform)
