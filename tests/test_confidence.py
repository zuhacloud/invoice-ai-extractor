import math

from docextract.llm import _leaf_spans, field_confidences
from docextract.ocr import Word, layout_text


def test_leaf_spans_nested():
    text = '{"total": "12.50", "line_items": [{"amount": "3"}, {"amount": null}], "tax": 1}'
    spans = _leaf_spans(text)
    assert text[slice(*spans["total"])] == '"12.50"'
    assert text[slice(*spans["line_items[0].amount"])] == '"3"'
    assert text[slice(*spans["line_items[1].amount"])] == "null"
    assert text[slice(*spans["tax"])] == "1"


def test_field_confidence_uses_least_certain_token():
    tokens = [('{"', 0.0), ("total", 0.0), ('": "', 0.0), ("12", math.log(0.9)), (".", 0.0),
              ("50", math.log(0.4)), ('"}', 0.0)]  # fmt: skip
    text = "".join(t for t, _ in tokens)
    conf = field_confidences(text, tokens)
    assert abs(conf["total"] - 0.4) < 1e-3


def test_field_confidence_without_logprobs():
    assert field_confidences('{"total": "1"}', []) == {}


def test_layout_text_keeps_item_and_price_on_one_line():
    words = [
        Word("Coffee", 0.99, 0, 0.05, 0.100, 0.20, 0.120),
        Word("4.50", 0.99, 0, 0.80, 0.102, 0.90, 0.122),  # same row, slightly lower
        Word("TOTAL", 0.99, 0, 0.05, 0.200, 0.20, 0.220),
        Word("4.50", 0.99, 0, 0.80, 0.200, 0.90, 0.220),
    ]
    lines = layout_text(words).splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("Coffee") and lines[0].endswith("4.50")


def test_confidence_with_fenced_json():
    tokens = [("```json\n", 0.0), ('{"total": "', 0.0), ("7", math.log(0.5)), ('"}', 0.0), ("\n```", 0.0)]
    text = "".join(t for t, _ in tokens)
    assert abs(field_confidences(text, tokens)["total"] - 0.5) < 1e-3
