import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from evaluate import field_scores, line_item_f1  # noqa: E402

from docextract.schema import Invoice, LineItem  # noqa: E402


def test_amounts_compared_as_numbers():
    assert field_scores("total", 12.5, 12.50) == (1.0, 1.0)
    assert field_scores("total", 12.0, 12.5) == (0.0, 0.0)


def test_text_fuzzy():
    exact, anls = field_scores("vendor_name", "ACME SDN BHD", "ACME SDN. BHD")
    assert exact == 0.0 and anls > 0.9


def test_line_item_f1():
    gold = Invoice(line_items=[LineItem(description="Ice Tea", amount=10), LineItem(description="Rice", amount=20)])
    pred = Invoice(line_items=[LineItem(description="ICE TEA", amount=10), LineItem(description="Rice", amount=25)])
    p, r, f1 = line_item_f1(pred, gold)
    assert (p, r) == (0.5, 0.5) and f1 == 0.5
