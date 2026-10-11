import pytest

from docextract.schema import Invoice, parse_amount


@pytest.mark.parametrize(
    "text, expected",
    [
        ("12.50", 12.5),
        ("1,234.56", 1234.56),
        ("1.234,56", 1234.56),  # European
        ("60.000", 60000.0),  # Indonesian rupiah thousands
        ("1,591,600", 1591600.0),
        ("Rp 45.500", 45500.0),
        ("RM 8.70", 8.7),
        ("$1,000", 1000.0),
        ("3,5", 3.5),
        ("0.500", 0.5),
        ("-4.00", -4.0),
        ("(4.00)", -4.0),
        ("12", 12.0),
        (7, 7.0),
        ("", None),
        ("n/a", None),
        (None, None),
    ],
)
def test_parse_amount(text, expected):
    assert parse_amount(text) == expected


def test_invoice_accepts_string_amounts():
    inv = Invoice.model_validate({"total": "1,250.00", "line_items": [{"description": "x", "amount": "50"}]})
    assert inv.total == 1250.0
    assert inv.line_items[0].amount == 50.0


def test_schema_asks_model_for_strings():
    schema = Invoice.model_json_schema()
    assert schema["properties"]["total"]["type"] == ["string", "null"]
