from docextract.pipeline import score_fields
from docextract.schema import Invoice, LineItem
from docextract.validate import check, check_grounding, normalize


def codes(inv):
    return {i.code for i in check(inv)}


def make(**kw):
    base = dict(
        vendor_name="ACME",
        invoice_date="2024-03-01",
        line_items=[LineItem(description="a", quantity=2, unit_price=5, amount=10),
                    LineItem(description="b", amount=5)],  # fmt: skip
        subtotal=15,
        tax=1.5,
        total=16.5,
    )
    base.update(kw)
    return Invoice(**base)


def test_consistent_invoice_has_no_issues():
    assert check(make()) == []


def test_items_not_adding_up_is_an_error():
    issues = check(make(subtotal=20, total=21.5))
    assert any(i.code == "items_sum_mismatch" and i.severity == "error" for i in issues)


def test_total_mismatch():
    assert "total_mismatch" in codes(make(total=99))


def test_line_quantity_times_price():
    inv = make(line_items=[LineItem(description="a", quantity=2, unit_price=5, amount=12)], subtotal=12, total=13.5)
    assert "line_mismatch" in codes(inv)


def test_receipt_without_subtotal_checks_items_against_total():
    inv = make(subtotal=None, tax=None, total=20)
    assert "items_sum_mismatch" in codes(inv)
    assert check(make(subtotal=None, tax=None, total=15)) == []


def test_missing_total_and_bad_date():
    inv = normalize(make(total=None, invoice_date="not a date"))
    assert {"missing", "bad_date"} <= codes(inv)


def test_normalize_dates_dayfirst():
    assert normalize(Invoice(invoice_date="03/04/2024")).invoice_date == "2024-04-03"
    assert normalize(Invoice(invoice_date="03/04/2024"), dayfirst=False).invoice_date == "2024-03-04"
    assert normalize(Invoice(invoice_date="15 Jan 2019")).invoice_date == "2019-01-15"


def test_due_before_issue():
    assert "due_before_issue" in codes(make(due_date="2024-02-01"))


def test_failed_check_lowers_confidence():
    inv = make(total=99)
    conf = score_fields(inv, {"total": 0.9, "vendor_name": 0.9}, check(inv))
    assert conf["total"] < 0.9
    assert conf["vendor_name"] == 0.9


def test_grounding_catches_calculated_total():
    ocr = "Subtotal  £1,031.20\nVAT 20%  £206.24\nTotal due (GBP)  £1,199.40"
    printed = make(subtotal=1031.2, tax=206.24, total=1199.4, line_items=[])
    calculated = make(subtotal=1031.2, tax=206.24, total=1237.44, line_items=[])
    assert check_grounding(printed, ocr) == []
    assert [i.field for i in check_grounding(calculated, ocr)] == ["total"]


def test_tax_inclusive_receipt_is_consistent():
    inv = make(line_items=[LineItem(description="ticket", quantity=2, amount=60000)],
               subtotal=60000, tax=5455, total=60000)  # fmt: skip
    assert check(inv) == []
