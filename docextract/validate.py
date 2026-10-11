"""Business-rule checks: catch extraction mistakes before they reach someone's books.

`normalize` cleans values (dates to ISO 8601, whitespace), `check` returns a list of Issues.
Arithmetic checks are the strongest signal we have: if the line items, subtotal, tax and total
all agree, it is very unlikely that any one of them was misread.
"""

import re
from datetime import date

from dateutil import parser as dateparser

from .schema import Invoice, Issue, parse_amount

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TOKEN_RE = re.compile(r"\d[\d.,']*")


def parse_date(text: str | None, dayfirst: bool = True) -> date | None:
    if not text:
        return None
    text = text.strip()
    try:
        if _ISO_RE.match(text):
            return date.fromisoformat(text)
        return dateparser.parse(text, dayfirst=dayfirst, fuzzy=True).date()
    except (ValueError, OverflowError):
        return None


def normalize(inv: Invoice, dayfirst: bool = True) -> Invoice:
    """Return a cleaned copy: trimmed strings, ISO dates where they parse, empty line items dropped."""
    inv = inv.model_copy(deep=True)
    for name in ("vendor_name", "vendor_address", "invoice_number", "currency"):
        value = getattr(inv, name)
        if isinstance(value, str):
            setattr(inv, name, " ".join(value.split()) or None)
    for name in ("invoice_date", "due_date"):
        parsed = parse_date(getattr(inv, name), dayfirst)
        if parsed:
            setattr(inv, name, parsed.isoformat())
    inv.line_items = [li for li in inv.line_items if li.description or li.amount is not None]
    return inv


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= max(tol, 0.005 * abs(b))  # absolute tolerance, or 0.5% for large amounts


def check(inv: Invoice, tol: float = 0.02) -> list[Issue]:
    issues: list[Issue] = []

    # Required fields
    if inv.total is None:
        issues.append(Issue(field="total", code="missing", message="No total amount found"))
    if not inv.vendor_name:
        issues.append(Issue(field="vendor_name", code="missing", message="No vendor name found", severity="warning"))

    # Dates
    for name in ("invoice_date", "due_date"):
        value = getattr(inv, name)
        if value and not _ISO_RE.match(value):
            issues.append(Issue(field=name, code="bad_date", message=f"Could not parse date '{value}'"))
    if inv.invoice_date and inv.due_date and _ISO_RE.match(inv.invoice_date) and _ISO_RE.match(inv.due_date):
        if inv.due_date < inv.invoice_date:
            issues.append(Issue(field="due_date", code="due_before_issue", message="Due date is before the issue date"))
    if inv.invoice_date and _ISO_RE.match(inv.invoice_date) and inv.invoice_date > date.today().isoformat():
        issues.append(Issue(field="invoice_date", code="future_date", message="Invoice date is in the future",
                            severity="warning"))  # fmt: skip

    # Amounts
    for name in ("subtotal", "tax", "total"):
        value = getattr(inv, name)
        if value is not None and value < 0:
            issues.append(Issue(field=name, code="negative", message=f"{name} is negative", severity="warning"))

    # Line items: quantity x unit price = amount
    for i, li in enumerate(inv.line_items):
        if li.quantity is not None and li.unit_price is not None and li.amount is not None:
            if not _close(li.quantity * li.unit_price, li.amount, tol):
                issues.append(Issue(
                    field=f"line_items[{i}].amount", code="line_mismatch",
                    message=f"{li.quantity:g} x {li.unit_price:,.2f} = {li.quantity * li.unit_price:,.2f}, "
                            f"but the line says {li.amount:,.2f}",
                    severity="warning",
                ))  # fmt: skip

    # Sum of line items = subtotal (or = total when there is no subtotal and no tax)
    amounts = [li.amount for li in inv.line_items if li.amount is not None]
    if amounts and len(amounts) == len(inv.line_items):
        items_sum = sum(amounts)
        target, target_name = (inv.subtotal, "subtotal") if inv.subtotal is not None else (None, None)
        if target is None and inv.tax in (None, 0) and inv.total is not None:
            target, target_name = inv.total, "total"
        if target is not None and not _close(items_sum, target, tol):
            issues.append(Issue(
                field=target_name, code="items_sum_mismatch",
                message=f"Line items add up to {items_sum:,.2f}, but {target_name} is {target:,.2f}",
            ))  # fmt: skip

    # Subtotal + tax = total. Prices that already include tax (common on receipts) give
    # subtotal = total, with the tax printed for information only; that is consistent too.
    if inv.subtotal is not None and inv.total is not None:
        expected = inv.subtotal + (inv.tax or 0)
        tax_inclusive = _close(inv.subtotal, inv.total, tol)
        if not tax_inclusive and not _close(expected, inv.total, tol):
            msg = (f"Subtotal {inv.subtotal:,.2f} + tax {inv.tax or 0:,.2f} = {expected:,.2f}, "
                   f"but total is {inv.total:,.2f}")
            # Always an error: a total that doesn't reconcile must not reach the books unchecked, even
            # when the cause is legitimate (service charge, discount). The eval's "false alarms"
            # column measures how often that happens.
            issues.append(Issue(field="total", code="total_mismatch", message=msg))

    return issues


def check_grounding(inv: Invoice, source_text: str) -> list[Issue]:
    """Every extracted amount must appear in the OCR text.

    This catches the model *calculating* a number instead of reading it, e.g. "fixing" a total
    that doesn't add up, which would hide a real discrepancy on the document.
    """
    printed = {v for tok in _TOKEN_RE.findall(source_text) if (v := parse_amount(tok)) is not None}
    printed = {round(abs(v), 2) for v in printed}
    issues = []
    for name in ("subtotal", "tax", "total"):
        value = getattr(inv, name)
        if value is not None and round(abs(value), 2) not in printed:
            issues.append(Issue(field=name, code="not_in_document",
                                message=f"{name} {value:,.2f} does not appear on the document"))  # fmt: skip
    return issues
