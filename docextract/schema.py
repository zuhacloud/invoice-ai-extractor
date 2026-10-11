"""The output schema: what a document turns into.

Models are asked to copy amounts *as printed* (strings), and `parse_amount` converts them to
numbers here. That keeps number parsing deterministic: a small model can't misread "60.000"
(sixty thousand rupiah) as 60.0, and "1.234,56" (European) and "1,234.56" both come out right.
"""

import re
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, WithJsonSchema

_NUM_RE = re.compile(r"-?\d[\d.,']*")


def parse_amount(value: Any) -> float | None:
    """Turn '1,234.56', '1.234,56', 'RM 12.50', '60.000' or 12.5 into a float. None if not a number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(" ", " ")
    match = _NUM_RE.search(text)
    if not match:
        return None
    negative = match.group().startswith("-") or (text.startswith("(") and text.endswith(")"))
    num = match.group().replace("'", "").lstrip("-")
    if "," in num and "." in num:
        # Whichever separator comes last is the decimal point.
        decimal = "," if num.rfind(",") > num.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        num = num.replace(thousands, "").replace(decimal, ".")
    elif "," in num or "." in num:
        sep = "," if "," in num else "."
        parts = num.split(sep)
        # "60.000" / "1,591,600": groups of exactly 3 digits are thousands. "12.50" / "3,5" are decimals.
        if len(parts) > 2 or (len(parts[-1]) == 3 and parts[0] not in ("", "0")):
            num = num.replace(sep, "")
        else:
            num = num.replace(sep, ".")
    num = num.strip(".")
    try:
        result = float(num)
    except ValueError:
        return None
    return -result if negative else result


# A number in Python, but a string in the JSON schema the model fills (see module docstring).
Amount = Annotated[
    float | None,
    BeforeValidator(parse_amount),
    WithJsonSchema({"type": ["string", "null"], "description": "Amount exactly as printed, or null"}),
]


class LineItem(BaseModel):
    description: str | None = Field(None, description="Item name as printed")
    quantity: Amount = Field(None, description="Quantity, if printed")
    unit_price: Amount = Field(None, description="Price per unit, if printed")
    amount: Amount = Field(None, description="Line total")


class Invoice(BaseModel):
    """Fields shared by invoices and receipts. Anything not on the document stays null."""

    vendor_name: str | None = Field(None, description="Seller / store / company that issued the document")
    vendor_address: str | None = Field(None, description="Seller's address on one line")
    invoice_number: str | None = Field(None, description="Invoice, receipt or bill number")
    invoice_date: str | None = Field(None, description="Issue date as printed")
    due_date: str | None = Field(None, description="Payment due date as printed (invoices only)")
    currency: str | None = Field(None, description="Currency code or symbol, e.g. USD, RM, Rp")
    line_items: Annotated[list[LineItem], BeforeValidator(lambda v: v or [])] = Field(default_factory=list)
    subtotal: Amount = Field(None, description="Total before tax")
    tax: Amount = Field(None, description="Total tax / VAT / GST")
    total: Amount = Field(None, description="Final amount to pay")


# Leaf fields that get a confidence score and appear as columns in exports.
HEADER_FIELDS = [
    "vendor_name", "vendor_address", "invoice_number", "invoice_date",
    "due_date", "currency", "subtotal", "tax", "total",
]  # fmt: skip


class Issue(BaseModel):
    """Something a human should look at."""

    field: str  # e.g. "total", "line_items[2].amount", or "document"
    code: str  # machine-readable, e.g. "sum_mismatch"
    message: str
    severity: str = "error"  # "error" fails validation; "warning" is only shown


class ExtractionResult(BaseModel):
    source: str  # file name
    method: str  # "ocr_llm" or "vlm"
    data: Invoice
    confidence: dict[str, float] = Field(default_factory=dict)  # field path -> 0..1
    issues: list[Issue] = Field(default_factory=list)
    needs_review: bool = False
    attempts: int = 1
    seconds: float = 0.0
    source_text: str | None = None  # the OCR text the model read (approach A only), shown in the UI
    raw: str | None = Field(None, exclude=True)  # the model's raw JSON, for debugging
