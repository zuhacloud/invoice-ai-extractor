"""Generate synthetic demo invoices in samples/ (fictional company, safe to commit).

    python scripts/make_samples.py

Creates a correct invoice and a copy whose total is wrong, to show the validation step catching it.
"""

from pathlib import Path

import pymupdf as fitz

OUT = Path(__file__).resolve().parents[1] / "samples"

ITEMS = [
    ("Website maintenance (March)", 1, 450.00),
    ("Cloud hosting - standard plan", 3, 39.90),
    ("SSL certificate renewal", 1, 79.00),
    ("Consulting hours", 4.5, 85.00),
]
TAX_RATE = 0.20


def make_invoice(path: Path, total_override: float | None = None) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 in points

    def text(x, y, s, size=10, bold=False, right=False):
        font = "hebo" if bold else "helv"
        if right:
            x -= fitz.get_text_length(s, fontname=font, fontsize=size)
        page.insert_text((x, y), s, fontsize=size, fontname=font)

    text(50, 70, "Northwind Digital Ltd", 18, bold=True)
    text(50, 90, "42 Harbour Street, Bristol BS1 4XY, United Kingdom")
    text(50, 104, "VAT No. GB 123 4567 89")
    text(545, 70, "INVOICE", 22, bold=True, right=True)
    text(545, 92, "Invoice no: INV-2024-0317", right=True)
    text(545, 106, "Date: 14/03/2024", right=True)
    text(545, 120, "Due date: 13/04/2024", right=True)

    text(50, 160, "Bill to:", bold=True)
    text(50, 174, "Bluebell Bakery")
    text(50, 188, "7 Mill Lane, Bath BA2 3DE")

    y = 240
    page.draw_rect(fitz.Rect(45, y - 14, 550, y + 6), color=None, fill=(0.9, 0.92, 0.96))
    text(50, y, "Description", bold=True)
    text(370, y, "Qty", bold=True, right=True)
    text(460, y, "Unit price", bold=True, right=True)
    text(545, y, "Amount", bold=True, right=True)
    subtotal = 0.0
    for desc, qty, price in ITEMS:
        y += 22
        amount = round(qty * price, 2)
        subtotal += amount
        text(50, y, desc)
        text(370, y, f"{qty:g}", right=True)
        text(460, y, f"{price:,.2f}", right=True)
        text(545, y, f"{amount:,.2f}", right=True)
    page.draw_line((45, y + 12), (550, y + 12), color=(0.6, 0.6, 0.6))

    tax = round(subtotal * TAX_RATE, 2)
    total = total_override if total_override is not None else subtotal + tax
    totals = [("Subtotal", subtotal, False), ("VAT 20%", tax, False), ("Total due (GBP)", total, True)]
    for label, value, bold in totals:
        y += 22
        text(460, y, label, bold=bold, right=True)
        text(545, y, f"£{value:,.2f}", bold=bold, right=True)

    text(50, 760, "Payment: bank transfer to Northwind Digital Ltd, sort code 12-34-56, account 00012345.", 8)
    text(50, 772, "Thank you for your business.", 8)
    doc.save(path)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    make_invoice(OUT / "invoice_northwind.pdf")
    make_invoice(OUT / "invoice_northwind_wrong_total.pdf", total_override=1199.40)
    print(f"Wrote sample invoices to {OUT}")


if __name__ == "__main__":
    main()
