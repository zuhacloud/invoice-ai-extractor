"""The two extraction approaches being compared.

A. OcrLlmExtractor: docTR reads the text with positions, a text LLM fills the schema.
B. VlmExtractor:    a vision-language model reads the page image and fills the schema directly.

Both expose `prepare(pages)` (the slow, once-per-document part: OCR or resizing) and
`extract(prepared, history)` (one model call, repeated on retry). Both return the same `Invoice`
plus per-field confidences, so validation, export and evaluation treat them identically.
"""

import json
from dataclasses import dataclass

from openai import APIError
from PIL import Image
from pydantic import ValidationError

from .config import Settings, resolve_device
from .llm import LLM, LLMReply, extract_json, field_confidences
from .ocr import OCR, layout_text
from .preprocess import resize_max_side
from .schema import Invoice

SCHEMA_HINT = """{
  "vendor_name": string|null,      // seller / store / company that issued the document
  "vendor_address": string|null,   // seller's address on one line
  "invoice_number": string|null,   // invoice, receipt or bill number
  "invoice_date": string|null,     // issue date exactly as printed
  "due_date": string|null,         // payment due date as printed (invoices only)
  "currency": string|null,         // e.g. "USD", "RM", "Rp", "EUR"
  "line_items": [                  // one entry per purchased item / service, in order
    {"description": string|null, "quantity": string|null, "unit_price": string|null, "amount": string|null}
  ],
  "subtotal": string|null,         // total before tax
  "tax": string|null,              // total tax / VAT / GST
  "total": string|null             // final amount to pay
}"""

RULES = """Rules:
- Return only one JSON object with exactly these keys.
- Copy text and numbers exactly as printed; keep amounts as strings including separators (e.g. "1,250.00").
- Use null for anything that is not on the document. Never invent values.
- Do not include discounts, payments, change or rounding lines as line items.
- "total" is the final amount due (after tax and discounts), not the cash tendered."""

SYSTEM_OCR = f"""You extract structured data from invoices and receipts.
You receive OCR text whose line breaks and spacing follow the printed layout.
Output this JSON schema:
{SCHEMA_HINT}

{RULES}"""

SYSTEM_VLM = f"""You extract structured data from images of invoices and receipts.
Output this JSON schema:
{SCHEMA_HINT}

{RULES}"""


@dataclass
class Extraction:
    invoice: Invoice
    confidence: dict[str, float]
    raw: str
    reply: LLMReply


class ExtractionError(Exception):
    pass


def call(llm: LLM, *args, **kwargs) -> LLMReply:
    """Model call whose API errors become ExtractionErrors, so a bad document is flagged instead of crashing a batch."""
    try:
        return llm.chat_json(*args, **kwargs)
    except APIError as e:
        raise ExtractionError(f"Model API error: {e}") from e


def parse_reply(reply: LLMReply) -> Extraction:
    """Validate the model's JSON against the schema."""
    try:
        data = json.loads(extract_json(reply.text)[0])
        invoice = Invoice.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ExtractionError(f"Model output did not match the schema: {e}") from e
    return Extraction(invoice, field_confidences(reply.text, reply.tokens), reply.text, reply)


def feedback_turns(previous: str, problems: list[str]) -> list[dict]:
    """Conversation turns that show the model its last answer and what failed validation."""
    msg = "Your answer failed these checks:\n- " + "\n- ".join(problems)
    msg += (
        "\nRe-read the document and return the corrected JSON. Keep values that were correct."
        "\nOnly fix values you misread. Never calculate a value: if the printed numbers really"
        " don't add up, return them exactly as printed."
    )
    return [{"role": "assistant", "content": previous}, {"role": "user", "content": msg}]


class OcrLlmExtractor:
    name = "ocr_llm"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ocr = OCR(settings.ocr_det_arch, settings.ocr_reco_arch, resolve_device(settings.device))
        self.llm = LLM(settings, settings.llm_model)

    def prepare(self, pages: list[Image.Image]) -> str:
        """Run OCR once; retries reuse the text."""
        return layout_text(self.ocr.read(pages))

    def source_text(self, text: str) -> str | None:
        """Text the extracted values can be checked against."""
        return text

    def extract(self, text: str, history: list[dict] | None = None) -> Extraction:
        reply = call(self.llm, SYSTEM_OCR, f"OCR text:\n```\n{text}\n```", history=history)
        return parse_reply(reply)


class VlmExtractor:
    name = "vlm"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLM(settings, settings.vlm_model)

    def prepare(self, pages: list[Image.Image]) -> list[Image.Image]:
        return [resize_max_side(p, self.settings.vlm_max_side) for p in pages]

    def source_text(self, images: list[Image.Image]) -> str | None:
        return None  # no text to check against (running OCR here would defeat the comparison)

    def extract(self, images: list[Image.Image], history: list[dict] | None = None) -> Extraction:
        reply = call(self.llm, SYSTEM_VLM, "Extract the data from this document.", images=images, history=history)
        return parse_reply(reply)


EXTRACTORS = {"ocr_llm": OcrLlmExtractor, "vlm": VlmExtractor}
