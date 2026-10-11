"""OCR with docTR: words with positions and confidences, plus a layout-preserving text view.

LLMs read receipts much better when an item and its price stay on the same line, so
`layout_text` regroups words into visual rows by their vertical position and spaces them out
horizontally, roughly reproducing the printed layout in plain text.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image


@dataclass
class Word:
    text: str
    confidence: float
    page: int
    # Box in relative page coordinates (0..1): left, top, right, bottom.
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@lru_cache
def _load_predictor(det_arch: str, reco_arch: str, device: str):
    from doctr.models import ocr_predictor

    model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True, assume_straight_pages=True)
    return model.to(device) if device == "cuda" else model


class OCR:
    def __init__(self, det_arch: str = "fast_base", reco_arch: str = "crnn_vgg16_bn", device: str = "cpu"):
        self.predictor = _load_predictor(det_arch, reco_arch, device)

    def read(self, pages: list[Image.Image]) -> list[Word]:
        result = self.predictor([np.asarray(p) for p in pages])
        words = []
        for page_idx, page in enumerate(result.pages):
            for block in page.blocks:
                for line in block.lines:
                    for w in line.words:
                        (x0, y0), (x1, y1) = w.geometry
                        words.append(Word(w.value, float(w.confidence), page_idx, x0, y0, x1, y1))
        return words


def group_rows(words: list[Word]) -> list[list[Word]]:
    """Cluster words into visual rows: a word joins a row if its centre is within half a line height."""
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.page, w.cy)):
        if rows:
            row = rows[-1]
            row_cy = sum(r.cy for r in row) / len(row)
            row_h = sum(r.height for r in row) / len(row)
            if row[0].page == w.page and abs(w.cy - row_cy) < 0.5 * max(row_h, w.height):
                row.append(w)
                continue
        rows.append([w])
    return [sorted(r, key=lambda w: w.x0) for r in rows]


def layout_text(words: list[Word], width: int = 100) -> str:
    """Render words as monospaced text: one line per visual row, columns roughly preserved."""
    lines = []
    page = 0
    left = min((w.x0 for w in words), default=0.0)  # drop the empty page margin
    for row in group_rows(words):
        if row[0].page != page:
            page = row[0].page
            lines.append(f"--- page {page + 1} ---")
        line = ""
        for w in row:
            col = int((w.x0 - left) * width)
            line += " " * max(1 if line else 0, col - len(line)) + w.text
        lines.append(line.rstrip())
    return "\n".join(lines)
