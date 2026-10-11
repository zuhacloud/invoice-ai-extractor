"""Turn an uploaded file into clean page images.

PDF pages are rendered with PyMuPDF; photos and scans are straightened (deskew) and given
local contrast (CLAHE), which helps OCR on faded thermal-paper receipts and phone photos.
"""

from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTS = IMAGE_EXTS | {".pdf"}


def load_pages(path: str | Path, dpi: int = 200, max_pages: int = 3) -> list[Image.Image]:
    """Read a PDF or image file into a list of RGB page images."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = []
        with fitz.open(path) as doc:
            for page in list(doc)[:max_pages]:
                pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return pages
    if ext in IMAGE_EXTS:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)  # phone photos store rotation in EXIF
            return [img.convert("RGB")]
    raise ValueError(f"Unsupported file type: {ext} (supported: {', '.join(sorted(SUPPORTED_EXTS))})")


def estimate_skew(gray: np.ndarray) -> float:
    """Angle (degrees) of the dominant text lines, from a Hough transform on edges."""
    small = gray
    scale = 1000 / max(gray.shape)
    if scale < 1:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80, minLineLength=small.shape[1] // 6, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):  # (N,1,4) in OpenCV 4, (N,4) in 5
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 15:  # near-horizontal lines only: text rows, table rules
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def deskew(img: Image.Image, min_angle: float = 0.3) -> Image.Image:
    """Rotate small tilts (up to ~15 degrees) back to horizontal."""
    gray = np.asarray(img.convert("L"))
    angle = estimate_skew(gray)
    if abs(angle) < min_angle:
        return img
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def enhance_contrast(img: Image.Image) -> Image.Image:
    """CLAHE on the lightness channel: boosts faint text without blowing out the background."""
    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_chan)
    return Image.fromarray(cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2RGB))


def clean_page(img: Image.Image) -> Image.Image:
    return enhance_contrast(deskew(img))


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    scale = max_side / max(img.size)
    if scale >= 1:
        return img
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def prepare(path: str | Path, dpi: int = 200, max_pages: int = 3, clean: bool = True) -> list[Image.Image]:
    """Load a document and return its cleaned page images."""
    pages = load_pages(path, dpi=dpi, max_pages=max_pages)
    return [clean_page(p) for p in pages] if clean else pages
