"""Central configuration.

Every setting has a default here and can be overridden with an environment variable or a line
in `.env`. Names are matched case-insensitively, so `vlm_model` is set by `VLM_MODEL=...`.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Models: any OpenAI-compatible endpoint (Ollama locally, or OpenAI / Groq / vLLM) ---
    llm_base_url: str = "http://127.0.0.1:11434/v1"  # 127.0.0.1, not localhost: avoids a slow IPv6 attempt on Windows
    llm_api_key: str = "ollama"  # Ollama ignores the key, but the OpenAI client requires one
    # Approach A: OCR text + a text-only LLM fills the schema.
    llm_model: str = "qwen2.5:3b"
    # Approach B: a vision-language model reads the page image directly.
    vlm_model: str = "qwen2.5vl:3b"
    temperature: float = 0.0  # extraction should be deterministic
    request_timeout: float = 300.0

    # --- OCR (docTR) ---
    ocr_det_arch: str = "fast_base"  # text detector
    ocr_reco_arch: str = "crnn_vgg16_bn"  # text recogniser
    device: str = "auto"  # "auto" picks "cuda" when a GPU is visible, else "cpu"

    # --- Preprocessing ---
    pdf_dpi: int = 200  # render resolution for PDF pages
    max_pages: int = 3  # pages read per document
    vlm_max_side: int = 1280  # longest image side sent to the VLM (more pixels = more tokens = more VRAM)

    # --- Validation ---
    amount_tolerance: float = 0.02  # absolute tolerance when checking sums (rounding)
    date_dayfirst: bool = True  # read 03/04/2024 as 3 April (most of the world); False for US invoices
    review_confidence: float = 0.6  # fields below this confidence are flagged for review
    max_retries: int = 1  # re-ask the model once, listing what failed validation


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
