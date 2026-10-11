# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tolerate slow or unstable connections: wait longer per request and retry failed downloads.
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# OpenCV (pulled in by docTR) needs these system libraries even without a display.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The cache mounts keep downloaded packages between builds (outside the image, so it stays small).
# If a build is interrupted, the next one reuses everything already downloaded.

# CPU-only PyTorch keeps the image small (~2 GB lighter). Only OCR runs in this container; the
# language models run in Ollama on the host, which uses the GPU.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Bake the OCR weights (~110 MB) into the image: the first document doesn't wait for a download,
# and the container works offline.
RUN python -c "from doctr.models import ocr_predictor; ocr_predictor('fast_base', 'crnn_vgg16_bn', pretrained=True)"

COPY docextract ./docextract
COPY app ./app
COPY samples ./samples
COPY .streamlit ./.streamlit

EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "docextract.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
