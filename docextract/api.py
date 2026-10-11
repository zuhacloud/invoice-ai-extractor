"""REST API: upload invoices / receipts, get validated results back. Started by `python run.py`.

    POST /extract?method=ocr_llm|vlm|both   multipart files  -> list of results (JSON)
    POST /export?format=xlsx|csv|json        results (JSON)   -> file download
    GET  /health                                              -> status of the API and the model server

Interactive docs at http://127.0.0.1:8000/docs. Models load on first use and stay in memory.
"""

import io
import tempfile
import threading
import urllib.request
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from . import __version__
from .config import get_settings
from .export import documents_frame, to_excel, to_json
from .pipeline import Pipeline
from .preprocess import SUPPORTED_EXTS
from .schema import ExtractionResult

app = FastAPI(title="DocExtract", version=__version__,
              description="Invoice and receipt extraction to validated JSON / Excel.")  # fmt: skip

_pipelines: dict[str, Pipeline] = {}
# One document at a time: the GPU is shared, and running two at once would only make both slower.
_lock = threading.Lock()

METHODS = {"ocr_llm": ["ocr_llm"], "vlm": ["vlm"], "both": ["ocr_llm", "vlm"]}


def pipeline(method: str) -> Pipeline:
    if method not in _pipelines:
        _pipelines[method] = Pipeline(method)
    return _pipelines[method]


@app.get("/health")
def health() -> dict:
    s = get_settings()
    base = s.llm_base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{base}/api/version", timeout=3):
            model_server = "ok"
    except OSError:
        # Not Ollama (e.g. OpenAI), or Ollama isn't running. Hosted APIs are assumed reachable.
        model_server = "unreachable" if "127.0.0.1" in base or "localhost" in base else "remote"
    return {"status": "ok", "model_server": model_server, "llm_model": s.llm_model, "vlm_model": s.vlm_model}


@app.post("/extract", response_model=list[ExtractionResult])
def extract(files: list[UploadFile] = File(...), method: str = Query("vlm", enum=list(METHODS))):
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            name = Path(f.filename or "document").name
            if Path(name).suffix.lower() not in SUPPORTED_EXTS:
                raise HTTPException(400, f"{name}: unsupported file type (use PDF, PNG, JPG, TIFF or WEBP)")
            path = Path(tmp) / name
            path.write_bytes(f.file.read())
            for m in METHODS[method]:
                # Model errors don't raise: they come back as a flagged result with the error as an issue.
                with _lock:
                    results.append(pipeline(m).run(path, name=name))
    return results


@app.post("/export")
def export(results: list[ExtractionResult], format: str = Query("xlsx", enum=["xlsx", "csv", "json"])):
    if format == "json":
        return Response(to_json(results), media_type="application/json",
                        headers={"Content-Disposition": "attachment; filename=extraction.json"})  # fmt: skip
    if format == "csv":
        return Response(documents_frame(results).to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=extraction.csv"})  # fmt: skip
    buf = io.BytesIO()
    to_excel(results, buf, review_confidence=get_settings().review_confidence)
    return Response(buf.getvalue(), headers={"Content-Disposition": "attachment; filename=extraction.xlsx"},
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")  # fmt: skip
