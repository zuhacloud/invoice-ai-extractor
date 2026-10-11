"""Web UI: upload invoices or receipts and review what was extracted. Started by `python run.py`.

How Streamlit works: the whole script re-runs from top to bottom on every interaction (a click,
an upload). Anything that must survive between re-runs, like the results so far, is kept in
`st.session_state`.

The UI holds no models; it calls the DocExtract API over HTTP. It only imports the light parts
of the package (PDF rendering for the preview, field names).
"""

import io
import os
import sys
from pathlib import Path

import pandas as pd
import pymupdf
import requests
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from docextract.schema import HEADER_FIELDS  # noqa: E402

# 127.0.0.1 rather than localhost: on Windows, localhost tries IPv6 first and stalls ~2 s per request.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

METHODS = {"vlm": "Vision model", "ocr_llm": "OCR + LLM", "both": "Compare both"}
METHOD_HELP = {
    "vlm": "Qwen2.5-VL reads the image directly. Usually the most accurate.",
    "ocr_llm": "docTR reads the text, then Qwen2.5 fills the form. Shows what the OCR saw.",
    "both": "Runs both and shows where they agree. Takes twice as long.",
}
LABELS = {
    "vendor_name": "Vendor", "vendor_address": "Address", "invoice_number": "Invoice no.",
    "invoice_date": "Date", "due_date": "Due date", "currency": "Currency",
    "subtotal": "Subtotal", "tax": "Tax", "total": "Total",
}  # fmt: skip
AMOUNTS = {"subtotal", "tax", "total"}

st.set_page_config(page_title="DocExtract", page_icon="🧾", layout="wide")
st.session_state.setdefault("docs", [])  # newest first: {"name", "data", "results": [...]}


# --- API -----------------------------------------------------------------------------------------
def api_health() -> dict | None:
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        return None


def api_extract(name: str, data: bytes, method: str) -> list[dict] | None:
    try:
        r = requests.post(f"{API_URL}/extract", params={"method": method},
                          files=[("files", (name, data, "application/octet-stream"))], timeout=900)  # fmt: skip
    except requests.RequestException as e:
        st.error(f"Could not reach the API at {API_URL}: {e}")
        return None
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        st.error(f"{name}: {detail}")
        return None
    return r.json()


def api_export(results: list[dict], fmt: str) -> bytes | None:
    try:
        r = requests.post(f"{API_URL}/export", params={"format": fmt}, json=results, timeout=60)
        return r.content if r.ok else None
    except requests.RequestException:
        return None


# --- Formatting ----------------------------------------------------------------------------------
def money(value, currency: str | None = None) -> str:
    if value is None:
        return "–"
    text = f"{value:,.2f}"
    return f"{currency} {text}" if currency else text


def field_value(name: str, value, currency: str | None) -> str:
    if value in (None, ""):
        return "–"
    return money(value, currency) if name in AMOUNTS else str(value)


@st.cache_data(show_spinner=False, max_entries=50)
def preview(name: str, data: bytes) -> list[Image.Image]:
    """First pages of the document as images, for display next to the results."""
    if name.lower().endswith(".pdf"):
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages = []
            for page in list(doc)[:3]:
                pix = page.get_pixmap(dpi=110, colorspace=pymupdf.csRGB)
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
            return pages
    return [Image.open(io.BytesIO(data)).convert("RGB")]


# --- Result views --------------------------------------------------------------------------------
def status_box(r: dict) -> None:
    errors = [i for i in r["issues"] if i["severity"] == "error"]
    warnings = [i for i in r["issues"] if i["severity"] != "error"]
    if errors:
        st.error("**Needs review.** " + " ".join(f"{i['message']}." for i in errors), icon="🚩")
    elif r["needs_review"]:
        st.warning("**Needs review:** the model was unsure about some fields (highlighted below).", icon="🔎")
    else:
        st.success("**All checks passed.** Line items, subtotal, tax and total are consistent.", icon="✅")
    if warnings:
        with st.expander(f"{len(warnings)} note(s)"):
            for i in warnings:
                st.markdown(f"- **{LABELS.get(i['field'], i['field'])}**: {i['message']}")


def fields_table(r: dict) -> None:
    d, conf = r["data"], r["confidence"]
    flagged = {i["field"] for i in r["issues"] if i["severity"] == "error"}
    rows = []
    for name in HEADER_FIELDS:
        if name in AMOUNTS:
            continue
        label = f"🚩 {LABELS[name]}" if name in flagged else LABELS[name]
        rows.append({"Field": label, "Value": field_value(name, d[name], None), "Confidence": conf.get(name)})
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "Field": st.column_config.TextColumn(width=110),
            "Value": st.column_config.TextColumn(width="medium"),
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1, width=110),
        },
    )  # fmt: skip


def totals_row(r: dict) -> None:
    d, conf = r["data"], r["confidence"]
    flagged = {i["field"] for i in r["issues"] if i["severity"] == "error"}
    cols = st.columns(3)
    for col, name in zip(cols, ["subtotal", "tax", "total"], strict=True):
        c = conf.get(name)
        note = "🚩 check" if name in flagged else (f"{c:.0%} sure" if c is not None else None)
        label = f"{LABELS[name]} ({d['currency']})" if d["currency"] else LABELS[name]
        col.metric(label, money(d[name]), delta=note, delta_color="off", delta_arrow="off", border=True)


def line_items(r: dict) -> None:
    items, conf = r["data"]["line_items"], r["confidence"]
    if not items:
        st.caption("No line items found.")
        return
    rows = []
    for i, li in enumerate(items):
        scores = [conf[k] for k in conf if k.startswith(f"line_items[{i}].")]
        rows.append({"Description": li["description"], "Qty": li["quantity"], "Unit price": li["unit_price"],
                     "Amount": li["amount"], "Confidence": min(scores) if scores else None})  # fmt: skip
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "Description": st.column_config.TextColumn(width="medium"),
            "Qty": st.column_config.NumberColumn(format="%g", width="small"),
            "Unit price": st.column_config.NumberColumn(format="accounting"),
            "Amount": st.column_config.NumberColumn(format="accounting"),
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1, width=110),
        },
    )  # fmt: skip
    amounts = [li["amount"] for li in items if li["amount"] is not None]
    target = r["data"]["subtotal"] if r["data"]["subtotal"] is not None else r["data"]["total"]
    if amounts and target is not None:
        ok = abs(sum(amounts) - target) <= max(0.02, 0.005 * abs(target))
        label = "subtotal" if r["data"]["subtotal"] is not None else "total"
        mark = "matches" if ok else "does **not** match"
        st.caption(f"{'✅' if ok else '🚩'} Line items add up to {money(sum(amounts))}, which {mark} the {label} "
                   f"({money(target)}).")  # fmt: skip


def result_view(r: dict) -> None:
    attempts = "1 attempt" if r["attempts"] == 1 else f"{r['attempts']} attempts (retried after a failed check)"
    st.caption(f"{METHODS[r['method']]} · {r['seconds']:.1f} s · {attempts}")
    status_box(r)
    totals_row(r)
    st.markdown("**Details**")
    fields_table(r)
    st.markdown("**Line items**")
    line_items(r)
    if r.get("source_text"):
        with st.expander("What the OCR read"):
            st.code(r["source_text"], language=None)
    with st.expander("JSON"):
        st.json(r["data"], expanded=False)


def compare_view(a: dict, b: dict) -> None:
    """Field by field: where the two approaches agree, a value is very likely right."""
    rows = []
    for name in HEADER_FIELDS:
        va, vb = a["data"][name], b["data"][name]
        same = (va == vb) if name not in AMOUNTS else (va is not None and vb is not None and abs(va - vb) < 0.011)
        rows.append({"Field": LABELS[name], METHODS[a["method"]]: field_value(name, va, None),
                     METHODS[b["method"]]: field_value(name, vb, None),
                     "Agree": "✅" if same else ("–" if va in (None, "") and vb in (None, "") else "❌")})  # fmt: skip
    rows.append({"Field": "Line items", METHODS[a["method"]]: str(len(a["data"]["line_items"])),
                 METHODS[b["method"]]: str(len(b["data"]["line_items"])),
                 "Agree": "✅" if len(a["data"]["line_items"]) == len(b["data"]["line_items"]) else "❌"})  # fmt: skip
    rows.append({"Field": "Seconds", METHODS[a["method"]]: f"{a['seconds']:.1f}",
                 METHODS[b["method"]]: f"{b['seconds']:.1f}", "Agree": ""})  # fmt: skip
    df = pd.DataFrame(rows)
    agree = (df["Agree"] == "✅").sum()
    compared = df["Agree"].isin(["✅", "❌"]).sum()
    st.markdown(f"The two approaches agree on **{agree} of {compared}** fields. Fields where they differ are the "
                "first place to look.")  # fmt: skip
    # A static table: interactive dataframes mis-size themselves inside a tab that starts hidden.
    st.table(df.set_index("Field"))


def document_card(doc: dict) -> None:
    results = doc["results"]
    with st.container(border=True):
        head, badges = st.columns([3, 2], vertical_alignment="center")
        head.subheader(doc["name"], anchor=False)
        with badges:
            for r in results:
                color, word = ("red", "Needs review") if r["needs_review"] else ("green", "OK")
                st.badge(f"{METHODS[r['method']]}: {word}", color=color)
        left, right = st.columns([2, 3], gap="large")
        with left:
            pages = preview(doc["name"], doc["data"])
            for i, page in enumerate(pages):
                st.image(page, width="stretch", caption=f"Page {i + 1}" if len(pages) > 1 else None)
        with right:
            if len(results) == 1:
                result_view(results[0])
            else:
                tabs = st.tabs([METHODS[r["method"]] for r in results] + ["Compare"])
                for tab, r in zip(tabs, results, strict=False):
                    with tab:
                        result_view(r)
                with tabs[-1]:
                    compare_view(results[0], results[1])


def run_extraction(files: list[tuple[str, bytes]], method: str) -> None:
    progress = st.progress(0.0, text="Starting...")
    for k, (name, data) in enumerate(files):
        what = "both methods" if method == "both" else f"the {METHODS[method].lower()}"
        progress.progress(k / len(files), text=f"Reading {name} with {what} ({k + 1}/{len(files)})...")
        results = api_extract(name, data, method)
        if results:
            st.session_state.docs.insert(0, {"name": name, "data": data, "results": results})
    progress.empty()


# --- Page ----------------------------------------------------------------------------------------
with st.sidebar:
    st.title("🧾 DocExtract")
    st.caption("Invoices and receipts in, checked data out. Everything runs on this computer.")
    health = api_health()
    if health is None:
        st.error(f"API not reachable at {API_URL}. Start the app with `python run.py`.")
    elif health["model_server"] == "unreachable":
        st.warning("Ollama isn't running. Start it from the Start menu, then reload this page.")
    else:
        st.caption(f"🟢 Ready · {health['vlm_model']} · {health['llm_model']}")

    method = st.segmented_control("Method", list(METHODS), format_func=METHODS.get, default="vlm",
                                  required=True)  # fmt: skip
    st.caption(METHOD_HELP[method])
    uploads = st.file_uploader("Documents", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "webp"],
                               accept_multiple_files=True)  # fmt: skip
    if st.button("Extract", type="primary", width="stretch", disabled=not uploads or health is None):
        run_extraction([(u.name, u.getvalue()) for u in uploads], method)

    samples = sorted((ROOT / "samples").glob("*.pdf"))
    if samples:
        st.divider()
        st.markdown("**Try a sample**")
        for s in samples:
            if st.button(s.stem.replace("_", " "), key=f"sample-{s.name}", width="stretch",
                         disabled=health is None):  # fmt: skip
                run_extraction([(s.name, s.read_bytes())], method)

docs = st.session_state.docs
if not docs:
    st.title("Extract data from invoices and receipts")
    st.markdown(
        "Upload a PDF, scan or photo on the left, or try one of the samples. For each document you get the vendor, "
        "dates, line items, tax and total, a confidence score per field, and a check that the numbers add up."
    )
    c1, c2, c3 = st.columns(3)
    c1.info("**1. Upload**\n\nPDFs, scans and phone photos. Several at once is fine.", icon="📤")
    c2.info("**2. Review**\n\nRed flags mark totals that don't add up or values the model wasn't sure of.", icon="🔎")
    c3.info("**3. Export**\n\nDownload everything as Excel, CSV or JSON.", icon="📊")
    st.stop()

all_results = [r for d in docs for r in d["results"]]
flagged = sum(any(r["needs_review"] for r in d["results"]) for d in docs)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Documents", len(docs), border=True)
m2.metric("Needs review", flagged, border=True)
m3.metric("Passed all checks", len(docs) - flagged, border=True)
m4.metric("Avg. seconds per document", f"{sum(r['seconds'] for r in all_results) / len(all_results):.1f}", border=True)

e1, e2, e3, e4 = st.columns([1, 1, 1, 3])
for col, fmt, label, mime in [
    (e1, "xlsx", "Excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    (e2, "csv", "CSV", "text/csv"),
    (e3, "json", "JSON", "application/json"),
]:
    data = api_export(all_results, fmt)
    if data:
        col.download_button(f"⬇ {label}", data, file_name=f"extraction.{fmt}", mime=mime, width="stretch")
if e4.button("Clear results"):
    st.session_state.docs = []
    st.rerun()

for doc in docs:
    document_card(doc)
