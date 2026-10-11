"""Document in, validated result out: preprocess -> extract -> normalize -> validate -> (retry) -> score."""

import time
from pathlib import Path

from .config import Settings, get_settings
from .extractors import EXTRACTORS, ExtractionError, feedback_turns
from .preprocess import prepare
from .schema import HEADER_FIELDS, ExtractionResult, Invoice, Issue
from .validate import check, check_grounding, normalize

# Which fields an issue casts doubt on (used to lower their confidence).
_ISSUE_FIELDS = {
    "items_sum_mismatch": ["subtotal", "total", "line_items"],
    "total_mismatch": ["subtotal", "tax", "total"],
}


def score_fields(inv: Invoice, token_conf: dict[str, float], issues: list[Issue]) -> dict[str, float]:
    """Confidence per field: the model's token probability, halved for fields involved in a failed check.

    Fields the model left empty get no score. Without logprobs (some APIs don't return them),
    a neutral 0.8 is used so that validation still drives the review flag.
    """
    conf: dict[str, float] = {}
    for name in HEADER_FIELDS:
        if getattr(inv, name) not in (None, ""):
            conf[name] = token_conf.get(name, 0.8)
    for i, li in enumerate(inv.line_items):
        for name in ("description", "quantity", "unit_price", "amount"):
            if getattr(li, name) is not None:
                path = f"line_items[{i}].{name}"
                conf[path] = token_conf.get(path, 0.8)
    for issue in issues:
        if issue.severity != "error":
            continue
        targets = _ISSUE_FIELDS.get(issue.code, [issue.field])
        for path in conf:
            if any(path == t or path.startswith((t + "[", t + ".")) for t in targets):
                conf[path] = round(conf[path] * 0.5, 4)
    return conf


class Pipeline:
    def __init__(self, method: str = "ocr_llm", settings: Settings | None = None):
        if method not in EXTRACTORS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {', '.join(EXTRACTORS)}")
        self.settings = settings or get_settings()
        self.method = method
        self.extractor = EXTRACTORS[method](self.settings)

    def run(self, path: str | Path, name: str | None = None) -> ExtractionResult:
        """Extract one document. `name` is the file name to report (defaults to the path's)."""
        s = self.settings
        name = name or Path(path).name
        start = time.perf_counter()
        pages = prepare(path, dpi=s.pdf_dpi, max_pages=s.max_pages)
        prepared = self.extractor.prepare(pages)
        source_text = self.extractor.source_text(prepared)

        history: list[dict] = []
        attempts = 0
        best = None  # (n_errors, extraction, invoice, issues)
        while attempts <= s.max_retries:
            attempts += 1
            try:
                ext = self.extractor.extract(prepared, history=history or None)
            except ExtractionError as e:
                if attempts > s.max_retries:
                    if best:
                        break
                    return ExtractionResult(
                        source=name, method=self.method, data=Invoice(), needs_review=True, source_text=source_text,
                        issues=[Issue(field="document", code="parse_error", message=str(e)[:500])],
                        attempts=attempts, seconds=round(time.perf_counter() - start, 2),
                    )  # fmt: skip
                history = []
                continue
            inv = normalize(ext.invoice, dayfirst=s.date_dayfirst)
            issues = check(inv, tol=s.amount_tolerance)
            if source_text is not None:
                issues += check_grounding(inv, source_text)
            errors = [i for i in issues if i.severity == "error"]
            if best is None or len(errors) < best[0]:
                best = (len(errors), ext, inv, issues)
            if not errors:
                break
            history = feedback_turns(ext.raw, [i.message for i in errors])

        _, ext, inv, issues = best
        conf = score_fields(inv, ext.confidence, issues)
        low = [f for f, c in conf.items() if c < s.review_confidence]
        model_conf = score_fields(inv, ext.confidence, [])  # before validation penalties
        for f in low:
            if model_conf[f] < s.review_confidence and not any(i.field == f for i in issues):
                issues.append(Issue(field=f, code="low_confidence", severity="warning",
                                    message=f"Model was unsure about {f} ({model_conf[f]:.0%})"))  # fmt: skip
        return ExtractionResult(
            source=name,
            method=self.method,
            data=inv,
            confidence=conf,
            issues=issues,
            needs_review=any(i.severity == "error" for i in issues) or bool(low),
            attempts=attempts,
            seconds=round(time.perf_counter() - start, 2),
            source_text=source_text,
            raw=ext.raw,
        )
