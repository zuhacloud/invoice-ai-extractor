"""Benchmark both extraction approaches on SROIE / CORD.

    python eval/evaluate.py --limit 50                      # both datasets, both methods
    python eval/evaluate.py --dataset cord --methods vlm --limit 20

Metrics, per field:
    exact  - normalised exact match (amounts compared as numbers, dates as ISO dates)
    anls   - Average Normalised Levenshtein Similarity: string similarity, 0 below 0.5 (standard for DocVQA)
Plus line-item F1 (CORD), seconds per document, and how often validation flags a document whose
extraction is actually wrong ("errors caught").

Writes predictions to eval/runs/<dataset>_<method>.jsonl and a summary table to eval/results.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docextract.config import get_settings  # noqa: E402
from docextract.pipeline import Pipeline  # noqa: E402
from docextract.schema import Invoice  # noqa: E402
from docextract.validate import normalize  # noqa: E402

AMOUNT_FIELDS = {"subtotal", "tax", "total", "quantity", "unit_price", "amount"}


def norm_text(v) -> str:
    return " ".join(str(v).casefold().split()) if v is not None else ""


def field_scores(name: str, pred, gold) -> tuple[float, float]:
    """(exact, anls) for one field."""
    if gold in (None, ""):
        return (1.0, 1.0) if pred in (None, "") else (0.0, 0.0)
    if pred in (None, ""):
        return 0.0, 0.0
    if name in AMOUNT_FIELDS:
        ok = abs(float(pred) - float(gold)) < 0.011
        return float(ok), float(ok)
    p, g = norm_text(pred), norm_text(gold)
    sim = Levenshtein.normalized_similarity(p, g)
    return float(p == g), (sim if sim >= 0.5 else 0.0)


def line_item_f1(pred: Invoice, gold: Invoice) -> tuple[float, float, float]:
    """Greedy matching: a predicted item is correct if its description is similar (>= 0.8) and its amount equal."""
    unmatched = list(gold.line_items)
    tp = 0
    for p in pred.line_items:
        best, best_sim = None, 0.8
        for g in unmatched:
            sim = Levenshtein.normalized_similarity(norm_text(p.description), norm_text(g.description))
            same_amount = p.amount is not None and g.amount is not None and abs(p.amount - g.amount) < 0.011
            if same_amount and sim >= best_sim:
                best, best_sim = g, sim
        if best is not None:
            unmatched.remove(best)
            tp += 1
    precision = tp / len(pred.line_items) if pred.line_items else float(not gold.line_items)
    recall = tp / len(gold.line_items) if gold.line_items else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def load_labels(dataset: str, split: str, limit: int | None) -> list[dict]:
    path = ROOT / "data" / dataset / split / "labels.jsonl"
    if not path.exists():
        sys.exit(f"{path} not found. Run: python scripts/download_data.py --dataset {dataset} --split {split}")
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in recs:
        r["image"] = str(path.parent / r["image"])
    return recs[:limit] if limit else recs


def evaluate(dataset: str, method: str, recs: list[dict], pipe: Pipeline) -> dict:
    settings = get_settings()
    out_dir = ROOT / "eval" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = recs[0]["fields"]
    scalar_fields = [f for f in fields if f != "line_items"]
    totals = {f: [0.0, 0.0] for f in scalar_fields}
    f1s, secs, wrong, wrong_flagged, right, right_flagged = [], [], 0, 0, 0, 0

    with open(out_dir / f"{dataset}_{method}.jsonl", "w", encoding="utf-8") as fout:
        for k, rec in enumerate(recs, 1):
            gold = normalize(Invoice.model_validate(rec["gold"]), dayfirst=settings.date_dayfirst)
            t0 = time.perf_counter()
            result = pipe.run(rec["image"])
            secs.append(time.perf_counter() - t0)
            pred = result.data

            doc_correct = True
            per_field = {}
            for f in scalar_fields:
                exact, anls = field_scores(f, getattr(pred, f), getattr(gold, f))
                totals[f][0] += exact
                totals[f][1] += anls
                per_field[f] = exact
                if f in AMOUNT_FIELDS and not exact:
                    doc_correct = False
            if "line_items" in fields:
                _, _, f1 = line_item_f1(pred, gold)
                f1s.append(f1)
                per_field["line_items_f1"] = round(f1, 3)
            # "Errors caught": among documents with a wrong amount, how many did validation flag?
            if doc_correct:
                right += 1
                right_flagged += result.needs_review
            else:
                wrong += 1
                wrong_flagged += result.needs_review

            fout.write(json.dumps({"id": rec["id"], "scores": per_field, "result": result.model_dump(),
                                   "gold": gold.model_dump()}, ensure_ascii=False) + "\n")  # fmt: skip
            print(f"[{dataset}/{method}] {k}/{len(recs)}  {secs[-1]:.1f}s  {per_field}")

    n = len(recs)
    return {
        "dataset": dataset,
        "method": method,
        "n": n,
        "fields": {f: {"exact": totals[f][0] / n, "anls": totals[f][1] / n} for f in scalar_fields},
        "line_items_f1": sum(f1s) / len(f1s) if f1s else None,
        "sec_per_doc": sum(secs) / n,
        "errors_caught": wrong_flagged / wrong if wrong else None,
        "false_alarms": right_flagged / right if right else None,
    }


def pct(v) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def write_report(summaries: list[dict]) -> str:
    lines = ["# Evaluation results", ""]
    for ds in dict.fromkeys(s["dataset"] for s in summaries):
        rows = [s for s in summaries if s["dataset"] == ds]
        fields = list(rows[0]["fields"])
        lines += [f"## {ds.upper()} (n={rows[0]['n']})", ""]
        header = ["Method", *[f"{f} (exact / ANLS)" for f in fields]]
        if rows[0]["line_items_f1"] is not None:
            header.append("line-item F1")
        header += ["sec/doc", "errors caught", "false alarms"]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for s in rows:
            cells = [s["method"]]
            cells += [f"{pct(s['fields'][f]['exact'])} / {pct(s['fields'][f]['anls'])}" for f in fields]
            if s["line_items_f1"] is not None:
                cells.append(pct(s["line_items_f1"]))
            cells += [f"{s['sec_per_doc']:.1f}", pct(s["errors_caught"]), pct(s["false_alarms"])]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "*errors caught*: share of documents with a wrong amount that validation flagged for review.  ",
        "*false alarms*: share of fully correct documents that were flagged anyway.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["sroie", "cord", "all"], default="all")
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods", nargs="+", choices=["ocr_llm", "vlm"], default=["ocr_llm", "vlm"])
    ap.add_argument("--limit", type=int, default=50, help="Documents per dataset (0 = all)")
    args = ap.parse_args()

    datasets = ["sroie", "cord"] if args.dataset == "all" else [args.dataset]
    summaries = []
    for method in args.methods:
        pipe = Pipeline(method)
        for ds in datasets:
            summaries.append(evaluate(ds, method, load_labels(ds, args.split, args.limit or None), pipe))
    report = write_report(summaries)
    (ROOT / "eval" / "results.md").write_text(report, encoding="utf-8")
    (ROOT / "eval" / "results.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
