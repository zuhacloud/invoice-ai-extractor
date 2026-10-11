"""Command line: extract one or more documents and print / export the results.

    python -m docextract samples/receipt.jpg
    python -m docextract data/invoices/*.pdf --method vlm --out results.xlsx
    python -m docextract receipt.jpg --method both     # compare the two approaches side by side
"""

import argparse
import glob
import sys
from pathlib import Path

from .export import save
from .pipeline import Pipeline
from .preprocess import SUPPORTED_EXTS
from .schema import ExtractionResult


def expand(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in patterns:
        path = Path(p)
        if path.is_dir():
            files += sorted(f for f in path.iterdir() if f.suffix.lower() in SUPPORTED_EXTS)
        else:
            files += [Path(f) for f in sorted(glob.glob(p))] or [path]
    return files


def print_result(r: ExtractionResult) -> None:
    status = "NEEDS REVIEW" if r.needs_review else "OK"
    print(f"\n=== {r.source}  [{r.method}]  {r.seconds:.1f}s  attempts={r.attempts}  {status}")
    for name, value in r.data.model_dump(exclude={"line_items"}).items():
        if value is not None:
            conf = r.confidence.get(name)
            print(f"  {name:<15} {value!s:<45} {'' if conf is None else f'{conf:.0%}'}")
    if r.data.line_items:
        print(f"  line items ({len(r.data.line_items)}):")
        for li in r.data.line_items:
            qty = "" if li.quantity is None else f"{li.quantity:g} x "
            amt = "" if li.amount is None else f"{li.amount:,.2f}"
            print(f"    - {qty}{li.description or '?'}  {amt}")
    for i in r.issues:
        print(f"  [{i.severity}] {i.field}: {i.message}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="docextract", description="Extract invoice / receipt data to JSON, CSV or Excel.")
    ap.add_argument("files", nargs="+", help="PDF or image files, folders, or glob patterns")
    ap.add_argument("--method", choices=["ocr_llm", "vlm", "both"], default="ocr_llm")
    ap.add_argument("--out", help="Save results to .json, .csv or .xlsx")
    args = ap.parse_args(argv)

    files = [f for f in expand(args.files) if f.exists()]
    if not files:
        print("No input files found.", file=sys.stderr)
        return 1
    methods = ["ocr_llm", "vlm"] if args.method == "both" else [args.method]

    results = []
    for method in methods:
        pipe = Pipeline(method)
        for f in files:
            r = pipe.run(f)
            print_result(r)
            results.append(r)
    if args.out:
        save(results, args.out)
        print(f"\nSaved {len(results)} result(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
