"""Download the benchmark datasets and convert their labels to our schema.

    python scripts/download_data.py              # test splits of SROIE and CORD
    python scripts/download_data.py --dataset cord --split train

Output, per dataset:
    data/<dataset>/<split>/images/<id>.jpg
    data/<dataset>/<split>/labels.jsonl     {"id", "image", "gold": {...Invoice fields...}}

SROIE (ICDAR 2019): ~1,000 scanned Malaysian receipts labelled with company, address, date, total.
CORD v2: ~1,000 Indonesian receipt photos labelled with line items, subtotal, tax and total.
Together they cover header fields (SROIE) and line items + arithmetic (CORD).
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without Developer Mode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docextract.schema import Invoice, LineItem, parse_amount  # noqa: E402

SOURCES = {
    "sroie": "podbilabs/sroie-donut",
    "cord": "naver-clova-ix/cord-v2",
}


def parquet_files(repo: str, split: str) -> list[str]:
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet") and Path(f).name.startswith(split))


def _first(value):
    """CORD stores a field as a list when it appears more than once on a receipt; take the first."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def sroie_gold(gt: dict) -> Invoice:
    p = gt["gt_parse"]
    return Invoice(vendor_name=p.get("company"), vendor_address=p.get("address"),
                   invoice_date=p.get("date"), total=p.get("total"))  # fmt: skip


def cord_gold(gt: dict) -> Invoice:
    p = gt["gt_parse"]
    menu = p.get("menu") or []
    if isinstance(menu, dict):
        menu = [menu]
    items = []
    for m in menu:
        items.append(LineItem(
            description=_first(m.get("nm")),
            quantity=parse_amount(_first(m.get("cnt"))),
            unit_price=_first(m.get("unitprice")),
            amount=_first(m.get("price")),
        ))  # fmt: skip
    sub = p.get("sub_total") or {}
    total = p.get("total") or {}
    if isinstance(sub, list):
        sub = sub[0]
    if isinstance(total, list):
        total = total[0]
    return Invoice(
        line_items=items,
        subtotal=_first(sub.get("subtotal_price")),
        tax=_first(sub.get("tax_price")),
        total=_first(total.get("total_price")),
    )


GOLD = {"sroie": sroie_gold, "cord": cord_gold}

# Which fields each dataset actually labels (others are left out of its evaluation).
LABELLED_FIELDS = {
    "sroie": ["vendor_name", "vendor_address", "invoice_date", "total"],
    "cord": ["line_items", "subtotal", "tax", "total"],
}


def iter_cord(split: str):
    """CORD is published as parquet files with the image bytes inline."""
    repo = SOURCES["cord"]
    for name in parquet_files(repo, split):
        print(f"[cord] downloading {name}")
        df = pd.read_parquet(hf_hub_download(repo, name, repo_type="dataset"))
        for _, row in df.iterrows():
            yield io.BytesIO(row["image"]["bytes"]), row["ground_truth"]


def iter_sroie(split: str):
    """This SROIE copy is an image folder plus metadata.jsonl; images are fetched one by one as needed."""
    repo = SOURCES["sroie"]
    folder = {"validation": "val"}.get(split, split)
    meta = hf_hub_download(repo, f"{folder}/metadata.jsonl", repo_type="dataset")
    for line in Path(meta).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        yield hf_hub_download(repo, f"{folder}/{rec['file_name']}", repo_type="dataset"), rec["ground_truth"]


ROWS = {"sroie": iter_sroie, "cord": iter_cord}


def download(dataset: str, split: str, limit: int | None, root: Path) -> None:
    out = root / dataset / split
    (out / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for image, ground_truth in ROWS[dataset](split):
            if limit and n >= limit:
                break
            doc_id = f"{dataset}_{split}_{n:04d}"
            img_path = out / "images" / f"{doc_id}.jpg"
            Image.open(image).convert("RGB").save(img_path, quality=95)
            gold = GOLD[dataset](json.loads(ground_truth))
            rec = {"id": doc_id, "image": f"images/{img_path.name}",  # relative to labels.jsonl
                   "fields": LABELLED_FIELDS[dataset], "gold": gold.model_dump()}  # fmt: skip
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[{dataset}] {n} documents -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=[*SOURCES, "all"], default="all")
    ap.add_argument("--split", default="test", help="test (default), validation or train")
    ap.add_argument("--limit", type=int, help="Keep only the first N documents")
    ap.add_argument("--root", default="data", help="Output folder")
    args = ap.parse_args()
    datasets = list(SOURCES) if args.dataset == "all" else [args.dataset]
    for d in datasets:
        download(d, args.split, args.limit, Path(args.root))


if __name__ == "__main__":
    main()
