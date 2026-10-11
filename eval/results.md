# Evaluation results

## SROIE (n=10)

| Method | vendor_name (exact / ANLS) | vendor_address (exact / ANLS) | invoice_date (exact / ANLS) | total (exact / ANLS) | sec/doc | errors caught | false alarms |
|---|---|---|---|---|---|---|---|
| ocr_llm | 40.0% / 62.0% | 0.0% / 64.2% | 100.0% / 100.0% | 60.0% / 60.0% | 15.2 | 100.0% | 100.0% |
| vlm | 100.0% / 100.0% | 40.0% / 99.2% | 100.0% / 100.0% | 50.0% / 50.0% | 16.7 | 100.0% | 60.0% |

## CORD (n=10)

| Method | subtotal (exact / ANLS) | tax (exact / ANLS) | total (exact / ANLS) | line-item F1 | sec/doc | errors caught | false alarms |
|---|---|---|---|---|---|---|---|
| ocr_llm | 60.0% / 60.0% | 90.0% / 90.0% | 60.0% / 60.0% | 30.0% | 7.2 | 83.3% | 100.0% |
| vlm | 70.0% / 70.0% | 100.0% / 100.0% | 80.0% / 80.0% | 55.0% | 12.6 | 100.0% | 100.0% |

*errors caught*: share of documents with a wrong amount that validation flagged for review.  
*false alarms*: share of fully correct documents that were flagged anyway.
