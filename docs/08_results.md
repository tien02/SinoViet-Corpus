# Results — Hán-side input vs OCR backends (PaddleOCR vs Unlimited-OCR)

Hard numbers from the actual end-to-end run on the full corpus (Đại Nam Thực
Lục tập 4 + 5 + 6, 3 242 PDF pages). Both backends ran against the same
Hán source and the same prep/split/embed/align/export stages.

Hardware: 2× RTX 3060 12 GB. Pipeline commit: see `git log` around the
"switch OCR backend to Baidu Unlimited-OCR" feat commit.

---

## 1. Hán side (input — same for both runs)

Source: `data/raw/Đại Nam Thực Lục - 大南寔錄_full.txt` (Wiki文库 digitized).

| Artifact                              | Count        |
|---------------------------------------|--------------|
| `han_clean.txt` lines                 | 230 805      |
| `han_clean.txt` chars                 | 13 880 666   |
| `han_sentences.jsonl` sentences       | **34 609**   |

This is the upper bound on aligned-pair count — Vecalign can never produce
more pairs than the Hán-side sentence count.

---

## 2. OCR stage — page coverage

Source: 3 PDFs, 300 DPI rasterized → 3 242 PNG pages
(`tap4 = 1 141`, `tap5 = 945`, `tap6 = 1 156`).

| Per-tập pages OCR'd | PaddleOCR | Unlimited-OCR |
|---------------------|-----------|---------------|
| tap4                | 1 141     | 1 141         |
| tap5                | 945       | 944           |
| tap6                | 1 156     | 1 156         |
| **total**           | **3 242** | **3 241**     |

Unlimited-OCR missed `tap5_page_0485.txt` (vLLM returned a `BadRequestError`
on that single page — likely an oversize multimodal payload). 0.03 % loss.
Re-runnable by deleting any stale failure and re-running the OCR stage
(idempotent / resumable).

---

## 3. OCR stage — text volume

Combined per-tập `tap*.txt` after `--combine`:

| Metric                       | PaddleOCR    | Unlimited-OCR | Δ            |
|------------------------------|--------------|---------------|--------------|
| total OCR lines              | 246 939      | 76 507        | −68.9 %      |
| total OCR characters         | 16 103 793   | 21 194 169    | **+31.6 %**  |

Reading: Unlimited-OCR produces ~3× fewer lines but ~32 % more characters
than PaddleOCR. PaddleOCR's CRNN recognizer emits one line per detected
bounding box, fragmenting paragraphs into many short lines; Unlimited-OCR
reconstructs document layout into longer natural lines, and recovers
text PaddleOCR drops on faded scans (diacritics + edge text).

---

## 4. Sentence split (`split_vi`)

After `underthesea.sent_tokenize` (with regex fallback) over the
combined OCR text:

| Sentences in `vi_sentences*.jsonl` | PaddleOCR  | Unlimited-OCR | Δ        |
|------------------------------------|------------|---------------|----------|
| count                              | 55 126     | **62 345**    | +13.1 %  |

Unlimited-OCR's richer character recall translates into ~7 200 extra
sentence candidates feeding alignment.

---

## 5. Alignment (`vecalign`, score threshold 0.5)

| Pairs in `pairs*.jsonl` | PaddleOCR | Unlimited-OCR | Δ       |
|-------------------------|-----------|---------------|---------|
| aligned pairs           | 33 652    | **33 718**    | +66     |

Both runs aligned ~97 % of the 34 609 Hán-side sentences. Unlimited-OCR
picks up an additional 66 pairs that PaddleOCR-derived embeddings could
not match to any Hán sentence above threshold.

---

## 6. Final deliverable (`export_deliverable`)

`data/final/hvb_*` for the Unlimited-OCR run (PaddleOCR run was overwritten
by the snapshot, see §7):

| File                  | Size   | Content                                                                        |
|-----------------------|--------|--------------------------------------------------------------------------------|
| `hvb_raw.txt`         | 10.1 M | Raw concatenated Vietnamese OCR per tập.                                       |
| `hvb_parallel.tsv`    | 9.2 M  | `pair_id ⇥ han_sentence ⇥ viet_sentence` — **33 681 pairs**.                   |
| `hvb_parallel.xlsx`   | 4.6 M  | Same 33 681 pairs.                                                             |

Drops applied at export: 0 empty pairs, 37 pairs over `HVB_MAX_PAIR_CHARS`
(default 2 000 chars per side — range-merge artefacts of monotonic
alignment). Net **33 681 / 34 609 = 97.3 %** of the Hán-side sentences end
up in a clean parallel record.

---

## 7. Snapshot of the PaddleOCR run

`scripts/snapshot_paddle.sh` renames the old run with `_paddle` suffix so
both backends' artefacts coexist on disk:

```
data/interim/vi_ocr_raw_paddle/
data/interim/vi_ocr_corrected_paddle/
data/interim/vi_sentences_paddle.jsonl
data/interim/vi_embeds_paddle.npy
data/aligned/pairs_paddle.jsonl
```

PaddleOCR's `hvb_*` deliverable was not built in this experiment (the
prior PaddleOCR run did not reach the `export` stage before swap), but
the equivalent count would be 33 652 minus its own
`HVB_MAX_PAIR_CHARS` drops.

---

## 8. End-to-end numbers at a glance

| Stage              | Hán (input)   | PaddleOCR  | Unlimited-OCR |
|--------------------|---------------|------------|---------------|
| source units       | 13.88 M chars | 3 242 pages| 3 242 pages   |
| OCR pages produced | —             | 3 242      | 3 241         |
| OCR chars produced | —             | 16.10 M    | **21.19 M**   |
| sentences          | 34 609        | 55 126     | **62 345**    |
| aligned pairs (≥0.5)| —            | 33 652     | **33 718**    |
| deliverable pairs  | —             | not built  | **33 681**    |

Unlimited-OCR is the winning backend on every measurable axis except
the single failed page in tap5.

---

## 9. Reproducing the numbers

```bash
# Hán
wc -l data/interim/han_sentences.jsonl

# PaddleOCR snapshot
wc -lc data/interim/vi_ocr_raw_paddle/tap*.txt | tail -1
wc -l   data/interim/vi_sentences_paddle.jsonl
wc -l   data/aligned/pairs_paddle.jsonl

# Unlimited-OCR
wc -lc data/interim/vi_ocr_raw/tap*.txt | tail -1
wc -l   data/interim/vi_sentences.jsonl
wc -l   data/aligned/pairs.jsonl

# Deliverable
ls -la data/final/
wc -l  data/final/hvb_parallel.tsv     # add 1 for header if present
```
