# Results — Hán-side input vs OCR backends (PaddleOCR vs Unlimited-OCR vs PaddleOCR-VL-1.6)

Hard numbers from the actual end-to-end run on the full corpus (Đại Nam Thực
Lục tập 4 + 5 + 6, 3 242 PDF pages). All backends ran against the same
Hán source and the same prep/split/embed/align/export stages.

Hardware: 2× RTX 3060 12 GB. PaddleOCR-VL-1.6 OCR output was ingested from
`silver.tar.gz` (`parsing_res_list` JSON per page) via
`scripts/convert_paddlevl.py`; split/embed/align/export re-ran on top of it.

---

## 1. Hán side (input — same for both runs)

Source: `data/raw/Đại Nam Thực Lục - 大南寔錄_full.txt` (Wiki文库 digitized).

| Artifact                              | Pre-fix   | Post-fix    |
|---------------------------------------|-----------|-------------|
| `han_clean.txt` lines                 | 230 805   | 230 805     |
| `han_clean.txt` chars (bytes)         | 13 880 666| 13 880 666  |
| `han_clean.txt` paragraphs (`\n\n`)   | 1         | **633**     |
| `han_sentences.jsonl` sentences       | 34 609    | **54 991**  |
| Sentence max length (chars)           | 989 086   | **13 163**  |

This is the upper bound on aligned-pair count — Vecalign can never produce
more pairs than the Hán-side sentence count.

**Post-fix** = paragraph preservation in `normalize_han` + full-width /
ASCII terminator regex in `split_han` + zero-terminator edict fallback.
See [09_han_pipeline.md](09_han_pipeline.md) for full details.

---

## 2. OCR stage — page coverage

Source: 3 PDFs, 300 DPI rasterized → 3 242 PNG pages
(`tap4 = 1 141`, `tap5 = 945`, `tap6 = 1 156`).

| Per-tập pages OCR'd | PaddleOCR | Unlimited-OCR | PaddleOCR-VL-1.6 |
|---------------------|-----------|---------------|------------------|
| tap4                | 1 141     | 1 141         | 1 141            |
| tap5                | 945       | 944           | 945              |
| tap6                | 1 156     | 1 156         | 1 156            |
| **total**           | **3 242** | **3 241**     | **3 242**        |

Unlimited-OCR missed `tap5_page_0485.txt` (vLLM returned a `BadRequestError`
on that single page — likely an oversize multimodal payload). 0.03 % loss.
Re-runnable by deleting any stale failure and re-running the OCR stage
(idempotent / resumable).

PaddleOCR-VL-1.6 output was delivered pre-computed as JSON
(`parsing_res_list` per page). 100 % page coverage; no failed pages.

---

## 3. OCR stage — text volume

Combined per-tập `tap*.txt` after `--combine`:

| Metric                       | PaddleOCR    | Unlimited-OCR | PaddleOCR-VL-1.6 |
|------------------------------|--------------|---------------|------------------|
| total OCR lines              | 246 939      | 76 507        | **48 467**       |
| total OCR characters (bytes) | 16 103 793   | 21 194 169    | 10 433 119       |

Reading: Unlimited-OCR produces ~3× fewer lines but ~32 % more characters
than PaddleOCR. PaddleOCR's CRNN recognizer emits one line per detected
bounding box, fragmenting paragraphs into many short lines; Unlimited-OCR
reconstructs document layout into longer natural lines, and recovers
text PaddleOCR drops on faded scans (diacritics + edge text).

PaddleOCR-VL-1.6 emits the fewest lines (best paragraph consolidation) and
the fewest bytes because `convert_paddlevl.py` drops page furniture
(`header`, `footer`, `number`, `image`, `table`, `display_formula`) and
keeps only narrative-text blocks (`text`, `doc_title`, `paragraph_title`,
`abstract`, `aside_text`, `footnote`, `reference_content`, `content`,
`vertical_text`). Despite the smaller byte volume, downstream sentence
count is the highest of the three (see §4) — the retained text is
denser and cleaner.

---

## 4. Sentence split (`split_vi`)

After `underthesea.sent_tokenize` (with regex fallback) over the
combined OCR text:

| Sentences in `vi_sentences*.jsonl` | PaddleOCR  | Unlimited-OCR | PaddleOCR-VL-1.6 |
|------------------------------------|------------|---------------|------------------|
| count                              | 55 126     | 62 345        | **66 615**       |

Unlimited-OCR's richer character recall translates into ~7 200 extra
sentence candidates feeding alignment.

PaddleOCR-VL-1.6 adds ~4 270 more (+6.8 % over Unlimited-OCR) despite
smaller byte volume: layout-aware paragraph reconstruction produces cleaner
sentence boundaries for `underthesea.sent_tokenize`, so fewer tokens fall
below the 2-char keep threshold in `split_vi.py`.

---

## 5. Alignment (`vecalign`, score threshold 0.5)

| Pairs in `pairs*.jsonl` | PaddleOCR | Unlimited-OCR | PaddleOCR-VL-1.6 (Hán pre-fix) | PaddleOCR-VL-1.6 (Hán post-fix) |
|-------------------------|-----------|---------------|-------------------------------|--------------------------------|
| aligned pairs           | 33 652    | 33 718        | 33 718                        | **52 710**                     |

Prior three runs all hit the same 33 718 ceiling — capped by the broken
Hán side (34 609 sentences, most of which were oversized monster
segments). Fixing `normalize_han` + `split_han` unblocked **19 000 extra
pairs** with the same OCR backend (PaddleOCR-VL-1.6). Hán-side sentence
pool grew from 34 609 → 54 991, and Vecalign now aligns 96 % of it.

---

## 6. Final deliverable (`export_deliverable`)

`data/final/hvb_*` is the current PaddleOCR-VL-1.6 run; the prior
Unlimited-OCR deliverable is preserved as `hvb_*_unlimited.*` via
`scripts/snapshot_unlimited.sh`.

| Run                              | `_raw.txt` | `_parallel.tsv` | `_parallel.xlsx` | Pairs kept | Over-length drops |
|----------------------------------|------------|-----------------|------------------|------------|-------------------|
| Unlimited-OCR                    | 10.1 M     | 9.2 M           | 4.6 M            | 33 681     | 37                |
| PaddleOCR-VL-1.6 (Hán pre-fix)   | 9.9 M      | 8.9 M           | 4.5 M            | 33 692     | 26                |
| PaddleOCR-VL-1.6 (Hán post-fix)  | 9.9 M      | 21.5 M          | 11.1 M           | **52 693** | 17                |

Current default paths (`data/final/hvb_*`) hold the post-fix run.
Over-length drops fell from 37 → 17 because Hán side no longer emits
989 086-char monster sentences into the aligner. Yield relative to the
new Hán pool: **52 693 / 54 991 = 95.8 %**.

---

## 7. Snapshots of prior runs

Both prior backends' artefacts coexist on disk via the snapshot scripts.

`scripts/snapshot_paddle.sh` (PaddleOCR CRNN → `_paddle` suffix):

```
data/interim/vi_ocr_raw_paddle/
data/interim/vi_ocr_corrected_paddle/
data/interim/vi_sentences_paddle.jsonl
data/interim/vi_embeds_paddle.npy
data/aligned/pairs_paddle.jsonl
```

`scripts/snapshot_unlimited.sh` (Baidu Unlimited-OCR → `_unlimited` suffix):

```
data/interim/vi_ocr_raw_unlimited/
data/interim/vi_ocr_corrected_unlimited/
data/interim/vi_sentences_unlimited.jsonl
data/interim/vi_embeds_unlimited.npy
data/aligned/pairs_unlimited.jsonl
data/final/hvb_raw_unlimited.txt
data/final/hvb_parallel_unlimited.tsv
data/final/hvb_parallel_unlimited.xlsx
```

PaddleOCR-VL-1.6 output now occupies the default paths
(`data/interim/vi_ocr_raw/`, `data/final/hvb_*`).

The old PaddleOCR CRNN `hvb_*` deliverable was not built (the prior run
did not reach `export` before swap); equivalent count would be 33 652
minus its own `HVB_MAX_PAIR_CHARS` drops.

---

## 8. End-to-end numbers at a glance

| Stage              | Hán (input)              | PaddleOCR  | Unlimited-OCR | PaddleOCR-VL-1.6 (Hán post-fix) |
|--------------------|--------------------------|------------|---------------|---------------------------------|
| source units       | 13.88 M chars            | 3 242 pages| 3 242 pages   | 3 242 pages                     |
| OCR pages produced | —                        | 3 242      | 3 241         | **3 242**                       |
| OCR chars produced | —                        | 16.10 M    | 21.19 M       | 10.43 M                         |
| Hán sentences      | 34 609 → **54 991** (fix)| —          | —             | 54 991                          |
| Việt sentences     | —                        | 55 126     | 62 345        | **66 615**                      |
| aligned pairs (≥0.5)| —                       | 33 652     | 33 718        | **52 710**                      |
| deliverable pairs  | —                        | not built  | 33 681        | **52 693**                      |

Two independent wins compounded: (1) PaddleOCR-VL-1.6 on the Việt side —
better page coverage + denser prose after layout-aware furniture filter;
(2) Hán normalize + split fix — paragraph preservation, full-width /
ASCII terminator regex, zero-terminator edict fallback. See
[09_han_pipeline.md](09_han_pipeline.md) for the Hán fix breakdown.

---

## 9. Reproducing the numbers

```bash
# Hán
wc -l data/interim/han_sentences.jsonl

# PaddleOCR snapshot
wc -lc data/interim/vi_ocr_raw_paddle/tap*.txt | tail -1
wc -l   data/interim/vi_sentences_paddle.jsonl
wc -l   data/aligned/pairs_paddle.jsonl

# Unlimited-OCR snapshot
wc -lc data/interim/vi_ocr_raw_unlimited/tap4.txt \
       data/interim/vi_ocr_raw_unlimited/tap5.txt \
       data/interim/vi_ocr_raw_unlimited/tap6.txt
wc -l   data/interim/vi_sentences_unlimited.jsonl
wc -l   data/aligned/pairs_unlimited.jsonl

# PaddleOCR-VL-1.6 (current default paths)
wc -lc data/interim/vi_ocr_raw/tap4.txt \
       data/interim/vi_ocr_raw/tap5.txt \
       data/interim/vi_ocr_raw/tap6.txt
wc -l   data/interim/vi_sentences.jsonl
wc -l   data/aligned/pairs.jsonl

# Deliverable
ls -la data/final/
wc -l  data/final/hvb_parallel.tsv     # add 1 for header if present
```
