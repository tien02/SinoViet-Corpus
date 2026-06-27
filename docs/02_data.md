# Data Specification

Mô tả input, output, schema JSONL cho HVB pipeline.

## Input data (`data/raw/`)

### Hán TXT

**Path:** `data/raw/Đại Nam Thực Lục - 大南寔錄_full.txt`

**Source:** Wiki文库 (Vietnamese Wikipedia version của 大南寔錄)

**Stats:**
- 233,493 lines
- 4.9M chars
- Encoding: UTF-8
- Format: plain text với markdown-style headers (`#`, `##`)

**Cấu trúc:**
```
大南寔錄

## 大南寔錄前編

# 上諭

姊妹计划 : 数据项
紹治四年 三月十一日，上諭：...
```

Wiki headers markers (sẽ bị strip ở Stage 1a):
- `姊妹计划 : 数据项` — wiki project link
- `#`, `##` — markdown headers
- `---` — section separators
- `【...】` — wiki annotation blocks

### Việt PDFs

**Paths:**
- `data/raw/Đại Nam Thực Lục tập 4 - Quốc Sử Quán Triều Nguyễn.pdf` — 1,141 pages, 81.8 MB
- `data/raw/Đại Nam Thực Lục tập 5 - Quốc Sử Quán Triều Nguyễn.pdf` — 945 pages, 35.1 MB
- `data/raw/Đại Nam Thực Lục tập 6 - Quốc Sử Quán Triều Nguyễn.pdf` — 1,156 pages, 66.3 MB

**Total:** 3,242 pages, 183 MB

**Source:** Bản in Quốc Sử Quán Triều Nguyễn (electronically scanned)

**Format:** PDF với scanned images (không phải born-digital). Layout 2 cột hoặc full-width tùy trang.

**Encoding note:** Font Việt có thể chứa Nôm characters và dấu câu cổ (。、；：). OCR cần handle các font này.

## Output data

### Stage 1: `data/interim/`

| File | Format | Schema |
|------|--------|--------|
| `han_clean.txt` | UTF-8 text | Plain text, wiki markers removed |
| `vi_pages/tap{4,5,6}_{page:04d}.png` | PNG image | 300 DPI grayscale/color, named by tập + page |
| `vi_ocr_raw/tap{4,5,6}_page_{n:04d}.txt` | UTF-8 text | OCR raw per page |
| `vi_ocr_raw/tap{4,5,6}.txt` | UTF-8 text | OCR raw concatenated per tập |
| `vi_ocr_corrected/tap{4,5,6}.txt` | UTF-8 text | OCR sau LLM post-fix |

### Stage 3: Sentence JSONL

**`data/interim/han_sentences.jsonl`** — một JSON object per line:

```json
{"idx": 0, "text": "紹治四年三月十一日，上諭：茲據奉充史館總裁之太保..."}
{"idx": 1, "text": "覽奏深慰朕懷，且昭代信史，所以垂示將來..."}
```

**`data/interim/vi_sentences.jsonl`**:

```json
{"idx": 0, "tap": "tap4", "page": 1, "text": "Năm Thiệu Trị thứ tư, ngày 11 tháng 3, có chỉ dụ:"}
{"idx": 1, "tap": "tap4", "page": 1, "text": "Nay theo bài tấu của Thái bảo Văn Minh Điện Đại học sĩ..."}
```

### Stage 4: Embeddings

**`data/interim/{han,vi}_embeds.npy`** — numpy float32 arrays:
- Shape: `(N_sentences, 768)` cho LaBSE
- Normalized: L2 norm = 1 mỗi row (cosine = dot product)

### Stage 5: Aligned pairs

**`data/aligned/pairs.jsonl`** — một cặp per line:

```json
{
  "src_idx": [42],
  "tgt_idx": [58],
  "src": "紹治四年三月十一日",
  "tgt": "Năm Thiệu Trị thứ tư, ngày 11 tháng 3",
  "score": 0.823
}
```

**Trường hợp 1-many alignment:**
```json
{
  "src_idx": [100],
  "tgt_idx": [120, 121, 122],
  "src": "...",
  "tgt": "...",
  "score": 0.71
}
```

**Filter:** Pairs với `score < 0.5` bị drop (`ALIGN_MIN_SCORE` trong config).

### Stage 6: Entities

**`data/interim/entities_han.jsonl`:**
```json
{"idx": 0, "text": "紹治四年三月十一日，上諭...", "entities": [
  {"text": "紹治", "label": "TIME"},
  {"text": "太保", "label": "TITLE"}
]}
```

**`data/interim/entities_vi.jsonl`:**
```json
{"idx": 0, "tap": "tap4", "page": 1, "text": "...", "entities": [
  {"text": "Thiệu Trị", "label": "DATE"},
  {"text": "Thái bảo", "label": "TITLE"}
]}
```

**`data/aligned/entities.jsonl`:**
```json
{
  "src_idx": [42],
  "tgt_idx": [58],
  "han_entities": [{"text": "紹治", "label": "TIME"}],
  "vi_entities": [{"text": "Thiệu Trị", "label": "DATE"}],
  "matches": [{"han": "紹治", "vi": "Thiệu Trị", "score": 1.0}]
}
```

### Stage 7: Final corpus

**`data/final/hvb_corpus.jsonl`:**
```json
{
  "src": "紹治四年三月十一日",
  "tgt": "Năm Thiệu Trị thứ tư, ngày 11 tháng 3",
  "src_idx": [42],
  "tgt_idx": [58],
  "labse_score": 0.823,
  "entities": [
    {"han": "紹治", "vi": "Thiệu Trị", "score": 1.0}
  ]
}
```

### Stage 7: Eval reports

**`data/final/eval/auto_metrics.json`:**
```json
{
  "n_pairs": 4231,
  "labse_cosine": {"mean": 0.71, "median": 0.74, "stdev": 0.12},
  "comet_qe": {"mean": 0.52, "median": 0.55},
  "bertscore": {"precision_mean": 0.82, "recall_mean": 0.79, "f1_mean": 0.80},
  "bleu_chrf": {
    "bleu_zh2vi": 12.3, "bleu_vi2zh": 11.8,
    "chrf_zh2vi": 38.2, "chrf_vi2zh": 37.5
  }
}
```

**`data/final/eval/flores_sanity.json`:**
```json
{
  "n_pairs": 2009,
  "diagonal_cosine_mean": 0.83,
  "precision_at_1": 0.96,
  "comet_qe_mean_first500": 0.42
}
```

**`data/final/eval/round_trip.json`:** summary + 500 sample pairs với `han_roundtrip`.

**`data/final/eval/holdout_mt.json`:**
```json
{"n_train": 3385, "n_test": 846, "bleu": 18.4, "chrf": 42.1}
```
Hoặc `{"skipped": true, "reason": "insufficient pairs"}` nếu < 5000.

**`data/final/eval/llm_ensemble_judge.json`:**
```json
{
  "summary": {
    "n_sample": 500,
    "models": ["qwen2.5:7b", "seallm:7b"],
    "krippendorff_alpha": {
      "adequacy": 0.52, "fluency": 0.48, "alignment": 0.55,
      "fidelity": 0.50, "terminology": 0.61
    },
    "mean_per_model": {
      "qwen2.5:7b": {"adequacy": 3.8, "fluency": 3.9, "alignment": 4.1, "fidelity": 3.7, "terminology": 3.5},
      "seallm:7b": {"adequacy": 3.6, "fluency": 3.7, "alignment": 3.9, "fidelity": 3.5, "terminology": 3.4}
    }
  }
}
```

## Gold data (`data/gold/`)

### OCR gold (`data/gold/ocr_gold/`)

10 trang gõ tay cho CER eval:
- `tap4_page_0042.txt` — text chuẩn
- `tap5_page_0100.txt`

Tên file khớp với `data/interim/vi_pages/` để compare.

### NER gold (`data/gold/ner_gold/`)

Optional: JSONL với annotation thủ công:
```json
{"text": "Năm Gia Long thứ năm", "entities": [["Năm Gia Long thứ năm", "DATE"]]}
```

## Benchmark data (`data/benchmark/`)

### FLORES-200 zh-vi

Auto-download qua HuggingFace `datasets` nếu chưa có. Có thể pre-cache:
```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('facebook/flores', 'zho_Hans_vie_Latn', split='devtest')
print(len(ds))
"
```

Hoặc copy thủ công vào `data/benchmark/flores_zh_vi/{zho_Hans,vie_Latn}.devtest`.

### OPUS zh-vi

Tải từ https://opus.nlpl.eu/ — không auto-download. Format tương tự FLORES.

## Checkpoint (`data/interim/.checkpoint/`)

Mỗi stage tạo 1 file flag khi hoàn thành:
```
.checkpoint/normalize_han
.checkpoint/pdf_to_images
.checkpoint/paddle_ocr
```

Xóa file flag để re-run stage đó.

## Naming conventions

- **Tập prefix:** `tap4`, `tap5`, `tap6` (không dấu, lowercase)
- **Page zero-pad:** `{page:04d}` — ví dụ `tap4_0042.png`
- **JSONL:** 1 record per line, UTF-8, `ensure_ascii=False`
- **Embeddings:** `.npy` numpy float32
- **No Vietnamese diacritics** trong filename trung gian (ASCII only)
