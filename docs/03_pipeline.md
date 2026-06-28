# Pipeline chi tiết

Giải thích từng stage, code flow, params, output.

## Tổng quan pipeline

```
Stage 1: prep
  ├── 1a normalize_han     (Han TXT -> han_clean.txt)
  └── 1b pdf_to_images     (3 PDF -> 3242 PNG)

Stage 2: ocr
  ├── 2a paddle_ocr        (PNG -> vi_ocr_raw/*.txt)
  └── 2b llm_correct       (raw -> vi_ocr_corrected/*.txt via vLLM)

Stage 3: split
  ├── 3a split_han         (han_clean.txt -> han_sentences.jsonl)
  └── 3b split_vi          (vi_ocr_corrected -> vi_sentences.jsonl)

Stage 4: embed
  └── labse_embed          (both sides -> {han,vi}_embeds.npy)

Stage 5: align
  └── vecalign_runner      (sentences + embeds -> pairs.jsonl)

Stage 6: ner
  ├── 6a ner_han           (han_sentences -> entities_han.jsonl)
  ├── 6b ner_vi            (vi_sentences -> entities_vi.jsonl)
  └── 6c ner_bridge        (match entities across pairs)

Stage 7: eval
  ├── 7a auto_metrics      (LaBSE/COMET/BERTScore/BLEU)
  ├── 7b flores_sanity     (FLORES-200 zh-vi)
  ├── 7c round_trip        (Viet -> Han via LLM)
  ├── 7d holdout_mt        (train MarianMT, eval hold-out)
  ├── 7e llm_ensemble      (Qwen2.5-7B-Instruct judge, mean-only với 1 model)
  └── export_corpus        (final hvb_corpus.jsonl)
```

## Stage 1a: normalize_han

**File:** `src/01_prep/normalize_han.py`

**Input:** `data/raw/Đại Nam Thực Lục - 大南寔錄_full.txt`

**Output:** `data/interim/han_clean.txt`

**Logic:**
1. Strip wiki markers (`姊妹计划`, `数据项`, `#`, `##`, `---`)
2. Strip wiki annotation blocks `【...】`
3. Convert full-width Latin `！-～` → half-width
4. Collapse multiple newlines
5. Strip whitespace per line, drop empty lines

**Key regex:**
- `WIKI_HEADER_RE`: match wiki header markers
- `BRACKET_RE`: match `【...】` blocks
- `FULLWIDTH_RE`: full-width char range
- `MULTI_NEWLINE_RE`: 3+ newlines → 2

**Run:**
```bash
uv run python -m src.01_prep.normalize_han
```

**Verify:**
- Output chars thường 90-95% input (do strip wiki overhead)
- First line nên là chapter title (vd: `大南寔錄前編`)

## Stage 1b: pdf_to_images

**File:** `src/01_prep/pdf_to_images.py`

**Input:** 3 PDF paths

**Output:** `data/interim/vi_pages/tap{4,5,6}_{page:04d}.png`

**Params:**
- `OCR_DPI = 300` (config.py)
- `thread_count=4` pdf2image

**Logic:**
1. Loop qua từng PDF trong `VI_PDFS`
2. `pdf2image.convert_from_path(dpi=300, thread_count=4)` → list of PIL images
3. Save mỗi page as PNG `{tap_key}_{page:04d}.png`

**Tap key extract:** detect "tập 4" / "tập 5" / "tập 6" trong filename → `tap4` / `tap5` / `tap6`

**Verify:**
- 1,141 PNGs cho tap4
- 945 PNGs cho tap5
- 1,156 PNGs cho tap6
- Total: 3,242 files

**Performance:** ~1-2 sec/page → ~1.5 giờ total

## Stage 2a: paddle_ocr

**File:** `src/02_ocr/paddle_ocr.py`

**Input:** `data/interim/vi_pages/*.png`

**Output:**
- `data/interim/vi_ocr_raw/tap{4,5,6}_page_{n:04d}.txt` — per-page
- `data/interim/vi_ocr_raw/tap{4,5,6}.txt` — concatenated per tap

**Params:**
- `PADDLE_LANG = "vietnam"`
- `PADDLE_USE_GPU = True` (auto-detect CUDA)
- `use_angle_cls=True` — handle rotated text

**Logic:**
1. Build PaddleOCR instance một lần
2. Loop qua PNGs (sorted), OCR từng image
3. Extract text từ result (bỏ bounding boxes, keep text + confidence ngầm)
4. Save per-page + accumulate per-tap

**Common issues:**
- Pages lớn có thể OOM — giảm `PADDLE_BATCH` trong config
- Nôm chars bị nhầm → cần TrOCR fine-tune (Stage 2a.optional)
- Rotated pages → giữ `use_angle_cls=True`

**Verify:** Spot-check 5-10 pages random so với PDF gốc. Kỳ vọng CER 8-15% (sau Stage 2b xuống < 5%).

## Stage 2b: llm_correct

**File:** `src/02_ocr/llm_correct.py`

**Input:** `data/interim/vi_ocr_raw/tap*.txt`

**Output:** `data/interim/vi_ocr_corrected/tap*.txt`

**Params:**
- `VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"` — single model
- `LLM_MODELS = [VLLM_MODEL]` — backward-compat alias, mặc định dùng `[0]`
- `VLLM_BASE_URL = "http://localhost:8001/v1"`
- `LLM_TIMEOUT = 180` seconds
- Chunk size: 500 chars

**Logic:**
1. Load raw OCR text
2. Chunk by paragraphs (target ~500 chars each, không cắt giữa paragraph)
3. Cho mỗi chunk: prompt LLM sửa OCR errors
4. Ghép lại các chunks corrected

**Prompt template (trong code):**
- Hướng LLM về lỗi phổ biến: Nôm chars, dấu câu cổ, dấu thanh Việt
- Output: chỉ text đã sửa, không giải thích
- Temperature thấp (0.2) để giảm hallucination

**Run:**
```bash
# Default model
uv run python -m src.02_ocr.llm_correct

# Specific model
uv run python -m src.02_ocr.llm_correct --model Qwen/Qwen2.5-7B-Instruct
```

**Performance:** ~2-3 sec/chunk × ~6500 chunks = ~5 giờ

**Verify:**
- CER trên 10 gold pages < 5%
- So sánh raw vs corrected: giảm obvious OCR errors (vd `0` → `o`)

## Stage 3a: split_han

**File:** `src/03_split/split_han.py`

**Input:** `data/interim/han_clean.txt`

**Output:** `data/interim/han_sentences.jsonl`

**Logic:**
1. Split text theo paragraphs (`\n\n`)
2. Mỗi paragraph: protect annotation brackets `〈...〉「...」『...』（...）` bằng placeholder
3. Split trên classical terminators: `。！？；`
4. Restore annotations
5. Strip + filter empty

**HanLP integration:** Currently dùng custom regex (faster, không cần model download). Có thể switch sang HanLP `sentence_split` nếu muốn.

**Verify:**
- ~50K-80K sentences (tùy content)
- Avg len: 30-60 chars/sentence
- Không có sentence > 500 chars (over-long = missing terminator)

## Stage 3b: split_vi

**File:** `src/03_split/split_vi.py`

**Input:** `data/interim/vi_ocr_corrected/tap*_page_*.txt`

**Output:** `data/interim/vi_sentences.jsonl` với fields `{idx, tap, page, text}`

**Logic:**
1. Loop qua per-page files (corrected)
2. Split theo `\n\n` paragraphs
3. Pre-split trên classical carryovers `。；！？`
4. Mỗi chunk: `underthesea.sent_tokenize` (fallback regex nếu underthesea fail)
5. Filter sentences len >= 2

**Verify:**
- Total sentences thường 80%-110% Han count (1-1 với drift)
- Page field populate đúng

## Stage 4: labse_embed

**File:** `src/04_embed/labse_embed.py`

**Input:** `data/interim/{han,vi}_sentences.jsonl`

**Output:** `data/interim/{han,vi}_embeds.npy`

**Params:**
- `LABSE_MODEL = "sentence-transformers/LaBSE"`
- `EMBED_BATCH = 64`
- `max_seq_length = 256` (truncate dài hơn)
- `normalize_embeddings = True` (cosine = dot product)

**Logic:**
1. Load sentences from JSONL
2. Encode batch trên GPU
3. Save numpy `.npy` float32

**Verify:**
- `han_embeds.npy` shape = `(N_han, 768)`
- `vi_embeds.npy` shape = `(N_vi, 768)`
- L2 norm mỗi row = 1.0

**Performance:** ~30-45 min cho ~100K sentences.

## Stage 5: vecalign_runner

**File:** `src/05_align/vecalign_runner.py`

**Input:**
- `data/interim/han_sentences.jsonl`
- `data/interim/vi_sentences.jsonl`
- `data/interim/han_embeds.npy`
- `data/interim/vi_embeds.npy`
- `external/vecalign/vecalign.py` (git clone)

**Output:** `data/aligned/pairs.jsonl`

**Params:**
- `ALIGN_MIN_SCORE = 0.5` (filter threshold)

**Logic:**
1. Prep temp files: 1 sentence per line (Vecalign format)
2. Subprocess call `vecalign.py` với:
   - `--src`, `--tgt`: sentence files
   - `--src-embeddings`, `--tgt-embeddings`: `.npy` files (no extension)
   - `--embeddings-format labse`
   - `--output`: alignment file
3. Parse Vecalign output format `: <score>\t<src_idx>\t<tgt_idx>`
4. Expand range indices (`0--2` → `[0,1,2]`)
5. Join text from sentences, write `pairs.jsonl`

**Vecalign output format:**
```
: 1.0  0  0      (sentence 0 = sentence 0, score 1.0)
: 0.83 1  1
: 0.71 2--3 2    (Han sentences 2,3 = Viet sentence 2)
```

**Verify:**
- Score distribution: mode ở 0.7-0.9 (good aligns)
- 1-1 alignments là phổ biến (>80%)
- 1-many / many-1 < 20%

**Performance:** ~1-2 giờ.

## Stage 6a/b/c: NER

**Files:**
- `src/06_ner/ner_han.py` (HanLP)
- `src/06_ner/ner_vi.py` (Underthesea)
- `src/06_ner/ner_bridge.py` (match)

**Input:** Sentence JSONLs + pairs.jsonl

**Output:**
- `data/interim/entities_han.jsonl`
- `data/interim/entities_vi.jsonl`
- `data/aligned/entities.jsonl`

**Logic ner_bridge:**
1. Index entities theo sentence idx
2. Loop pairs: collect entities từ src_idxs + tgt_idxs
3. Transliterate Han entity → approximate Sino-Viet (qua `HANVIET_MAP`)
4. Match với Viet entity nếu substring match
5. Tính NER-Bridge coverage = pairs có match / pairs có entity

**Sino-Viet map (`HANVIET_MAP`):** ~50 chars phổ biến (vua, quan, địa danh). Mở rộng trong code nếu cần.

**Verify:**
- HanLP NER tìm được PER/LOC/TIME cho tên vua, địa danh, niên hiệu
- Underthesea NER hoạt động tốt cho tên người VN, địa danh
- Coverage kỳ vọng > 50% (không 100% do transliteration approx)

## Stage 7: Eval suite

Chi tiết methodology xem [`04_eval.md`](04_eval.md).

## Stage 8: export_corpus

**File:** `src/07_eval/export_corpus.py`

**Input:** `data/aligned/pairs.jsonl` + `data/aligned/entities.jsonl`

**Output:** `data/final/hvb_corpus.jsonl`

**Logic:** Merge pairs với entities (lookup bằng `(src_idx, tgt_idx)` key). Final record schema:
```json
{
  "src": "...", "tgt": "...",
  "src_idx": [42], "tgt_idx": [58],
  "labse_score": 0.82,
  "entities": [{"han": "...", "vi": "...", "score": 1.0}]
}
```

## Checkpoint resume

`scripts/run_pipeline.sh` ghi checkpoint file tại `data/interim/.checkpoint/{stage_name}` sau mỗi stage thành công. Re-run script:
- Nếu checkpoint tồn tại + stage không phải `all`: skip
- Nếu stage = `all`: chạy hết

Force re-run: xóa checkpoint file tương ứng rồi gọi lại stage.

## Performance budgets

| Stage | Wall-clock (1x RTX 3060) | VRAM peak |
|-------|--------------------------|-----------|
| 1a normalize_han | 30 sec | 0 |
| 1b pdf_to_images | 1.5 giờ | 0 |
| 2a paddle_ocr | 3-4 giờ | ~4 GB |
| 2b llm_correct | 5 giờ | ~6 GB (vLLM) |
| 3a/b split | 5-15 min | 0 |
| 4 labse_embed | 30-45 min | ~3 GB |
| 5 vecalign | 1-2 giờ | ~2 GB |
| 6 NER | 30 min - 1 giờ | ~3 GB |
| 7a auto_metrics | 30 min - 1 giờ | ~5 GB (COMET) |
| 7b flores | 5-10 min | ~5 GB |
| 7c round_trip | 1-2 giờ | 6 GB |
| 7d holdout_mt | 6-12 giờ | ~8 GB |
| 7e llm_ensemble | 3-5 giờ | ~6 GB |
| 8 export | 1 min | 0 |

**Total: ~2-3 ngày GPU.**
