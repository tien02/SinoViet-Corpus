# HVB — Han-Viet Parallel Corpus (Đại Nam Thực Lục)

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-0.10-purple)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Pipeline xây dựng corpus song ngữ Hán-Việt từ **Đại Nam Thực Lục** (Nguyễn dynasty royal annals), phục vụ NLP nghiên cứu lịch sử Việt Nam.

## Tổng quan

**Input:**
- Hán TXT: `data/raw/Đại Nam Thực Lục - 大南寔錄_full.txt` — 233K dòng / 4.9M chars (Wiki文库)
- 3 PDF Việt: `data/raw/Đại Nam Thực Lục tập {4,5,6} - Quốc Sử Quán Triều Nguyễn.pdf` — 3,242 trang

**Output (deliverable):**
- `data/final/{prefix}_parallel.tsv` + `.xlsx` — cặp câu dóng hàng: `pair_id ⇥ han_sentence ⇥ viet_sentence`
- `data/final/{prefix}_raw.txt` — OCR thô (Việt). `{prefix}` = mã số sinh viên qua `HVB_DELIVERABLE_PREFIX`.

## Kiến trúc pipeline

```
[Han TXT] → Normalize ──┐
                         ├─→ Sent Split ─┐
[3 Viet PDFs] → PNG → OCR (PaddleOCR + LLM optional) ─→ Sent Split ─┤
                                                            ├─→ LaBSE Embed
                                                            │     ↓
                                                            │  Vecalign
                                                            │     ↓
                                                            │  pairs.jsonl
                                                            │     ↓
                                                            └─→ Export deliverable
                                                               (raw.txt + parallel.tsv/.xlsx)
```

## Quickstart

### Yêu cầu hệ thống
- Python 3.11
- GPU NVIDIA (đã test RTX 3060 12GB x2)
- Docker (cho vLLM — OpenAI-compatible serving)
- `poppler-utils` (cho pdf2image)
- `git` (clone vecalign)

### Cài đặt

```bash
# Mighty one-button setup (pre-flight checks + installs everything):
./scripts/setup.sh

# Hoặc chỉ check deps mà không install:
./scripts/setup.sh --check
```

`scripts/setup.sh` sẽ:
1. Pre-flight checks (warn-only, không auto-install)
2. Tạo `uv venv` Python 3.11 + `uv sync` deps từ `pyproject.toml`
3. Clone `external/vecalign/` từ `thompsonb/vecalign` (fork live — `neulab/vecalign` đã 404)
4. Verify NVIDIA GPU
5. (Optional) `--with-vllm` để start vLLM docker — mặc định skip, pipeline chạy không cần LLM

### Chạy pipeline

```bash
# Toàn bộ pipeline (3-5 ngày GPU tùy OCR)
./scripts/run_pipeline.sh all

# Theo stage
./scripts/run_pipeline.sh prep    # Stage 1: normalize Han + PDF → PNG
./scripts/run_pipeline.sh ocr     # Stage 2: PaddleOCR + vLLM fix (or HVB_SKIP_LLM_CORRECT=1)
./scripts/run_pipeline.sh split   # Stage 3: sentence split
./scripts/run_pipeline.sh embed   # Stage 4: LaBSE
./scripts/run_pipeline.sh align   # Stage 5: Vecalign
./scripts/run_pipeline.sh export  # Stage 7: build deliverable (raw.txt + parallel.tsv/.xlsx)
```

Set mã số sinh viên cho tên file deliverable: `HVB_DELIVERABLE_PREFIX=<mssv> ./scripts/run_pipeline.sh export`

### Smoke test (subset)

Verify pipeline mechanics trên subset nhỏ trước khi chạy full. Subset mode qua env vars — không cần sửa code.

```bash
# Subset 10 trang đầu của tap4 (cover pages, OCR noise — mechanics only)
HVB_SUBSET=10 ./scripts/run_pipeline.sh prep
HVB_SUBSET=10 ./scripts/run_pipeline.sh ocr
HVB_SUBSET=10 ./scripts/run_pipeline.sh split
HVB_SUBSET=10 ./scripts/run_pipeline.sh embed
HVB_SUBSET=10 ./scripts/run_pipeline.sh align

# Subset 10 trang bỏ qua 50 trang đầu (skip cover, có nội dung thực)
HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh prep
HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh ocr
HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh split
HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh embed
HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh align
```

Output đi vào `data/interim_sub{N}/`, `data/aligned_sub{N}/` — tách biệt khỏi `data/interim/` của full run.

Env vars:
- `HVB_SUBSET=N` — số trang tap4 cần xử lý (mặc định 0 = full)
- `HVB_SUBSET_OFFSET=K` — bỏ qua K trang đầu (mặc định 0)
- `ALIGN_MIN_SCORE=X` — ngưỡng score giữ cặp aligned (mặc định 0.5)
- `HVB_KEEP_TMP=1` — giữ `data/aligned_*/_vecalign_tmp/` để debug

Inspect kết quả:
```bash
wc -l data/aligned_sub10_off50/pairs.jsonl
head -5 data/aligned_sub10_off50/pairs.jsonl | python -c "
import json, sys
for line in sys.stdin:
    p = json.loads(line)
    print(f'score={p[\"score\"]:.3f}')
    print(f'  Han:  {p[\"src\"][:80]}')
    print(f'  Viet: {p[\"tgt\"][:80]}')
"
```

## Deliverable (export)

Stage `export` (`src/07_export/export_deliverable.py`) đọc `pairs.jsonl` + OCR thô và sinh 3 file trong `data/final/`:

| File | Nội dung |
|------|----------|
| `{prefix}_raw.txt` | OCR thô Việt (tap4 + tap5 + tap6) |
| `{prefix}_parallel.tsv` | `pair_id ⇥ han_sentence ⇥ viet_sentence` |
| `{prefix}_parallel.xlsx` | 3 cột như trên, định dạng Excel |

Cặp dị thường (Hán/Việt > `HVB_MAX_PAIR_CHARS`, mặc định 2000 — do Vecalign range-merge) bị loại để giữ chất lượng và vừa giới hạn ô Excel.

## Cấu trúc project

```
NLP/
├── data/
│   ├── raw/                      # Input PDFs + TXT (không chỉnh sửa)
│   ├── interim/                  # Trung gian: han_clean.txt, vi_pages/, ocr/, sentences/, embeds/
│   ├── aligned/                  # pairs.jsonl
│   ├── gold/                     # Gold standard (OCR gold)
│   └── final/                    # Deliverable: {prefix}_raw.txt + {prefix}_parallel.tsv/.xlsx
├── external/
│   └── vecalign/                 # Git clone của thompsonb/vecalign
├── src/
│   ├── utils/config.py           # Tất cả paths, model IDs, hyperparams
│   ├── 01_prep/                  # normalize_han, pdf_to_images
│   ├── 02_ocr/                   # paddle_ocr, llm_correct (vLLM, optional via HVB_SKIP_LLM_CORRECT)
│   ├── 03_split/                 # split_han, split_vi
│   ├── 04_embed/                 # labse_embed
│   ├── 05_align/                 # vecalign_runner
│   └── 07_export/                # export_deliverable
├── scripts/
│   ├── setup.sh                  # Cài môi trường (uv venv + vecalign; vLLM optional)
│   └── run_pipeline.sh           # Runner có checkpoint
├── docs/                         # Hướng dẫn chi tiết (xem docs/README.md)
├── pyproject.toml                # uv-managed deps
├── CLAUDE.md                     # Project context cho Claude Code
└── README.md                     # File này
```

## Tính năng chính

- **OCR đa GPU**: PaddleOCR (GPU) shard qua nhiều GPU (`HVB_OCR_GPUS`), det cap `HVB_DET_SIDE_LEN` chống OOM. Post-fix bằng local LLM (Qwen2.5-7B-Instruct qua vLLM docker) — optional, set `HVB_SKIP_LLM_CORRECT=1` để bypass.
- **Dóng hàng đa ngữ**: LaBSE embeddings + Vecalign dynamic programming
- **Export deliverable**: TSV + XLSX (`pair_id ⇥ han ⇥ viet`) + OCR thô, lọc cặp dị thường
- **Checkpoint**: mỗi stage ghi checkpoint, có thể resume
- **Reproducible**: pin versions trong `pyproject.toml`, paths centralized trong `config.py`

## Hạn chế

- LLM local (Qwen2.5-7B-Instruct) chất lượng thấp hơn GPT-4o cho văn cổ → kỳ vọng 10-15% OCR error còn sót. Bỏ qua LLM correct thì mất nhiều dấu thanh Việt, giảm chất lượng dóng hàng.
- Vecalign giả định monotonic alignment — nếu thứ tự PDF khác TXT cần segment theo chapter trước (triệu chứng: cặp Hán dài bất thường, bị lọc khi export)

## Tài liệu chi tiết

- [`docs/00_problem.md`](docs/00_problem.md) — **đọc đầu tiên** — bài toán, pipeline lý do, chiến lược đánh giá
- [`docs/01_setup.md`](docs/01_setup.md) — Cài đặt chi tiết + troubleshooting env
- [`docs/02_data.md`](docs/02_data.md) — Spec input data + định dạng output
- [`docs/03_pipeline.md`](docs/03_pipeline.md) — Giải thích từng stage + code flow
- [`docs/05_troubleshooting.md`](docs/05_troubleshooting.md) — Lỗi thường gặp + fix
- [`docs/06_extend.md`](docs/06_extend.md) — Mở rộng: thêm PDF, đổi model
- [`CLAUDE.md`](CLAUDE.md) — Context cho Claude Code sessions

## License

- **Code:** MIT (xem [`LICENSE`](LICENSE))
- **Data** (PDFs, TXT, derived corpus under `data/`): CC-BY-NC-4.0 (xem [`DATA_LICENSE`](DATA_LICENSE)) — academic/non-commercial use only.

Source material: **Đại Nam Thực Lục** (大南寔錄) by Quốc Sử Quán Triều Nguyễn. Digitized PDFs © respective rights holders.
