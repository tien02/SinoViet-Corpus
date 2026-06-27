# HVB — Han-Viet Parallel Corpus (Đại Nam Thực Lục)

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-0.10-purple)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Pipeline xây dựng corpus song ngữ Hán-Việt từ **Đại Nam Thực Lục** (Nguyễn dynasty royal annals), phục vụ NLP nghiên cứu lịch sử Việt Nam.

## Tổng quan

**Input:**
- Hán TXT: `data/raw/Đại Nam Thực Lục - 大南寔錄_full.txt` — 233K dòng / 4.9M chars (Wiki文库)
- 3 PDF Việt: `data/raw/Đại Nam Thực Lục tập {4,5,6} - Quốc Sử Quán Triều Nguyễn.pdf` — 3,242 trang

**Output:**
- `data/final/hvb_corpus.jsonl` — cặp câu Hán-Việt dóng hàng, kèm điểm LaBSE + entities
- `data/final/eval/*.json` — báo cáo đánh giá tự động (5 trụ)

## Kiến trúc pipeline

```
[Han TXT] → Normalize ──┐
                         ├─→ Sent Split ─┐
[3 Viet PDFs] → PNG → OCR (PaddleOCR + LLM) ─→ Sent Split ─┤
                                                            ├─→ LaBSE Embed
                                                            │     ↓
                                                            │  Vecalign
                                                            │     ↓
                                                            │  pairs.jsonl
                                                            │     ↓
                                                            ├─→ NER (HanLP + Underthesea) ─→ entities.jsonl
                                                            │     ↓
                                                            └─→ Export hvb_corpus.jsonl
                                                                  ↓
                                                              Evaluation
                                                  (auto + FLORES + round-trip
                                                   + hold-out MT + LLM ensemble)
```

## Quickstart

### Yêu cầu hệ thống
- Python 3.11
- GPU NVIDIA (đã test RTX 3060 12GB x2)
- Docker (cho Ollama)
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
1. Pre-flight + auto-install: Python 3.11, uv, git-lfs, docker, poppler-utils
2. Tạo `uv venv` Python 3.11 + `uv sync` deps từ `pyproject.toml`
3. Clone `external/vecalign/` từ `thompsonb/vecalign` (fork live — `neulab/vecalign` đã 404)
4. Start Ollama docker container (GPU-enabled, tên container: `ollama`)
5. Pull 2 models: `qwen2.5:7b`, `seallm:7b` (~5GB VRAM mỗi model khi chạy)
6. Verify NVIDIA GPU

Lệnh cũ `scripts/setup_uv.sh` vẫn hoạt động (subset của `setup.sh`).

### Chạy pipeline

```bash
# Toàn bộ pipeline (3-5 ngày GPU tùy OCR)
./scripts/run_pipeline.sh all

# Theo stage
./scripts/run_pipeline.sh prep    # Stage 1: normalize Han + PDF → PNG
./scripts/run_pipeline.sh ocr     # Stage 2: PaddleOCR + Ollama fix
./scripts/run_pipeline.sh split   # Stage 3: sentence split
./scripts/run_pipeline.sh embed   # Stage 4: LaBSE
./scripts/run_pipeline.sh align   # Stage 5: Vecalign
./scripts/run_pipeline.sh ner     # Stage 6: NER + bridge
./scripts/run_pipeline.sh eval    # Stage 7: 5 eval modules + export
```

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

## Đánh giá chất lượng (Full-auto, 5 trụ)

| Trụ | Module | Mô tả |
|-----|--------|-------|
| 7a | `auto_metrics.py` | LaBSE cosine, COMET-QE-22, BERTScore, BLEU/chrF bi-directional |
| 7b | `flores_sanity.py` | FLORES-200 zh-vi: precision@1, COMET (sanity check, NOT domain match) |
| 7c | `round_trip.py` | Việt→Hán (LLM), chrF/BLEU vs Hán gốc (500 cặp stratified) |
| 7d | `holdout_mt.py` | Train MarianMT zh-vi 80/20, BLEU/chrF trên hold-out |
| 7e | `llm_ensemble_judge.py` | Qwen2.5 + SeaLLM chấm 1-5 (500 cặp), Krippendorff α |

Target metrics (xem `docs/04_eval.md`):

| Metric | Target |
|--------|--------|
| LaBSE cosine mean | > 0.6 |
| COMET-QE mean | > 0.5 |
| Round-trip chrF | > 0.4 |
| Hold-out MT BLEU | > 15 (skip nếu < 5000 pairs) |
| Krippendorff α (LLM) | ≥ 0.5 |
| NER-Bridge F1 | > 0.75 |

## Cấu trúc project

```
NLP/
├── data/
│   ├── raw/                      # Input PDFs + TXT (không chỉnh sửa)
│   ├── interim/                  # Trung gian: han_clean.txt, vi_pages/, ocr/, sentences/, embeds/
│   ├── aligned/                  # pairs.jsonl + entities.jsonl
│   ├── gold/                     # Gold standard (OCR gold, NER gold)
│   ├── benchmark/                # FLORES-200, OPUS (zh-vi)
│   └── final/                    # Output cuối: hvb_corpus.jsonl + eval/
├── external/
│   └── vecalign/                 # Git clone của thompsonb/vecalign
├── src/
│   ├── utils/config.py           # Tất cả paths, model IDs, hyperparams
│   ├── 01_prep/                  # normalize_han, pdf_to_images
│   ├── 02_ocr/                   # paddle_ocr, llm_correct (Ollama)
│   ├── 03_split/                 # split_han, split_vi
│   ├── 04_embed/                 # labse_embed
│   ├── 05_align/                 # vecalign_runner
│   ├── 06_ner/                   # ner_han, ner_vi, ner_bridge
│   └── 07_eval/                  # auto_metrics, flores, round_trip, holdout_mt, llm_ensemble, export
├── scripts/
│   ├── setup_uv.sh               # Cài môi trường
│   └── run_pipeline.sh           # Runner có checkpoint
├── docs/                         # Hướng dẫn chi tiết (xem docs/README.md)
├── pyproject.toml                # uv-managed deps
├── CLAUDE.md                     # Project context cho Claude Code
└── README.md                     # File này
```

## Tính năng chính

- **Đa mô hình OCR**: PaddleOCR (GPU) + post-fix bằng local LLM (Qwen2.5/SeaLLM qua Ollama docker)
- **Dóng hàng đa ngữ**: LaBSE embeddings + Vecalign dynamic programming
- **NER cross-lingual**: HanLP (Hán) + Underthesea (Việt) + bridge matching bằng Sino-Vietnamese transliteration
- **5-trụ đánh giá**: auto + FLORES + round-trip + hold-out MT + LLM ensemble (Krippendorff α)
- **Checkpoint**: mỗi stage ghi checkpoint, có thể resume
- **Reproducible**: pin versions trong `pyproject.toml`, paths centralized trong `config.py`

## Hạn chế

- FLORES-200 zh-vi là tiếng Trung hiện đại — chỉ sanity check pipeline, không claim đánh giá domain
- LLM local (Qwen/SeaLLM) chất lượng thấp hơn GPT-4o cho văn cổ → kỳ vọng 10-15% OCR error còn sót
- Vecalign giả định monotonic alignment — nếu thứ tự PDF khác TXT cần segment theo chapter trước
- Hold-out MT cần ≥ 5000 cặp aligned để train, nếu ít hơn sẽ tự skip

## Tài liệu chi tiết

- [`docs/00_problem.md`](docs/00_problem.md) — **đọc đầu tiên** — bài toán, pipeline lý do, chiến lược đánh giá
- [`docs/01_setup.md`](docs/01_setup.md) — Cài đặt chi tiết + troubleshooting env
- [`docs/02_data.md`](docs/02_data.md) — Spec input data + định dạng output
- [`docs/03_pipeline.md`](docs/03_pipeline.md) — Giải thích từng stage + code flow
- [`docs/04_eval.md`](docs/04_eval.md) — Methodology đánh giá 5 trụ
- [`docs/05_troubleshooting.md`](docs/05_troubleshooting.md) — Lỗi thường gặp + fix
- [`docs/06_extend.md`](docs/06_extend.md) — Mở rộng: thêm PDF, đổi model, custom eval
- [`CLAUDE.md`](CLAUDE.md) — Context cho Claude Code sessions

## License

- **Code:** MIT (xem [`LICENSE`](LICENSE))
- **Data** (PDFs, TXT, derived corpus under `data/`): CC-BY-NC-4.0 (xem [`DATA_LICENSE`](DATA_LICENSE)) — academic/non-commercial use only.

Source material: **Đại Nam Thực Lục** (大南寔錄) by Quốc Sử Quán Triều Nguyễn. Digitized PDFs © respective rights holders.
