# CLAUDE.md

Project-specific instructions for Claude Code working on **HVB — Hán-Việt Parallel Corpus (Đại Nam Thực Lục)**.

## Project context

Build sentence-aligned Hán-Việt parallel corpus from **Đại Nam Thực Lục** (Nguyễn dynasty royal annals):

- **Hán side:** `data/raw/Đại Nam Thực Lục - 大南寔錄_full.txt` — Wiki文库 digitized, 233K lines / 4.9M chars
- **Việt side:** 3 scanned PDFs (Quốc Sử Quán Triều Nguyễn):
  - Tập 4: 1,141 pages
  - Tập 5: 945 pages
  - Tập 6: 1,156 pages
  - Total: 3,242 pages

**Output (course deliverable):** `data/final/{prefix}_parallel.tsv` + `.xlsx` — aligned `pair_id ⇥ han_sentence ⇥ viet_sentence`, plus `{prefix}_raw.txt` (raw OCR). `{prefix}` = mã số sinh viên via `HVB_DELIVERABLE_PREFIX`.

## Environment

- **Package manager:** `uv` (NOT pip / conda). `pyproject.toml` is source of truth.
- **Python:** 3.11+.
- **GPU:** CUDA 12.1 wheels for PyTorch + PaddlePaddle (custom uv indexes in `pyproject.toml`).
- **LLM:** vLLM Docker container (`vllm/vllm-openai:latest`) on `http://localhost:8001/v1` (host port 8001 → container 8000). Model: `Qwen/Qwen2.5-7B-Instruct` (OpenAI-compatible API; PagedAttention 5-10x faster than Ollama). LLM post-correction optional — set `HVB_SKIP_LLM_CORRECT=1` to bypass.
- **Vecalign:** git clone at `external/vecalign/` (not on PyPI).

## Run commands

```bash
# First-time setup (uv venv + vecalign; vLLM docker optional)
./scripts/setup.sh                 # core only
./scripts/setup.sh --with-vllm     # also start vLLM docker

# Run pipeline stages
./scripts/run_pipeline.sh prep     # Stage 1: normalize Hán + PDF → PNG
./scripts/run_pipeline.sh ocr      # Stage 2: PaddleOCR (+ optional LLM correct)
./scripts/run_pipeline.sh split    # Stage 3: sentence split both sides
./scripts/run_pipeline.sh embed    # Stage 4: LaBSE embeddings
./scripts/run_pipeline.sh align    # Stage 5: Vecalign
./scripts/run_pipeline.sh export   # Stage 7: build deliverable (raw.txt + parallel.tsv/.xlsx)

# Or all stages (prep → ocr → split → embed → align → export)
./scripts/run_pipeline.sh all

# Force re-run a stage
rm data/interim/.checkpoint/<stage_name>
./scripts/run_pipeline.sh <stage>
```

## Architecture decisions (do not change without user approval)

1. **Scope = aligned parallel corpus only.** Deliverable is the sentence-aligned
   bitext (TSV + XLSX). No NER and no automated eval stage — both were removed.
2. **Local LLM via vLLM Docker** (OpenAI-compatible, never cloud APIs like GPT-4o / Claude API). Only used for optional OCR post-correction.
3. **uv for environment**, never pip install / conda.
4. **Vecalign + LaBSE** for alignment — no other aligner without explicit approval.
5. **7B models, not 14B** — VRAM constraint (2x RTX 3060 12GB).

## Key conventions

- **All paths in `src/utils/config.py`** — never hardcode paths in stage scripts.
- **JSONL**: 1 record per line, UTF-8, `ensure_ascii=False`.
- **File naming**:
  - Tập prefix: `tap4`, `tap5`, `tap6` (ASCII, no diacritics)
  - Page zero-pad: `{page:04d}` → `tap4_0042.png`
  - Embeddings: `.npy` float32
- **Checkpoint pattern**: `data/interim/.checkpoint/{stage_name}` flag files.

## Stage dependencies

```
prep ──► ocr ──► split ──► embed ──► align ──► export
```

Each stage reads previous stage's output from `data/interim/` or `data/aligned/`. Re-running a stage without re-running upstream stages is safe (checkpoint pattern).

## Common pitfalls

1. **Vecalign assumes monotonic alignment.** If Hán TXT order ≠ Việt PDF order, chunk by chapter heading first. Symptom: a few pairs with absurd Hán length (range-merge) — `export_deliverable` drops pairs over `HVB_MAX_PAIR_CHARS` (default 2000).
2. **OCR Nôm characters may need TrOCR fine-tune** if PaddleOCR CER is high on gold pages. Raw OCR without LLM correction has heavy diacritic loss, which lowers alignment quality.
3. **PaddleOCR OOM on 300 DPI scans** — detection capped to `HVB_DET_SIDE_LEN` (default 1536px). Multi-GPU sharding via `HVB_OCR_GPUS` (default `0,1`).
4. **split_vi reads corrected OCR, falls back to raw** when LLM correction is skipped. It fails loudly on 0 sentences (never checkpoints an empty file).
5. **Underthesea sent_tokenize** can crash on empty / whitespace input — wrapped in try/except with regex fallback in `split_vi.py`.
6. **vLLM model weights cached** in docker volume `vllm` (`/root/.cache/huggingface`). Container auto-restarts via `--restart unless-stopped`. First-run downloads ~5GB weights (10-20 min).

## File structure

```
.
├── data/
│   ├── raw/                    # Input PDFs + TXT (read-only)
│   ├── interim/                # Stage outputs + .checkpoint/
│   ├── aligned/                # pairs.jsonl
│   ├── gold/                   # OCR gold (manual)
│   └── final/                  # {prefix}_raw.txt + {prefix}_parallel.tsv/.xlsx
├── src/
│   ├── utils/config.py         # All paths + hyperparams (edit here)
│   ├── 01_prep/                # normalize_han, pdf_to_images
│   ├── 02_ocr/                 # paddle_ocr, llm_correct
│   ├── 03_split/               # split_han, split_vi
│   ├── 04_embed/               # labse_embed
│   ├── 05_align/               # vecalign_runner
│   └── 07_export/              # export_deliverable
├── external/vecalign/          # git clone
├── scripts/
│   ├── setup.sh                # First-time install (uv venv + vecalign; vLLM optional)
│   └── run_pipeline.sh         # Stage runner
├── docs/                       # Detailed guides (see docs/README.md)
├── pyproject.toml              # uv deps
├── README.md                   # Project overview
└── CLAUDE.md                   # This file
```

## Documentation pointers

- **Setup / install:** `docs/01_setup.md`
- **Data schemas (input + output):** `docs/02_data.md`
- **Pipeline stage details:** `docs/03_pipeline.md`
- **Troubleshooting common errors:** `docs/05_troubleshooting.md`
- **Extension patterns (add PDF, swap model):** `docs/06_extend.md`

> Note: NER and the 5-pillar automated eval were removed from scope. Any
> `docs/04_eval.md` or eval references in other docs are stale.

## Editing rules

- **Config changes go in `src/utils/config.py`**, never inline in stage scripts.
- **When adding a new stage**: update `scripts/run_pipeline.sh` case statement + add checkpoint logic.
- **When changing model**: update both `config.py` AND `docs/06_extend.md` "Đổi models" section.
- **When changing the deliverable schema**: update `export_deliverable.py` AND `docs/02_data.md`.
- **Never commit** files in `data/interim/`, `data/aligned/`, `data/final/` (runtime artifacts). Only `data/raw/`, `data/gold/` are tracked.

## Verification after changes

- `uv run python -c "from src.utils import config; print(config.HAN_TXT, config.VI_PDFS)"` — config importable
- `uv run python -c "import paddleocr, sentence_transformers, underthesea, openpyxl; print('OK')"` — deps OK
- `docker ps | grep vllm` — LLM serving (only needed for optional OCR correction)
- `ls data/interim/.checkpoint/` — see which stages have completed
- `./scripts/run_pipeline.sh export` then check `data/final/*_parallel.tsv` — deliverable builds
