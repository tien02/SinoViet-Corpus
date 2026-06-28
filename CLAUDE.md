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

**Output:** `data/final/hvb_corpus.jsonl` — aligned sentence pairs with LaBSE scores + cross-lingual entities.

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
./scripts/run_pipeline.sh ocr      # Stage 2: PaddleOCR + LLM correct
./scripts/run_pipeline.sh split    # Stage 3: sentence split both sides
./scripts/run_pipeline.sh embed    # Stage 4: LaBSE embeddings
./scripts/run_pipeline.sh align    # Stage 5: Vecalign
./scripts/run_pipeline.sh ner      # Stage 6: HanLP + Underthesea + bridge
./scripts/run_pipeline.sh eval     # Stage 7: 5-pillar eval + export

# Or all stages
./scripts/run_pipeline.sh all

# Force re-run a stage
rm data/interim/.checkpoint/<stage_name>
./scripts/run_pipeline.sh <stage>
```

## Architecture decisions (do not change without user approval)

1. **No human annotation.** Evaluation is full-auto with 5 pillars:
   - Auto metrics (LaBSE / COMET-QE / BERTScore / BLEU / chrF)
   - FLORES-200 sanity check (NOT domain match — pipeline correctness only)
   - Round-trip consistency (Viet → Han via LLM, compare to original)
   - Internal hold-out MT (80/20 split, fine-tune MarianMT, BLEU on hold-out)
   - LLM ensemble judge (Qwen2.5-7B-Instruct; α undefined với 1 model — report mean-only, α needs ≥ 2 in `LLM_MODELS`)
2. **Local LLM via vLLM Docker** (OpenAI-compatible, never cloud APIs like GPT-4o / Claude API).
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
- **Stratified sampling** (3 buckets by LaBSE score: low < 0.5 < mid < 0.7 < high) for eval subsets.

## Stage dependencies

```
prep ──► ocr ──► split ──► embed ──► align ──► ner ──► eval
                                              └──► export
```

Each stage reads previous stage's output from `data/interim/` or `data/aligned/`. Re-running a stage without re-running upstream stages is safe (checkpoint pattern).

## Common pitfalls

1. **Vecalign assumes monotonic alignment.** If Hán TXT order ≠ Việt PDF order, chunk by chapter heading first.
2. **FLORES-200 is modern zh-vi** — passes pipeline sanity, NOT domain eval. Don't claim HVB quality from FLORES numbers.
3. **LLM ensemble α target is 0.5**, not 0.7 like human κ. LLMs share biases, lower agreement expected.
4. **OCR Nôm characters may need TrOCR fine-tune** if PaddleOCR CER > 15% on gold pages.
5. **Round-trip Viet → Han LLM** is weaker than Han → Viet; expect chrF 0.4 not 0.7.
6. **Hold-out MT auto-skips if < 5000 pairs** — check `HOLDOUT_MIN_PAIRS` in config.
7. **Underthesea sent_tokenize** can crash on empty / whitespace input — wrapped in try/except with regex fallback in `split_vi.py`.
8. **vLLM model weights cached** in docker volume `vllm` (`/root/.cache/huggingface`). Container auto-restarts via `--restart unless-stopped`. First-run downloads ~5GB weights (10-20 min).

## File structure

```
.
├── data/
│   ├── raw/                    # Input PDFs + TXT (read-only)
│   ├── interim/                # Stage outputs + .checkpoint/
│   ├── aligned/                # pairs.jsonl + entities.jsonl
│   ├── gold/                   # OCR gold + NER gold (manual)
│   ├── benchmark/              # FLORES-200 cache
│   └── final/                  # hvb_corpus.jsonl + eval/*.json
├── src/
│   ├── utils/config.py         # All paths + hyperparams (edit here)
│   ├── 01_prep/                # normalize_han, pdf_to_images
│   ├── 02_ocr/                 # paddle_ocr, llm_correct
│   ├── 03_split/               # split_han, split_vi
│   ├── 04_embed/               # labse_embed
│   ├── 05_align/               # vecalign_runner
│   ├── 06_ner/                 # ner_han, ner_vi, ner_bridge
│   └── 07_eval/                # auto_metrics, flores_sanity, round_trip,
│                               # holdout_mt, llm_ensemble_judge, export_corpus
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
- **Eval methodology (5 pillars):** `docs/04_eval.md`
- **Troubleshooting common errors:** `docs/05_troubleshooting.md`
- **Extension patterns (add PDF, swap model, custom eval):** `docs/06_extend.md`

## Editing rules

- **Config changes go in `src/utils/config.py`**, never inline in stage scripts.
- **When adding a new stage**: update `scripts/run_pipeline.sh` case statement + add checkpoint logic.
- **When changing model**: update both `config.py` AND `docs/06_extend.md` "Đổi models" section.
- **When changing JSONL schema**: update `docs/02_data.md` example AND downstream stages that read it.
- **Never commit** files in `data/interim/`, `data/aligned/`, `data/final/` (runtime artifacts). Only `data/raw/`, `data/gold/`, `data/benchmark/` are tracked.

## Verification after changes

- `uv run python -c "from src.utils import config; print(config.HAN_TXT, config.VI_PDFS)"` — config importable
- `uv run python -c "import paddleocr, sentence_transformers, hanlp, underthesea; print('OK')"` — deps OK
- `docker ps | grep vllm` — LLM serving; `curl http://localhost:8001/v1/models` — health-check
- `ls data/interim/.checkpoint/` — see which stages have completed

## Eval targets (after full pipeline)

| Metric | Target |
|--------|--------|
| LaBSE cosine mean | > 0.6 |
| COMET-QE mean | > 0.5 |
| FLORES precision@1 | > 0.95 |
| Round-trip chrF | > 0.40 |
| Hold-out BLEU | > 15 |
| Krippendorff α (LLM) | ≥ 0.5 |
| NER-Bridge coverage | > 50% |

If targets missed, see `docs/05_troubleshooting.md` for diagnostic patterns.
