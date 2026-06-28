"""Central config: paths, model IDs, device, batch sizes."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"

# Subset mode: HVB_SUBSET=N selects N pages of tap4 + limited Hán chars.
# HVB_SUBSET_OFFSET=K skips first K pages (avoid cover/blank front matter).
# Outputs go to data/{interim,aligned,final}_subN[_offK]/ to keep full-run clean.
SUBSET_N = int(os.environ.get("HVB_SUBSET", "0") or "0")
SUBSET_OFFSET = int(os.environ.get("HVB_SUBSET_OFFSET", "0") or "0")
SUBSET_HAN_CHARS = 500 * SUBSET_N  # rough: 500 Han chars per Viet page
SUBSET_HAN_OFFSET = 500 * SUBSET_OFFSET

if SUBSET_N > 0:
    tag = f"sub{SUBSET_N}" if SUBSET_OFFSET == 0 else f"sub{SUBSET_N}_off{SUBSET_OFFSET}"
    INTERIM = DATA / f"interim_{tag}"
    ALIGNED = DATA / f"aligned_{tag}"
    FINAL = DATA / f"final_{tag}"
else:
    INTERIM = DATA / "interim"
    ALIGNED = DATA / "aligned"
    FINAL = DATA / "final"

GOLD = DATA / "gold"
BENCHMARK = DATA / "benchmark"

HAN_TXT = RAW / "Đại Nam Thực Lục - 大南寔錄_full.txt"
VI_PDF_TAP4 = RAW / "Đại Nam Thực Lục tập 4 - Quốc Sử Quán Triều Nguyễn.pdf"
VI_PDF_TAP5 = RAW / "Đại Nam Thực Lục tập 5 - Quốc Sử Quán Triều Nguyễn.pdf"
VI_PDF_TAP6 = RAW / "Đại Nam Thực Lục tập 6 - Quốc Sử Quán Triều Nguyễn.pdf"
VI_PDFS = [VI_PDF_TAP4, VI_PDF_TAP5, VI_PDF_TAP6]
if SUBSET_N > 0:
    VI_PDFS = [VI_PDF_TAP4]

HAN_CLEAN = INTERIM / "han_clean.txt"
VI_PAGES_DIR = INTERIM / "vi_pages"
VI_OCR_RAW_DIR = INTERIM / "vi_ocr_raw"
VI_OCR_CORRECTED_DIR = INTERIM / "vi_ocr_corrected"
HAN_SENT = INTERIM / "han_sentences.jsonl"
VI_SENT = INTERIM / "vi_sentences.jsonl"
HAN_EMBEDS = INTERIM / "han_embeds.npy"
VI_EMBEDS = INTERIM / "vi_embeds.npy"

PAIRS_JSONL = ALIGNED / "pairs.jsonl"
ENTITIES_JSONL = ALIGNED / "entities.jsonl"
FINAL_CORPUS = FINAL / "hvb_corpus.jsonl"

FLORES_DIR = BENCHMARK / "flores_zh_vi"
OPUS_DIR = BENCHMARK / "opus_zh_vi"

OCR_GOLD = GOLD / "ocr_gold"
NER_GOLD = GOLD / "ner_gold"

import torch  # noqa: E402

CUDA_VISIBLE = torch.cuda.device_count()
DEVICE = "cuda" if CUDA_VISIBLE > 0 else "cpu"

PADDLE_LANG = "vi"
PADDLE_USE_GPU = CUDA_VISIBLE > 0
OCR_DPI = 300

LABSE_MODEL = "sentence-transformers/LaBSE"
EMBED_BATCH = 64

VECALIGN_REPO = ROOT / "external" / "vecalign"
ALIGN_MIN_SCORE = float(os.environ.get("ALIGN_MIN_SCORE", "0.5"))

HANLP_MODEL = "CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH"
PHONER_MODEL = "NlpHUST/ner-vietnamese-se-lstm"

# LLM backend: vLLM (OpenAI-compatible) — faster than Ollama via PagedAttention.
# Single Qwen2.5-7B-Instruct serves OCR correct + round-trip + judge.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")  # vLLM ignores key, OpenAI client requires non-empty
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LLM_MODELS = [VLLM_MODEL]  # backward-compat alias for downstream stages
LLM_TIMEOUT = 180
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 4096

# Optional LLM post-correction (Stage 2b).
# Set HVB_SKIP_LLM_CORRECT=1 to copy raw OCR → corrected dir and skip vLLM call.
# Useful when vLLM unavailable or for fast smoke tests on clean OCR.
SKIP_LLM_CORRECT = bool(os.environ.get("HVB_SKIP_LLM_CORRECT", ""))

COMET_MODEL = "unmt/comet-qe-22"
HOLDOUT_RATIO = 0.2
HOLDOUT_MIN_PAIRS = 5000
EVAL_SAMPLE = 500
LLM_JUDGE_RUBRIC = ["adequacy", "fluency", "alignment", "fidelity", "terminology"]

PADDLE_BATCH = 8
MT_BATCH = 16


def ensure_dirs() -> None:
    for d in [
        RAW, INTERIM, ALIGNED, GOLD, BENCHMARK, FINAL,
        VI_PAGES_DIR, VI_OCR_RAW_DIR, VI_OCR_CORRECTED_DIR,
        OCR_GOLD, NER_GOLD, FLORES_DIR, OPUS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"ROOT={ROOT}")
    print(f"DEVICE={DEVICE} CUDA_VISIBLE={CUDA_VISIBLE}")
    print(f"HAN_TXT exists: {HAN_TXT.exists()}")
    for p in VI_PDFS:
        print(f"  {p.name}: {p.exists()}")
